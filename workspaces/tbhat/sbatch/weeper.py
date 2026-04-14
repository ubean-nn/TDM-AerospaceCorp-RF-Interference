import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import pytorch_lightning as pl
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

# ==========================================
# 1. UTILITIES & LAYERS
# ==========================================

class StochasticDepth(nn.Module):
    """
    Adapted from Model 1: Randomly drops paths during training 
    to promote model robustness (regularization).
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor()

class SignalAugmenter:
    """
    Encapsulates the noise logic from Model 1 into a clean transform class.
    """
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def __call__(self, x):
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

# ==========================================
# 2. DATA PIPELINE
# ==========================================

class InterferenceDataset(Dataset):
    def __init__(self, file_paths, labels, noise_std=0.0):
        self.file_paths = file_paths
        self.labels = labels
        self.augmenter = SignalAugmenter(noise_std)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # Load raw signal
            # Assuming file structure: [512, 243] or similar
            raw = np.load(self.file_paths[idx]).astype(np.float32)
            
            # Standardization (Mean/Std)
            raw = (raw - raw.mean()) / (raw.std() + 1e-8)

            # Ensure Correct Dims: [Channels, Sequence Length]
            # Flattening 2D to 1D if necessary, or keeping 2D if using Conv2d
            # Model 2 used 1D, so we ensure [1, Sequence]
            if raw.ndim == 2:
                 # Option A: Flatten huge 2D to 1D
                 # raw = raw.flatten() 
                 # Option B: Treat rows as channels (if applicable)
                 raw = np.expand_dims(raw, axis=0) # [1, H, W] if 2d or [1, Len] if 1d
            
            data_tensor = torch.from_numpy(raw)
            
            # Apply Augmentations
            data_tensor = self.augmenter(data_tensor)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return data_tensor, label
            
        except Exception as e:
            # Fallback for corrupt files
            print(f"Error loading {self.file_paths[idx]}: {e}")
            return torch.zeros((1, 512, 243)), torch.tensor(0, dtype=torch.long)

class GNSSDataModule(pl.LightningDataModule):
    def __init__(self, data_root, train_list, test_list, batch_size=32, val_split=0.15, noise_level=0.05):
        super().__init__()
        self.data_root = data_root
        self.train_list = train_list
        self.test_list = test_list
        self.batch_size = batch_size
        self.val_split = val_split
        self.noise_level = noise_level
        self.num_classes = 9

    def _parse_txt(self, txt_file):
        paths, labels = [], []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Construct full path
                    full_path = os.path.join(self.data_root, os.path.basename(parts[0]))
                    paths.append(full_path)
                    labels.append(int(parts[1]))
        return paths, labels

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            tr_paths, tr_labels = self._parse_txt(self.train_list)
            full_ds = InterferenceDataset(tr_paths, tr_labels, noise_std=self.noise_level)
            
            # Calculate split sizes
            val_len = int(len(full_ds) * self.val_split)
            train_len = len(full_ds) - val_len
            
            self.train_ds, self.val_ds = random_split(
                full_ds, [train_len, val_len], 
                generator=torch.Generator().manual_seed(42)
            )

        if stage == 'test' or stage is None:
            te_paths, te_labels = self._parse_txt(self.test_list)
            # No noise on test set
            self.test_ds = InterferenceDataset(te_paths, te_labels, noise_std=0.0)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)

# ==========================================
# 3. ARCHITECTURE (The Ensemble)
# ==========================================

class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.act = nn.GELU() # Upgrade from ReLU
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.downsample = downsample
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply Stochastic Depth before adding residual
        out = self.drop_path(out) 
        out += identity
        out = self.act(out)
        return out

# ==========================================
# 2D ResNet
# ==========================================

class ResNetBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU() 
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out

class VisionSignalNet(pl.LightningModule):
    def __init__(self, num_classes=9, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        if class_weights is not None:
            self.register_buffer("loss_weights", class_weights)
        else:
            self.loss_weights = None

        # Input is [Batch, 1, 512, 243] -> Treated as a Grayscale Image
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet18-style blocks (2D)
        # We increase channels but keep spatial dims reasonably large
        self.layer1 = self._make_layer(32, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(256, num_classes)

        # Metrics
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def _make_layer(self, in_c, out_c, stride):
        # Stack 2 blocks per layer for better feature extraction
        layers = [ResNetBlock2D(in_c, out_c, stride), ResNetBlock2D(out_c, out_c, 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is 4D: [Batch, Channel, Height, Width]
        if x.dim() == 3: 
            x = x.unsqueeze(1) 
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)

    def _calc_loss(self, logits, y):
        return F.cross_entropy(logits, y, weight=self.loss_weights)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calc_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calc_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc(preds, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calc_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("test_acc", self.acc(preds, y))
        self.conf_mat.update(preds, y)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }

    def on_test_epoch_end(self):
        # 1. Compute the final matrix from all batches
        confmat = self.conf_mat.compute().cpu()
        
        # 2. Calculate per-class accuracy (Diagonal / Row Sums)
        # (Adding 1e-6 to avoid division by zero if a class has no samples)
        class_accuracies = confmat.diag() / (confmat.sum(1) + 1e-6)
        
        print("\n" + "="*30)
        print("PER-CLASS ACCURACY REPORT")
        print("="*30)
        
        # 3. Loop through and print readable percentages
        for i, acc in enumerate(class_accuracies):
            print(f"Class {i}: {acc.item():.2%}")
            # Log to WandB so you can track improvements over time
            self.log(f"acc_class_{i}", acc.item())
        print("="*30 + "\n")

        # 4. (Optional) Visualize the Matrix in WandB
        # This creates a heatmap image and uploads it
        import matplotlib.pyplot as plt
        import seaborn as sns # Ensure seaborn is installed or use standard plt
        
        plt.figure(figsize=(10, 8))
        # Plot heatmap: annot=True shows the numbers in the boxes
        sns.heatmap(confmat.numpy(), annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        # Log the image to WandB
        if self.logger:
            self.logger.experiment.log({"confusion_matrix_img": wandb.Image(plt)})
        
        plt.close()

# ==========================================
# 4. EXECUTION
# ==========================================

def get_weights(label_file, num_classes=9):
    """Calculates inverse class weights"""
    print("Calculating class balance...", flush=True)
    counts = Counter()
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                counts[int(parts[1])] += 1
    
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)

if __name__ == "__main__":
    # --- CONFIG ---
    # Update these paths to match your Anvil environment
    ROOT_DIR = '/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/'
    DATA_PATH = os.path.join(ROOT_DIR, 'data')
    TRAIN_TXT = os.path.join(ROOT_DIR, 'train.txt')
    TEST_TXT = os.path.join(ROOT_DIR, 'test.txt')
    SCRATCH_DIR = "/anvil/scratch/x-tbhat/wandb_cache"
    
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    # --- SETUP ---
    dm = GNSSDataModule(DATA_PATH, TRAIN_TXT, TEST_TXT, batch_size=32, noise_level=0.05) 
    
    weights = get_weights(TRAIN_TXT)

    # 2. Use the new 2D Model
    model = VisionSignalNet(num_classes=9, lr=1e-3, class_weights=weights)
    
    # --- LOGGING ---
    logger = WandbLogger(
        project="gnss-interference-ensemble", 
        name="2DResNet", 
        save_dir=SCRATCH_DIR,
        log_model=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=40,
        callbacks=[
            ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, filename='best-model'),
            EarlyStopping(monitor='val_loss', patience=7),
            RichProgressBar()
        ],
        logger=logger,
        log_every_n_steps=25
    )

    # --- RUN ---
    print("Starting training...", flush=True)
    trainer.fit(model, datamodule=dm)
    
    print("Testing best model...", flush=True)
    trainer.test(model, datamodule=dm, ckpt_path="best")
    
    wandb.finish()