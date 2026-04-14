import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ASTConfig, ASTForAudioClassification
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# -----------------
# Configuration
# -----------------
ROOT_DIR = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
TRAIN_TXT = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/train.txt"
TEST_TXT = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/test.txt"

# Model hyperparameters
TARGET_TIME_LEN = 240  # Trimmed (16 * 15 = 240)
FREQ_BINS = 512
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
PATCH_SIZE = 16
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 10
VAL_SPLIT = 0.15

# IMPORTANT: Replace these with your actual dataset statistics!
GLOBAL_MEAN = -32.10111102253011 
GLOBAL_STD = 11.108406176683284  

# -----------------
# Custom Focal Loss
# -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            # Registering as a buffer automatically moves it to the correct device
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------
# Dataset
# -----------------
class RFSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels=None, root_dir=None, target_time=240, freq_bins=512, global_mean=None, global_std=None):
        self.file_paths = file_paths
        self.labels = labels
        self.root_dir = root_dir
        self.target_time = target_time
        self.freq_bins = freq_bins
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        if self.root_dir is not None:
            path = os.path.join(self.root_dir, path)
        
        try:
            spec = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e
            
        x = torch.from_numpy(spec).float()
        if x.ndim != 2: raise ValueError(f"Shape error: {path}")
        
        x = x.transpose(0, 1) 
        
        # Global Normalization
        if self.global_mean is not None and self.global_std is not None:
            x = (x - self.global_mean) / (self.global_std + 1e-8)
        else:
            # Fallback to per-sample if globals aren't provided
            mean = x.mean()
            std = x.std()
            if std > 0: x = (x - mean) / std
        
        if x.shape[0] > self.target_time:
             x = x[:self.target_time, :]
        elif x.shape[0] < self.target_time:
            pad = torch.zeros((self.target_time - x.shape[0], x.shape[1]), device=x.device)
            x = torch.cat([x, pad], dim=0)
            
        y = self.labels[idx] if self.labels is not None else 0
        
        return x, torch.tensor(y, dtype=torch.long), path

# -----------------
# Lightning DataModule
# -----------------
class RFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_txt, test_txt, batch_size=8, val_split=0.15, global_mean=None, global_std=None):
        super().__init__()
        self.data_dir = data_dir
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.batch_size = batch_size
        self.val_split = val_split
        self.global_mean = global_mean
        self.global_std = global_std

    def _load_txt(self, txt_path):
        paths, labels = [], []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                paths.append(parts[0])
                labels.append(int(parts[1]))
        return paths, labels

    def setup(self, stage=None):
        train_paths, train_labels = self._load_txt(self.train_txt)
        
        unique_labels = sorted(set(train_labels))
        self.num_classes = len(unique_labels)
        self.label2id = {l:l for l in unique_labels}
        self.id2label = {v:k for k,v in self.label2id.items()}
        
        full_train_ds = RFSpectrogramDataset(
            file_paths=train_paths, 
            labels=train_labels, 
            root_dir=self.data_dir,
            target_time=TARGET_TIME_LEN,
            freq_bins=FREQ_BINS,
            global_mean=self.global_mean,
            global_std=self.global_std
        )
        
        val_size = int(len(full_train_ds) * self.val_split)
        train_size = len(full_train_ds) - val_size
        self.train_ds, self.val_ds = random_split(
            full_train_ds, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        if self.test_txt and os.path.exists(self.test_txt):
            test_paths, test_labels = self._load_txt(self.test_txt)
            self.test_ds = RFSpectrogramDataset(
                file_paths=test_paths,
                labels=test_labels,
                root_dir=self.data_dir,
                target_time=TARGET_TIME_LEN,
                freq_bins=FREQ_BINS,
                global_mean=self.global_mean,
                global_std=self.global_std
            )
            print(f"Setup Complete: Train={train_size}, Val={val_size}, Test={len(self.test_ds)}")
        else:
            self.test_ds = None
            print(f"Setup Complete: Train={train_size}, Val={val_size}, Test=None")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,  
            num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        if self.test_ds:
            return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)
        return None

# -----------------
# Lightning Module
# -----------------
class ASTClassifier(pl.LightningModule):
    def __init__(self, num_classes, label2id, id2label, learning_rate=1e-4, alpha_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        config = ASTConfig(
            num_mel_bins=FREQ_BINS,
            max_length=TARGET_TIME_LEN,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_ATTENTION_HEADS,
            patch_size=PATCH_SIZE,
            num_labels=num_classes,
            label2id=label2id,
            id2label=id2label
        )
        
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            ignore_mismatched_sizes=True
        )
        
        self.criterion = FocalLoss(gamma=2.0, alpha=alpha_weights)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.test_preds = []
        self.test_targets = []
        self.test_paths = []

    def forward(self, x):
        return self.model(input_values=x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        outputs = self(x)
        logits = outputs.logits
        loss = self.criterion(logits, y)
        
        preds = logits.argmax(dim=-1)
        self.train_acc(preds, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        outputs = self(x)
        logits = outputs.logits
        loss = self.criterion(logits, y)
        
        preds = logits.argmax(dim=-1)
        self.val_acc(preds, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y, paths = batch
        outputs = self(x)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        self.test_paths.extend(paths)
        
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds).numpy()
        all_targets = torch.cat(self.test_targets).numpy()
        
        class_names = [str(self.hparams.id2label[i]) for i in range(self.hparams.num_classes)]
        
        # --- 9-Class Report ---
        cm = confusion_matrix(all_targets, all_preds)
        df_cm = pd.DataFrame(
            cm, 
            index=[f"True_{c}" for c in class_names], 
            columns=[f"Pred_{c}" for c in class_names]
        )

        output_dir = "/anvil/projects/x-cis220051/corporate/aerospace-rf/ine/sbatch"
        os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
        
        # Map numeric IDs back to string labels for readability
        pred_labels_str = [self.hparams.id2label[p] for p in all_preds]
        true_labels_str = [self.hparams.id2label[t] for t in all_targets]
        
        # Create DataFrame
        results_df = pd.DataFrame({
            "Sample_ID": self.test_paths,
            "True_Label_ID": all_targets,
            "True_Label_Name": true_labels_str,
            "Predicted_Label_ID": all_preds,
            "Predicted_Label_Name": pred_labels_str
        })
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "test_predictions.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved test results CSV to: {csv_path}")
        print("="*60 + "\n")
        
        # Clear lists for memory management
        self.test_preds.clear()
        self.test_targets.clear()
        self.test_paths.clear() # <-- Don't forget to clear the new list!
        
        print("\n" + "="*60)
        print("FINAL TEST PERFORMANCE REPORT (9-CLASS)")
        print("="*60)
        print("\n[CONFUSION MATRIX]")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_cm)
            
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))
        print("="*60)
        
        # --- Macro-Class Report ---
        macro_map = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:2, 8:2}
        macro_names = ["None", "Chirp", "Lighter"]
        
        macro_targets = [macro_map[t] for t in all_targets]
        macro_preds = [macro_map[p] for p in all_preds]
        
        macro_cm = confusion_matrix(macro_targets, macro_preds)
        df_macro_cm = pd.DataFrame(
            macro_cm, 
            index=[f"True_{c}" for c in macro_names], 
            columns=[f"Pred_{c}" for c in macro_names]
        )
        
        print("\n" + "="*60)
        print("MACRO TEST PERFORMANCE REPORT (3-CLASS)")
        print("="*60)
        print("\n[MACRO CONFUSION MATRIX]")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_macro_cm)
            
        print("\n[MACRO CLASSIFICATION REPORT]")
        print(classification_report(macro_targets, macro_preds, target_names=macro_names, digits=4, zero_division=0))
        print("="*60 + "\n")
        
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Added a CosineAnnealingLR scheduler to help stabilize Transformer training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# -----------------
# Main
# -----------------
def main():
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(42)
    
    # 1. Read labels directly from text file to calculate weights without calling dm.setup() early
    print("Reading labels to calculate class weights...")
    train_labels = []
    if os.path.exists(TRAIN_TXT):
        with open(TRAIN_TXT, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    train_labels.append(int(parts[1]))
    else:
        raise FileNotFoundError(f"Could not find {TRAIN_TXT}")

    unique_labels = sorted(set(train_labels))
    num_classes = len(unique_labels)
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # 2. Calculate Alpha Weights
    alpha_weights = []
    for i in range(num_classes):
        w = total_samples / (num_classes * class_counts[i])
        alpha_weights.append(min(w, 50.0)) 
        
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float)
    print(f"Calculated Alpha Weights:\n{alpha_tensor}")

    # 3. Initialize DataModule (No dm.setup() call here!)
    print("Initializing DataModule...")
    dm = RFDataModule(
        data_dir=ROOT_DIR, 
        train_txt=TRAIN_TXT, 
        test_txt=TEST_TXT,
        batch_size=BATCH_SIZE, 
        val_split=VAL_SPLIT,
        global_mean=GLOBAL_MEAN,
        global_std=GLOBAL_STD
    )
    
    # 4. Initialize Model
    label2id = {l:l for l in unique_labels}
    id2label = {v:k for k,v in label2id.items()}
    
    model = ASTClassifier(
        num_classes=num_classes,
        label2id=label2id,
        id2label=id2label,
        learning_rate=LEARNING_RATE,
        alpha_weights=alpha_tensor 
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    early_stop_callback = EarlyStopping(monitor="val_acc", patience=3, mode="max")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    print("Starting Training...")
    trainer.fit(model, dm)
    
    # Check if a test set exists before testing
    if TEST_TXT and os.path.exists(TEST_TXT):
        print("Starting Testing on Best Checkpoint...")
        trainer.test(dataloaders=dm.test_dataloader(), ckpt_path="best")
    else:
        print("Skipping testing: No test dataset found.")

if __name__ == "__main__":
    main()