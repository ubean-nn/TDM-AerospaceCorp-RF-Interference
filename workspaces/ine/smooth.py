import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import Accuracy
import numpy as np
import glob
import os
from collections import Counter
from scipy.ndimage import gaussian_filter

from pytorch_lightning.callbacks import EarlyStopping

import wandb
from pytorch_lightning.loggers import WandbLogger

# -----------------------------
# Preprocessing Functions
# -----------------------------
def gaussian_smooth(data, sigma=3):
    """Apply Gaussian smoothing to 2D data"""
    return gaussian_filter(data, sigma=sigma)


# -----------------------------
# Dataset
# -----------------------------
class RFDataset(Dataset):
    """Custom dataset for RF signal numpy files"""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load npy file
        data = np.load(self.file_paths[idx]).astype(np.float32)

        # Normalize per-sample
        data = (data - data.mean()) / (data.std() + 1e-8)

        # Apply Gaussian smoothing with sigma=3
        data = gaussian_smooth(data, sigma=3)

        # Add channel dimension (1, 512, 243)
        data = np.expand_dims(data, axis=0)

        # Convert to tensor
        data = torch.from_numpy(data)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        return data, label


# -----------------------------
# DataModule
# -----------------------------
class RFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_labels_file, test_labels_file,
                 batch_size=32, val_split=0.15):
        super().__init__()
        self.data_dir = data_dir
        self.train_labels_file = train_labels_file
        self.test_labels_file = test_labels_file
        self.batch_size = batch_size
        self.val_split = val_split  # 15% of train data for validation
        self.dims = (1, 512, 243)
        self.num_classes = 9

        self.rf_train = None
        self.rf_val = None
        self.rf_test = None

    def prepare_data(self):
        files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        print(f"Found {len(files)} .npy files in {self.data_dir}")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load full train set
            train_files, train_labels = self._load_split(self.train_labels_file)
            full_train_dataset = RFDataset(train_files, train_labels)

            # Split into train and validation
            total_train = len(full_train_dataset)
            val_size = int(self.val_split * total_train)
            train_size = total_train - val_size

            self.rf_train, self.rf_val = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )

            print(f"Train set: {train_size} samples")
            print(f"Val set: {val_size} samples")

        if stage == 'test' or stage is None:
            # Load test set
            test_files, test_labels = self._load_split(self.test_labels_file)
            self.rf_test = RFDataset(test_files, test_labels)
            print(f"Test set: {len(test_files)} samples")

    def _load_split(self, labels_file):
        """Load file paths and labels from a label file"""
        file_paths = []
        labels = []

        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    file_path = parts[0]  # e.g., "data/004809.npy"
                    label = int(parts[1])

                    # Extract filename and create full path
                    filename = os.path.basename(file_path)
                    full_path = os.path.join(self.data_dir, filename)

                    if os.path.exists(full_path):
                        file_paths.append(full_path)
                        labels.append(label)
                    else:
                        print(f"Warning: File not found: {full_path}")

        print(f"Loaded {len(file_paths)} samples from {labels_file}")
        return file_paths, labels

    def train_dataloader(self):
        return DataLoader(self.rf_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.rf_val, batch_size=self.batch_size,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.rf_test, batch_size=self.batch_size,
                          num_workers=8)


# -----------------------------
# Lightning Model
# -----------------------------
class RFLitModel(pl.LightningModule):
    """Model architecture, training, testing and validation loops"""

    def __init__(self, input_shape, num_classes,
                 learning_rate=1e-3, class_weights=None):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters(ignore=['class_weights'])
        self.learning_rate = learning_rate

        # class balanced weights (stored as buffer so not learnable)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # macro-group mapping: 0–3 -> 0 (none), 4–6 -> 1 (chirp), 7–8 -> 2 (cigarette)
        macro_map = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
        self.register_buffer("macro_map", macro_map)

        # model architecture - adapted for 512x243 input
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)  # 1 channel input
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        # Batch norm for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        n_sizes = self._get_output_shape(input_shape)

        # linear layers for classifier head
        self.fc1 = nn.Linear(n_sizes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # storage for test-time analysis
        self._test_targets = []
        self._test_preds = []

    def _get_output_shape(self, shape):
        """returns the size of the output tensor from the conv layers"""
        batch_size = 1
        inp = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._feature_extractor(inp)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _feature_extractor(self, x):
        """extract features from the conv blocks"""
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.bn1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.bn2(x)
        return x

    def forward(self, x):
        """produce final model output"""
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def _ce_loss(self, logits, y):
        """cross-entropy with optional class weights"""
        if self.class_weights is not None:
            return F.cross_entropy(logits, y, weight=self.class_weights)
        else:
            return F.cross_entropy(logits, y)

    # train loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._ce_loss(logits, y)

        # metric
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._ce_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._ce_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)

        
        self._test_targets.append(y.detach())
        self._test_preds.append(preds.detach())

        return loss

    def on_test_epoch_start(self):
        # reset storage
        self._test_targets = []
        self._test_preds = []

    def on_test_epoch_end(self):
        """Extra evaluation: macro-groups + broad-vs-exact correctness."""
        if len(self._test_targets) == 0:
            return

        y_true = torch.cat(self._test_targets)
        y_pred = torch.cat(self._test_preds)

        # exact class accuracy (should match test_acc, but computed over whole epoch)
        exact_correct = (y_true == y_pred).float().mean()

        # macro group accuracy
        macro_true = self.macro_map[y_true]
        macro_pred = self.macro_map[y_pred]
        macro_correct = (macro_true == macro_pred).float().mean()

        # fraction where broad group is right but fine class is wrong
        broad_right_exact_wrong = (
            (macro_true == macro_pred) & (y_true != y_pred)
        ).float().mean()

        self.log("test_exact_acc_epoch", exact_correct, prog_bar=False, logger=True)
        self.log("test_macro_acc", macro_correct, prog_bar=True, logger=True)
        self.log("test_broad_only_frac", broad_right_exact_wrong, prog_bar=False, logger=True)

    # optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# -----------------------------
# Helper: compute class weights
# -----------------------------
def compute_class_weights_from_file(labels_file, num_classes):
    counts = Counter()
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                _, label = parts
                label = int(label)
                counts[label] += 1

    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        if counts[c] > 0:
            weights.append(total / (num_classes * counts[c]))
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # instantiate DataModule
    dm = RFDataModule(
        data_dir='/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/data/',
        train_labels_file='/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/train.txt',
        test_labels_file='/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/test.txt',
        batch_size=32,
        val_split=0.15
    )
    dm.prepare_data()
    dm.setup()

    # compute class-balanced weights from the full train label file
    class_weights = compute_class_weights_from_file(
        dm.train_labels_file, dm.num_classes
    )

    # model with class-balanced cross-entropy + macro evaluation
    model = RFLitModel((1, 512, 243), dm.num_classes,
                       learning_rate=1e-3,
                       class_weights=class_weights)

    wandb_logger = WandbLogger(
        project="rf-signal-classification",
        name="yogurt-v2",
        log_model=True
    )
    wandb_logger.experiment.config.update({
        "batch_size": dm.batch_size,
        "val_split": dm.val_split,
        "learning_rate": model.learning_rate,
        "model": "RFLitModel",
    })

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc', mode='max', save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',   
        mode='max',          
        patience=10,         # number of epochs with no improvement before stopping
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, dm)
    trainer.test(dataloaders=dm.test_dataloader())

    wandb.finish()