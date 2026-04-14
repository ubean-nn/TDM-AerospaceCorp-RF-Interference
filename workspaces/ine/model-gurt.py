import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import Accuracy
import numpy as np
import glob
import os

import wandb
from pytorch_lightning.loggers import WandbLogger

class RFDataset(Dataset):
    '''Custom dataset for RF signal numpy files'''
    
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load npy file
        data = np.load(self.file_paths[idx]).astype(np.float32)
        
        # Normalize
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        # Add channel dimension (1, 512, 243)
        data = np.expand_dims(data, axis=0)
        
        # Convert to tensor
        data = torch.from_numpy(data)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

class RFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_labels_file, test_labels_file, batch_size=32, val_split=0.15):
        super().__init__()
        self.data_dir = data_dir
        self.train_labels_file = train_labels_file
        self.test_labels_file = test_labels_file
        self.batch_size = batch_size
        self.val_split = val_split  # 15% of train data for validation
        self.dims = (1, 512, 243)
        self.num_classes = 9
    
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
        return DataLoader(self.rf_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.rf_val, batch_size=self.batch_size, num_workers=8) # num workes for larger cpu notebook
    
    def test_dataloader(self):
        return DataLoader(self.rf_test, batch_size=self.batch_size, num_workers=8)

class RFLitModel(pl.LightningModule):
    '''model architecture, training, testing and validation loops'''
    
    def __init__(self, input_shape, num_classes, learning_rate=1e-3):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
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

        # --- for confusion analysis ---
        self._test_targets = []
        self._test_preds = []
    
    def _get_output_shape(self, shape):
        '''returns the size of the output tensor from the conv layers'''
        batch_size = 1
        x = torch.rand(batch_size, *shape)
        with torch.no_grad():
            output_feat = self._feature_extractor(x)
        n_size = output_feat.view(batch_size, -1).size(1)
        return n_size
    
    def _feature_extractor(self, x):
        '''extract features from the conv blocks'''
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.bn1(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.bn2(x)
        return x
    
    def forward(self, x):
        '''produce final model output'''
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    # train loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss
    
    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    # test loop
    def on_test_epoch_start(self):
        # reset collectors each test run
        self._test_targets = []
        self._test_preds = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        # --- collect for confusion analysis ---
        self._test_targets.append(y.detach().cpu())
        self._test_preds.append(preds.detach().cpu())

        return loss

    def on_test_epoch_end(self):
        # --- build and print written confusion analysis ---
        if not self._test_targets:
            print("\n[Confusion Analysis] No test samples collected.")
            return

        targets = torch.cat(self._test_targets).numpy()
        preds = torch.cat(self._test_preds).numpy()
        num_classes = getattr(self.hparams, "num_classes", self.fc3.out_features)

        # counts[true, pred]
        counts = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(targets, preds):
            counts[int(t), int(p)] += 1

        total = counts.sum()
        correct = counts.trace()
        overall_acc = (correct / total) if total > 0 else 0.0

        print("\n==================== Confusion Analysis (Written) ====================")
        print(f"Overall: {correct}/{total} correct  |  Accuracy: {overall_acc:.2%}")
        print("\nPer-TRUE-label breakdown (sorted by frequency of guesses):")
        for t in range(num_classes):
            row = counts[t]
            total_t = row.sum()
            if total_t == 0:
                print(f"  True {t:>2}: (no samples)")
                continue
            pairs = [(p, row[p]) for p in range(num_classes) if row[p] > 0]
            pairs.sort(key=lambda x: x[1], reverse=True)
            summary = "  ".join([f"pred {p}: {c}" for p, c in pairs])
            acc_t = (row[t] / total_t) if total_t > 0 else 0.0
            print(f"  True {t:>2}: {summary}   | correct {row[t]}/{total_t} ({acc_t:.1%})")

        # Flat two-column form: "should be" vs "guessed"
        print("\nFlat counts table (columns = what it should be, what was guessed, count):")
        print(f"{'TRUE':<8}{'PRED':<8}{'COUNT':<8}")
        for t in range(num_classes):
            for p in range(num_classes):
                c = counts[t, p]
                if c > 0:
                    print(f"{t:<8}{p:<8}{c:<8}")
        print("=====================================================================\n")
    
    # optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == "__main__":
    # instantiate classes with the labels file
    dm = RFDataModule(
    data_dir='/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/data/',
    train_labels_file='/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/train.txt',
    test_labels_file='/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main/test.txt',
    batch_size=32,
    val_split=0.15
    )    
    dm.prepare_data()
    dm.setup()
    
    model = RFLitModel((1, 512, 243), dm.num_classes)


    wandb_logger = WandbLogger(
        project="rf-signal-classification",     
        name="bean_v1",           
        log_model=True                         
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)
    
    trainer = pl.Trainer(max_epochs=20, accelerator='auto', devices=1, callbacks=[checkpoint_callback])
    
    trainer.fit(model, dm)
    trainer.test(dataloaders=dm.test_dataloader())