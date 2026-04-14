# imports
import torch
import numpy as np
import lightning as L
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from collections import Counter
import time
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning.loggers import WandbLogger

# start timer
start_time = time.time()


# custom droppath
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x / keep_prob * random_tensor
        

# dataset creation
class HighwayDataset(Dataset):
    def __init__(self, txt_file, data_root, noise_std=0.01):
        self.samples = []
        self.data_root = data_root
        self.noise_std = noise_std

        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        npy_path = os.path.join(self.data_root, rel_path)
        arr = np.load(npy_path)

        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        elif arr.shape[-1] <= 4:
            arr = np.transpose(arr, (2, 0, 1))

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, arr.shape)
            arr = arr + noise

        x = torch.tensor(arr, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# CNN with bilinear interpolation and batch norm
class CNN_V6(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.pool = nn.MaxPool2d(2,2)

    self.dropout = nn.Dropout(0.55) #was 0.25
    self.droppath = DropPath(0.03)

    self.fc1 = None
    self.fc2 = nn.Linear(128, 9)

  def forward(self, x):
    x = F.interpolate(x, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=False)

    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.droppath(x)
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.droppath(x)
    x = self.pool(F.relu(self.bn3(self.conv3(x))))
    x = self.droppath(x)

    x = x.view(x.size(0), -1)

    if self.fc1 is None:
      self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

    x = self.dropout(F.relu(self.fc1(x)))

    x = self.fc2(x)

    return x


# lightning wrapper
class LightningCNN(L.LightningModule):
    def __init__(self, num_classes=9, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_V6()  # CHANGE THIS TO CHANGE MODEL
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}


# confusion matrix
def evaluate_confusion_matrix(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x.to(model.device))
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.argmax(dim=1).cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", xticks_rotation=45)

# SMOTE
def apply_smote(dataset, target_size=1000):
    X = []
    y = []
    for i in range(len(dataset)):
        arr, label = dataset[i]
        X.append(arr.flatten().numpy())
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    class_counts = Counter(y)
    print("Before SMOTE:", Counter(y))
    sampling_strategy = {cls: target_size for cls, count in class_counts.items() if count < target_size}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("After SMOTE:", Counter(y_res))

    # Convert back to tensors
    X_res = X_res.reshape((-1, 1, 512, 243))
    tensors = [(torch.tensor(x, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))
               for x, label in zip(X_res, y_res)]
    return tensors


# data setup
base = "../../../fiot_highway2-main"
train_txt = os.path.join(base, "train.txt")
test_txt = os.path.join(base, "test.txt")
data_dir = os.path.join(base, "data")

train_data = HighwayDataset(train_txt, base, noise_std=0.05)
test_data = HighwayDataset(test_txt, base, noise_std=0.0)
smote_data = apply_smote(train_data)

train_loader = DataLoader(train_data, batch_size = 32, shuffle=True, num_workers=4)
smote_loader = DataLoader(smote_data, batch_size = 32, shuffle=True, num_workers=4)
val_loader = DataLoader(test_data, batch_size = 32, shuffle=True, num_workers=4)

# wandb confusion matrix
def log_confusion_matrix_wandb(model, dataloader, logger):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    logger.experiment.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=[str(i) for i in range(9)]
        )
    })


# training
wandb_logger = WandbLogger(project="whitney_test_architecture", name="drop_path2/increase_dropout2", log_model=True)
trainer = L.Trainer(max_epochs=50, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=wandb_logger, log_every_n_steps=10)
model = LightningCNN(num_classes=9)
trainer.fit(model, smote_loader, val_loader)
trainer.test(model, dataloaders=val_loader)
log_confusion_matrix_wandb(model, val_loader, wandb_logger)
evaluate_confusion_matrix(model, val_loader)
plt.show()


# end time
end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal training + testing time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")