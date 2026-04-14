import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmetrics.classification import Accuracy, ConfusionMatrix

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb


# --- Load paths and labels from text files ---
root = "../fiot_highway2-main"


def read_split(fname):
    paths, labels = [], []
    with open(os.path.join(root, fname)) as f:
        for line in f:
            rel, label = line.strip().split()
            paths.append(os.path.join(root, rel))
            labels.append(int(label))
    return paths, labels


train_paths, train_labels = read_split("train.txt")
test_paths, test_labels = read_split("test.txt")


# --- Load data into memory (normalizing each sample) ---
def load_arrays(paths, labels):
    arrays, lbls = [], []
    for p, y in zip(paths, labels):
        a = np.load(p)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        arrays.append(a)
        lbls.append(y)
    X = torch.tensor(np.stack(arrays), dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)
    y = torch.tensor(lbls, dtype=torch.long)
    return TensorDataset(X, y)


train_ds = load_arrays(train_paths, train_labels)
test_ds = load_arrays(test_paths, test_labels)
val_ds = test_ds  # reuse test set for validation

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)


# --- Simple CNN Model with W&B-friendly logging ---
class RFICNN(L.LightningModule):
    def __init__(self, num_classes=9, lr=1e-3, wd=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

        # For confusion matrix on test set
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = torch.flatten(self.net(x), 1)
        return self.fc(x)

    def _shared(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(1)
        acc = self.acc(preds, y)

        # log to Lightning (and thus to W&B via WandbLogger)
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        if stage == "test":
            # accumulate confusion matrix data
            self.test_confmat.update(preds, y)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared(batch, "test")

    def on_test_epoch_end(self):
        # compute confusion matrix and log to W&B
        cm = self.test_confmat.compute().cpu().numpy()
        self.test_confmat.reset()

        class_names = [f"class_{i}" for i in range(self.hparams.num_classes)]

        # self.logger is a WandbLogger; experiment is the wandb.Run
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    conf_mat=cm,
                    class_names=class_names,
                )
            })

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    # --- W&B logger setup ---
    wandb_logger = WandbLogger(
        project="fiot-highway2-rfi",
        name="cnn-baseline",
        log_model="all",  # upload checkpoints
    )

    model = RFICNN()

    # Log gradients & parameters to W&B
    wandb_logger.watch(model, log="all", log_freq=100)

    # Callbacks: log LR, save best model
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=15,
        accelerator="gpu",
        devices=1,  # one GPU
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_cb],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)