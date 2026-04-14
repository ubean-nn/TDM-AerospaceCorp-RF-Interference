import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import Accuracy, ConfusionMatrix

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb


# =========================
# Config
# =========================
ROOT = "../fiot_highway2-main"

NUM_CLASSES = 9
BATCH_SIZE = 32
MAX_EPOCHS = 15

LR = 1e-3
WD = 1e-4

ENSEMBLE_SEEDS = [42, 123, 999, 2026, 7]  # K models
VAL_SPLIT = 0.2

USE_WANDB = True
WANDB_PROJECT = "fiot-highway2-rfi"

ACCELERATOR = "gpu" 
DEVICES = 1


# =========================
# Reproducibility
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


# =========================
# Data loading
# =========================
def read_split(fname: str):
    paths = []
    labels = []
    with open(os.path.join(ROOT, fname), "r") as f:
        for line in f:
            rel, label = line.strip().split()
            full_path = os.path.join(ROOT, rel)
            paths.append(full_path)
            labels.append(int(label))
    return paths, labels


def load_arrays(paths, labels):
    arrays = []
    lbls = []

    for p, y in zip(paths, labels):
        a = np.load(p)

        # per-sample normalization to [0, 1]
        a_min = a.min()
        a_max = a.max()
        a = (a - a_min) / (a_max - a_min + 1e-8)

        arrays.append(a)
        lbls.append(y)

    X = torch.tensor(np.stack(arrays), dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)
    y = torch.tensor(lbls, dtype=torch.long)
    return TensorDataset(X, y)


def build_dataloaders(batch_size=BATCH_SIZE, val_split=VAL_SPLIT, split_seed=2026):
    train_paths, train_labels = read_split("train.txt")
    test_paths, test_labels = read_split("test.txt")

    full_train_ds = load_arrays(train_paths, train_labels)
    test_ds = load_arrays(test_paths, test_labels)

    n_total = len(full_train_ds)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    gen = torch.Generator()
    gen.manual_seed(split_seed)

    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# =========================
# Model
# =========================
class RFICNN(L.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=LR, wd=WD):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # Separate metrics per stage (cleaner than sharing one metric)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # For single-model test confusion matrix (optional)
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_acc.update(preds, y)
        self.test_confmat.update(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        cm = self.test_confmat.compute().cpu().numpy()
        self.test_confmat.reset()

        class_names = []
        for i in range(self.hparams.num_classes):
            class_names.append(f"class_{i}")

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    "single_model_confusion_matrix": wandb.plot.confusion_matrix(
                        conf_mat=cm,
                        class_names=class_names,
                    )
                }
            )

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
            },
        }


# =========================
# Training one model
# =========================
def train_one_model(seed, train_loader, val_loader):
    set_seed(seed)

    run_name = f"cnn_seed_{seed}"

    wandb_logger = None
    if USE_WANDB:
        wandb_logger = WandbLogger(
            project=WANDB_PROJECT,
            name=run_name,
            log_model="all",
        )

    model = RFICNN()

    if wandb_logger is not None:
        wandb_logger.watch(model, log="all", log_freq=100)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_cb],
    )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    print(f"[seed={seed}] best checkpoint: {best_ckpt}")

    return best_ckpt


# =========================
# Ensemble inference
# =========================
@torch.no_grad()
def ensemble_evaluate(checkpoint_paths, test_loader, num_classes=NUM_CLASSES):
    device = "cuda"
    if not torch.cuda.is_available():
        device = "cpu"

    models = []
    for ckpt in checkpoint_paths:
        m = RFICNN.load_from_checkpoint(ckpt)
        m.eval()
        m.to(device)
        models.append(m)

    all_probs = []
    all_labels = []

    for x, y in test_loader:
        x = x.to(device)

        probs_sum = None

        for model in models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum = probs_sum + probs

        avg_probs = probs_sum / len(models)

        all_probs.append(avg_probs.cpu())
        all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = torch.argmax(all_probs, dim=1)
    acc = (preds == all_labels).float().mean().item()

    cm_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    cm = cm_metric(preds, all_labels).cpu().numpy()

    return acc, cm, preds, all_labels, all_probs


def log_ensemble_results_to_wandb(acc, cm, num_classes=NUM_CLASSES):
    if not USE_WANDB:
        return

    class_names = []
    for i in range(num_classes):
        class_names.append(f"class_{i}")

    run = wandb.init(project=WANDB_PROJECT, name="ensemble_eval", reinit=True)
    run.log(
        {
            "ensemble_test_acc": acc,
            "ensemble_confusion_matrix": wandb.plot.confusion_matrix(
                conf_mat=cm,
                class_names=class_names,
            ),
        }
    )
    run.finish()


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Build loaders once so every seed uses the same train/val split
    train_loader, val_loader, test_loader = build_dataloaders()

    # Train K copies of the same CNN
    checkpoint_paths = []
    for seed in ENSEMBLE_SEEDS:
        ckpt = train_one_model(seed, train_loader, val_loader)
        checkpoint_paths.append(ckpt)

    # Ensemble test
    ensemble_acc, ensemble_cm, preds, labels, probs = ensemble_evaluate(
        checkpoint_paths,
        test_loader,
        num_classes=NUM_CLASSES,
    )

    print(f"\nEnsemble Test Accuracy: {ensemble_acc:.4f}")

    # Log ensemble metrics/CM to W&B
    log_ensemble_results_to_wandb(ensemble_acc, ensemble_cm, num_classes=NUM_CLASSES)