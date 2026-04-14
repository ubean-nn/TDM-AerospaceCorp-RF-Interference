import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
TRAIN_TXT = os.path.join(ROOT, "train.txt")
TEST_TXT = os.path.join(ROOT, "test.txt")

NUM_CLASSES = 9
RANDOM_SEED = 42

MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None

DS_FREQ = 4
DS_TIME = 4

BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("TRAIN_TXT:", TRAIN_TXT)
print("TRAIN_TXT exists:", os.path.exists(TRAIN_TXT))
print("TEST_TXT:", TEST_TXT)
print("TEST_TXT exists:", os.path.exists(TEST_TXT))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


# -----------------------------
# Data loading helpers
# -----------------------------
def load_txt_list(txt_path):
    entries = np.loadtxt(txt_path, dtype=str)
    if entries.ndim == 1:
        entries = entries.reshape(1, -1)

    pairs = []
    for rel_path, label_str in entries:
        full_path = os.path.join(ROOT, rel_path)
        pairs.append((full_path, int(label_str)))
    return pairs


def maybe_subsample(pairs, max_samples):
    if max_samples is None or max_samples >= len(pairs):
        return pairs
    idx = np.random.choice(len(pairs), size=max_samples, replace=False)
    return [pairs[i] for i in idx]


# -----------------------------
# Dataset
# -----------------------------
class PSDDataset(Dataset):
    """
    Loads PSD .npy arrays, normalizes, downsamples, resizes, and returns a 3-channel tensor.
    Output:
      x: float32 tensor (3, 224, 224)
      y: int64 tensor ()
      sample_id: string
    """
    def __init__(self, pairs, ds_f=4, ds_t=4, out_size=224, train=False):
        self.pairs = pairs
        self.ds_f = ds_f
        self.ds_t = ds_t
        self.out_size = out_size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        mat = np.load(path).astype(np.float32)

        # Normalize each sample
        mat = (mat - mat.mean()) / (mat.std() + 1e-6)

        # Downsample
        mat = mat[::self.ds_f, ::self.ds_t]

        # Convert to tensor with one channel
        x = torch.from_numpy(mat).unsqueeze(0)

        # Resize to 224x224
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Very light augmentation for training
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = x + 0.01 * torch.randn_like(x)
            x = torch.clamp(x, -5, 5)

        # Repeat to 3 channels for pretrained ViT
        x = x.repeat(3, 1, 1)

        y = torch.tensor(label, dtype=torch.long)
        sample_id = os.path.basename(path).replace(".npy", "")
        return x, y, sample_id


# -----------------------------
# Lightning DataModule
# -----------------------------
class PSDDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        train_pairs = load_txt_list(TRAIN_TXT)
        test_pairs = load_txt_list(TEST_TXT)

        train_pairs = maybe_subsample(train_pairs, MAX_TRAIN_SAMPLES)
        test_pairs = maybe_subsample(test_pairs, MAX_TEST_SAMPLES)

        labels = np.array([y for _, y in train_pairs])

        idx_train, idx_val = train_test_split(
            np.arange(len(train_pairs)),
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=labels
        )

        train_split = [train_pairs[i] for i in idx_train]
        val_split = [train_pairs[i] for i in idx_val]

        self.train_ds = PSDDataset(
            train_split,
            ds_f=DS_FREQ,
            ds_t=DS_TIME,
            out_size=224,
            train=True
        )
        self.val_ds = PSDDataset(
            val_split,
            ds_f=DS_FREQ,
            ds_t=DS_TIME,
            out_size=224,
            train=False
        )
        self.test_ds = PSDDataset(
            test_pairs,
            ds_f=DS_FREQ,
            ds_t=DS_TIME,
            out_size=224,
            train=False
        )

        y_train_split = np.array([y for _, y in train_split])
        class_counts = np.bincount(y_train_split, minlength=NUM_CLASSES)

        print(f"Train samples: {len(self.train_ds)}")
        print(f"Val samples: {len(self.val_ds)}")
        print(f"Test samples: {len(self.test_ds)}")
        print("Class counts:", class_counts.tolist())

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )


# -----------------------------
# LightningModule
# -----------------------------
class LitViT(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float,
        weight_decay: float,
        scheduler: str = "cosine",
        step_size: int = 10,
        gamma: float = 0.5,
        plateau_patience: int = 3,
        plateau_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        # Accuracy-first setup: plain cross entropy, no sampler, no class weights
        self.criterion = nn.CrossEntropyLoss()

        self._val_preds = []
        self._val_true = []

        self._test_preds = []
        self._test_true = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.log("val_loss", loss, prog_bar=True, sync_dist=False)
        self._val_preds.append(preds.detach().cpu().numpy())
        self._val_true.append(y.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        y_pred = np.concatenate(self._val_preds) if len(self._val_preds) else np.array([])
        y_true = np.concatenate(self._val_true) if len(self._val_true) else np.array([])
        self._val_preds.clear()
        self._val_true.clear()

        if y_true.size == 0:
            return

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
        per_class_recall = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)
        acc = (y_pred == y_true).mean()

        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            output_dict=True,
            zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        self.log("val_acc", float(acc), prog_bar=True)
        self.log("val_mean_per_class_recall", float(per_class_recall.mean()), prog_bar=False)
        self.log("val_macro_f1", float(macro_f1), prog_bar=False)
        self.log("val_weighted_f1", float(weighted_f1), prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self._test_preds.append(preds.detach().cpu().numpy())
        self._test_true.append(y.detach().cpu().numpy())

    def on_test_epoch_end(self):
        y_pred = np.concatenate(self._test_preds) if len(self._test_preds) else np.array([])
        y_true = np.concatenate(self._test_true) if len(self._test_true) else np.array([])
        self._test_preds.clear()
        self._test_true.clear()

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
        per_class_recall = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)
        acc = (y_pred == y_true).mean() if y_true.size else 0.0

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            output_dict=True,
            zero_division=0
        )
        macro_f1 = report_dict["macro avg"]["f1-score"]
        weighted_f1 = report_dict["weighted avg"]["f1-score"]

        print("\n=== TEST RESULTS (Lightning) ===")
        print("Test accuracy:", round(float(acc), 4))
        print("Mean per-class recall:", round(float(per_class_recall.mean()), 4))
        print("Macro F1:", round(float(macro_f1), 4))
        print("Weighted F1:", round(float(weighted_f1), 4))
        print("Per-class accuracy (recall) %:", np.round(per_class_recall * 100, 2).tolist())
        print("\nClassification report (TEST):")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print("\nConfusion matrix (TEST):\n", cm)

        np.savetxt("test_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        pd.DataFrame({
            "class": np.arange(NUM_CLASSES),
            "recall": per_class_recall
        }).to_csv("test_per_class_recall.csv", index=False)

        self.log("test_acc", float(acc))
        self.log("test_macro_f1", float(macro_f1))
        self.log("test_weighted_f1", float(weighted_f1))


# -----------------------------
# Trainer builder
# -----------------------------
def build_trainer(run_name: str):
    logger = TensorBoardLogger(save_dir="tb_logs", name="aerospace_rf", version=run_name)

    # Accuracy-first setup: choose best checkpoint by validation accuracy
    ckpt = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="{epoch:02d}-{val_acc:.4f}",
        save_last=True,
    )

    early = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=6,
        verbose=True,
    )

    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=25,
        callbacks=[ckpt, early, lrmon],
        logger=logger,
    )
    return trainer


# -----------------------------
# Export predictions for Excel sheet
# -----------------------------
def export_test_predictions(model, datamodule, out_csv="best_vit_predictions.csv"):
    model.eval()
    model.to(DEVICE)

    loader = datamodule.test_dataloader()
    rows = []

    with torch.no_grad():
        for batch in loader:
            x, y, sample_ids = batch
            x = x.to(DEVICE)

            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for sid, pred in zip(sample_ids, preds):
                rows.append({
                    "sample_id": sid,
                    "prediction": int(pred)
                })

    df = pd.DataFrame(rows)
    df["sample_id_num"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df = (
        df.sort_values(["sample_id_num", "sample_id"])
          .drop(columns=["sample_id_num"])
          .reset_index(drop=True)
    )

    df.to_csv(out_csv, index=False)

    print(f"Saved predictions to {out_csv}")
    print(df.head())
    return df


# -----------------------------
# Train / select / export
# -----------------------------
def run_patch_sweep(scheduler: str = "cosine"):
    L.seed_everything(RANDOM_SEED, workers=True)

    dm = PSDDataModule()
    dm.setup("fit")

    # Try both patch sizes
    patch_models = [
        "vit_base_patch16_224",
        "vit_base_patch8_224",
    ]

    best_overall_score = -1.0
    best_model_name = None
    best_ckpt_path = None

    for model_name in patch_models:
        run_name = f"{model_name}_sched-{scheduler}_lr-{LR}"
        print(f"\n=== RUN: {run_name} ===")

        model = LitViT(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            scheduler=scheduler,
        )

        trainer = build_trainer(run_name)
        trainer.fit(model, datamodule=dm)

        best_path = trainer.checkpoint_callback.best_model_path
        score = trainer.checkpoint_callback.best_model_score

        if best_path:
            model = LitViT.load_from_checkpoint(
                best_path,
                model_name=model_name,
                num_classes=NUM_CLASSES,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                scheduler=scheduler,
            )

        trainer.test(model, datamodule=dm)

        if score is not None and float(score) > best_overall_score:
            best_overall_score = float(score)
            best_model_name = model_name
            best_ckpt_path = best_path

    print("\n=== BEST RUN (by val_acc) ===")
    print("Best model:", best_model_name)
    print("Best score:", best_overall_score)
    print("Best checkpoint:", best_ckpt_path)

    if best_ckpt_path is None:
        raise RuntimeError("No best checkpoint was found. Training may have failed.")

    best_model = LitViT.load_from_checkpoint(
        best_ckpt_path,
        model_name=best_model_name,
        num_classes=NUM_CLASSES,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        scheduler=scheduler,
    )

    export_test_predictions(best_model, dm, out_csv="best_vit_predictions.csv")


if __name__ == "__main__":
    run_patch_sweep(scheduler="cosine")