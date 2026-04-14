import argparse
import os
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
except ImportError:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.loggers import CSVLogger
    except ImportError as exc:
        raise ImportError(
            "Install either 'pytorch-lightning' or 'lightning' to run this script."
        ) from exc

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


INPUT_SHAPE = (1, 512, 243)
DEFAULT_CLUSTER_DATASET_ROOT = Path("/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main")
DEFAULT_CLUSTER_OUTPUT_DIR = Path("/anvil/projects/x-cis220051/corporate/aerospace-rf/ine/sbatch")


def default_num_workers() -> int:
    for env_name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        raw_value = os.getenv(env_name)
        if raw_value and raw_value.isdigit():
            return max(0, int(raw_value))
    return min(4, os.cpu_count() or 1)


def default_output_dir(script_name: str) -> Path:
    return DEFAULT_CLUSTER_OUTPUT_DIR


def resolve_split_path(explicit_path: Path | None, dataset_root: Path, filename: str) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    return (dataset_root / filename).resolve()


def validate_paths(dataset_root: Path, required_paths: Sequence[Path]) -> None:
    missing_paths = [path for path in [dataset_root, *required_paths] if not path.exists()]
    if missing_paths:
        missing_str = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Required dataset paths were not found. Pass the correct paths with "
            "--dataset-root/--train-txt/--val-txt/--test-txt.\n"
            f"{missing_str}"
        )


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x / keep_prob * random_tensor


class HighwayDataset(Dataset):
    def __init__(self, txt_file: Path, data_root: Path, noise_std: float = 0.0) -> None:
        self.samples: list[tuple[str, int]] = []
        self.data_root = data_root
        self.noise_std = noise_std

        with txt_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                self.samples.append((parts[0], int(parts[1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        relative_path, label = self.samples[index]
        sample_path = (self.data_root / relative_path).resolve()
        array = np.load(sample_path)

        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        elif array.shape[-1] <= 4:
            array = np.transpose(array, (2, 0, 1))

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, array.shape)
            array = array + noise

        x = torch.tensor(array, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, str(sample_path)


class CNNV6(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.55)
        self.droppath = DropPath(0.03)
        self.fc1 = nn.Linear(self._infer_feature_dim(INPUT_SHAPE), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(0.25, 0.25), mode="bilinear", align_corners=False)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.droppath(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.droppath(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.droppath(x)
        return x

    def _infer_feature_dim(self, input_shape: tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            features = self._extract_features(dummy)
        return int(features.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._extract_features(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class LightningCNN(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float,
        output_csv_path: str,
        label_offset: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = CNNV6(num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.test_paths: list[str] = []

    def _unpack_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
        if len(batch) == 3:
            x, y, paths = batch
            return x, y, paths
        x, y = batch
        return x, y, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = self._unpack_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int
    ) -> None:
        x, y, _ = self._unpack_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int
    ) -> None:
        x, y, paths = self._unpack_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        if paths is not None:
            self.test_paths.extend(paths)

    def on_test_epoch_end(self) -> None:
        if not self.test_preds:
            return

        label_values = list(
            range(
                self.hparams.label_offset,
                self.hparams.label_offset + self.hparams.num_classes,
            )
        )
        all_preds = torch.cat(self.test_preds).numpy() + self.hparams.label_offset
        all_targets = torch.cat(self.test_targets).numpy() + self.hparams.label_offset

        results_df = pd.DataFrame(
            {
                "Sample_Path": self.test_paths,
                "True_Label": all_targets,
                "Predicted_Label": all_preds,
                "True_Label_Name": [str(value) for value in all_targets],
                "Predicted_Label_Name": [str(value) for value in all_preds],
            }
        )

        output_csv_path = Path(self.hparams.output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nSaved test inference CSV to: {output_csv_path}")

        class_names = [str(label) for label in label_values]
        confusion = confusion_matrix(all_targets, all_preds, labels=label_values)
        confusion_df = pd.DataFrame(
            confusion,
            index=[f"True_{label}" for label in class_names],
            columns=[f"Pred_{label}" for label in class_names],
        )

        print("\n[CONFUSION MATRIX]")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(confusion_df)

        print("\n[CLASSIFICATION REPORT]")
        print(
            classification_report(
                all_targets,
                all_preds,
                labels=label_values,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )

        self.test_preds.clear()
        self.test_targets.clear()
        self.test_paths.clear()

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def random_oversample(
    features: np.ndarray,
    labels: np.ndarray,
    sampling_strategy: dict[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_blocks = [features]
    label_blocks = [labels]

    for label, target_size in sampling_strategy.items():
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) == 0:
            continue
        num_to_add = max(0, target_size - len(label_indices))
        if num_to_add == 0:
            continue
        sampled_indices = rng.choice(label_indices, size=num_to_add, replace=True)
        feature_blocks.append(features[sampled_indices])
        label_blocks.append(labels[sampled_indices])

    resampled_features = np.concatenate(feature_blocks, axis=0)
    resampled_labels = np.concatenate(label_blocks, axis=0)
    permutation = rng.permutation(len(resampled_labels))
    return resampled_features[permutation], resampled_labels[permutation]


def apply_smote(dataset: HighwayDataset, target_size: int, seed: int) -> TensorDataset:
    features = []
    labels = []
    for index in range(len(dataset)):
        x, y, _ = dataset[index]
        features.append(x.flatten().numpy())
        labels.append(int(y.item()))

    features_np = np.asarray(features)
    labels_np = np.asarray(labels)

    class_counts = Counter(labels_np.tolist())
    sampling_strategy = {
        label: target_size for label, count in class_counts.items() if count < target_size
    }

    print(f"Before SMOTE: {class_counts}")
    if not sampling_strategy:
        print("All classes already meet the target size. Using the original training set.")
        x_tensor = torch.tensor(features_np.reshape((-1, *INPUT_SHAPE)), dtype=torch.float32)
        y_tensor = torch.tensor(labels_np, dtype=torch.long)
        return TensorDataset(x_tensor, y_tensor)

    if SMOTE is not None:
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
        resampled_features, resampled_labels = sampler.fit_resample(features_np, labels_np)
    else:
        print("imblearn is not installed. Falling back to random oversampling instead of SMOTE.")
        resampled_features, resampled_labels = random_oversample(
            features_np,
            labels_np,
            sampling_strategy=sampling_strategy,
            seed=seed,
        )

    print(f"After SMOTE: {Counter(resampled_labels.tolist())}")
    x_tensor = torch.tensor(resampled_features.reshape((-1, *INPUT_SHAPE)), dtype=torch.float32)
    y_tensor = torch.tensor(resampled_labels, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-safe training script for the final CNN model."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_CLUSTER_DATASET_ROOT)
    parser.add_argument("--train-txt", type=Path, default=None)
    parser.add_argument("--val-txt", type=Path, default=None)
    parser.add_argument("--test-txt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir("final_model"))
    parser.add_argument("--run-name", type=str, default="final_model")
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--target-smote-size", type=int, default=1000)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--prediction-csv-name", type=str, default="run-2.csv")
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    dataset_root = args.dataset_root.expanduser().resolve()
    train_txt = resolve_split_path(args.train_txt, dataset_root, "train.txt")
    val_txt = resolve_split_path(args.val_txt, dataset_root, "test.txt")
    test_txt = resolve_split_path(args.test_txt, dataset_root, "test.txt")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_paths(dataset_root, [train_txt, val_txt, test_txt])

    train_data = HighwayDataset(train_txt, dataset_root, noise_std=args.noise_std)
    val_data = HighwayDataset(val_txt, dataset_root, noise_std=0.0)
    test_data = HighwayDataset(test_txt, dataset_root, noise_std=0.0)

    if args.no_smote:
        train_dataset: Dataset = train_data
        print("Using the original training set without SMOTE.")
    else:
        train_dataset = apply_smote(train_data, target_size=args.target_smote_size, seed=args.seed)

    train_loader = make_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_dataloader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger = CSVLogger(save_dir=str(output_dir / "logs"), name=args.run_name)
    inference_csv_path = output_dir / args.prediction_csv_name

    model = LightningCNN(
        num_classes=args.num_classes,
        lr=args.learning_rate,
        output_csv_path=str(inference_csv_path),
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=args.log_every_n_steps,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    checkpoint_path = output_dir / f"{args.run_name}.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    end_time = time.time()
    elapsed_seconds = end_time - start_time

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved Lightning metrics CSV to: {Path(logger.log_dir) / 'metrics.csv'}")
    print(f"Total training + testing time: {elapsed_seconds / 60:.2f} minutes ({elapsed_seconds:.1f} seconds)")


if __name__ == "__main__":
    main()
