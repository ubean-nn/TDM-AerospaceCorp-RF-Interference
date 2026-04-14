import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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


MIN_DB = -140.0
MAX_DB = 0.0
DEFAULT_CLUSTER_DATASET_ROOT = Path("/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main")
DEFAULT_CLUSTER_OUTPUT_DIR = Path("/anvil/projects/x-cis220051/corporate/aerospace-rf/ine/sbatch")
DEFAULT_BOOSTED_ROOT = Path("/anvil/projects/x-cis220051/corporate/aerospace-rf/ebroyles/BoostedHierarchialModel")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    depth: str
    augment: bool
    max_epochs: int


DEFAULT_EXPERIMENTS = {
    "shallow_no_aug": ExperimentConfig("shallow_no_aug", "shallow", False, 10),
    "shallow_aug": ExperimentConfig("shallow_aug", "shallow", True, 10),
    "deep_no_aug": ExperimentConfig("deep_no_aug", "deep", False, 10),
    "deep_aug": ExperimentConfig("deep_aug", "deep", True, 10),
    # The notebook used augment=True for this run even though the name says no_aug.
    "deeper_no_aug": ExperimentConfig("deeper_no_aug", "deeper", True, 8),
}


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


def read_split_samples(split_path: Path, min_label: int, max_label: int) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            relative_path = parts[0]
            label = int(parts[1])
            if min_label <= label <= max_label:
                samples.append((relative_path, label))
    return samples


def read_boosted_samples(boosted_root: Path, target_label: int) -> list[tuple[str, int]]:
    if not boosted_root.exists():
        print(f"Skipping boosted samples because the folder does not exist: {boosted_root}")
        return []

    return [(path.name, target_label) for path in sorted(boosted_root.glob("*.npy"))]


class HighwayDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        root_dir: Path,
        resize: tuple[int, int],
        min_label: int,
        min_db: float,
        max_db: float,
    ) -> None:
        self.samples = list(samples)
        self.root_dir = root_dir
        self.resize = resize
        self.min_label = min_label
        self.min_db = min_db
        self.max_db = max_db

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        relative_path, label = self.samples[index]
        sample_path = (self.root_dir / relative_path).resolve()

        array = np.load(sample_path)
        array = np.clip(array, self.min_db, self.max_db)
        array = (array - self.min_db) / (self.max_db - self.min_db)

        image = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
        image = TF.resize(image, self.resize)
        shifted_label = torch.tensor(label - self.min_label, dtype=torch.long)

        return image, shifted_label, str(sample_path)


class CNNClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        depth: str,
        learning_rate: float,
        output_csv_path: str,
        label_offset: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if depth == "shallow":
            self.model = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 16 * 16, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        elif depth == "deep":
            self.model = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        elif depth == "deeper":
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.4),
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
        else:
            raise ValueError("depth must be 'shallow', 'deep', or 'deeper'")

        self.loss_fn = nn.CrossEntropyLoss()
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.test_paths: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int) -> None:
        x, y, paths = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


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
        description="Standalone version of CNN0to8.ipynb for local or Slurm execution."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_CLUSTER_DATASET_ROOT)
    parser.add_argument("--train-txt", type=Path, default=None)
    parser.add_argument("--test-txt", type=Path, default=None)
    parser.add_argument(
        "--boosted-class1-root",
        type=Path,
        default=DEFAULT_BOOSTED_ROOT / "BoostedClass1",
    )
    parser.add_argument(
        "--boosted-class3-root",
        type=Path,
        default=DEFAULT_BOOSTED_ROOT / "BoostedClass3",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir("cnn0to8"))
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(DEFAULT_EXPERIMENTS.keys()),
        default=list(DEFAULT_EXPERIMENTS.keys()),
    )
    parser.add_argument("--min-label", type=int, default=0)
    parser.add_argument("--max-label", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resize-height", type=int, default=64)
    parser.add_argument("--resize-width", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--prediction-csv-name", type=str, default="run-1.csv")
    return parser.parse_args()


def validate_paths(dataset_root: Path, train_txt: Path, test_txt: Path) -> None:
    missing_paths = [path for path in (dataset_root, train_txt, test_txt) if not path.exists()]
    if missing_paths:
        missing_str = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Required dataset paths were not found. Pass the correct paths with "
            "--dataset-root/--train-txt/--test-txt.\n"
            f"{missing_str}"
        )


def metric_to_float(metric: object) -> float | None:
    if metric is None:
        return None
    if isinstance(metric, torch.Tensor):
        return float(metric.detach().cpu().item())
    if isinstance(metric, (float, int)):
        return float(metric)
    return None


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    dataset_root = args.dataset_root.expanduser().resolve()
    train_txt = resolve_split_path(args.train_txt, dataset_root, "train.txt")
    test_txt = resolve_split_path(args.test_txt, dataset_root, "test.txt")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_paths(dataset_root, train_txt, test_txt)

    train_samples = read_split_samples(train_txt, args.min_label, args.max_label)
    test_samples = read_split_samples(test_txt, args.min_label, args.max_label)
    boosted_class1_samples = read_boosted_samples(args.boosted_class1_root.expanduser().resolve(), 1)
    boosted_class3_samples = read_boosted_samples(args.boosted_class3_root.expanduser().resolve(), 3)

    resize = (args.resize_height, args.resize_width)
    train_ds = HighwayDataset(
        train_samples,
        dataset_root,
        resize=resize,
        min_label=args.min_label,
        min_db=MIN_DB,
        max_db=MAX_DB,
    )
    test_ds = HighwayDataset(
        test_samples,
        dataset_root,
        resize=resize,
        min_label=args.min_label,
        min_db=MIN_DB,
        max_db=MAX_DB,
    )

    boosted_datasets = []
    if boosted_class3_samples:
        boosted_datasets.append(
            HighwayDataset(
                boosted_class3_samples,
                args.boosted_class3_root.expanduser().resolve(),
                resize=resize,
                min_label=args.min_label,
                min_db=MIN_DB,
                max_db=MAX_DB,
            )
        )
    if boosted_class1_samples:
        boosted_datasets.append(
            HighwayDataset(
                boosted_class1_samples,
                args.boosted_class1_root.expanduser().resolve(),
                resize=resize,
                min_label=args.min_label,
                min_db=MIN_DB,
                max_db=MAX_DB,
            )
        )

    test_loader = make_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    num_classes = args.max_label - args.min_label + 1

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Boosted class 1 samples: {len(boosted_class1_samples)}")
    print(f"Boosted class 3 samples: {len(boosted_class3_samples)}")

    best_experiment_name: str | None = None
    best_accuracy: float | None = None
    best_csv_path: Path | None = None

    for experiment_name in args.experiments:
        experiment = DEFAULT_EXPERIMENTS[experiment_name]
        datasets = [train_ds]
        if experiment.augment:
            datasets.extend(boosted_datasets)

        merged_train_ds = ConcatDataset(datasets)
        train_loader = make_dataloader(
            merged_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        run_name = f"cnn_highway_{experiment.name}"
        if len(args.experiments) == 1:
            inference_csv_path = output_dir / args.prediction_csv_name
        else:
            csv_stem = Path(args.prediction_csv_name).stem
            csv_suffix = Path(args.prediction_csv_name).suffix or ".csv"
            inference_csv_path = output_dir / f"{csv_stem}_{experiment.name}{csv_suffix}"
        logger = CSVLogger(save_dir=str(output_dir / "logs"), name=run_name)

        print(f"\n=== Running {run_name} ===")

        model = CNNClassifier(
            num_classes=num_classes,
            depth=experiment.depth,
            learning_rate=args.learning_rate,
            output_csv_path=str(inference_csv_path),
            label_offset=args.min_label,
        )

        trainer = pl.Trainer(
            max_epochs=experiment.max_epochs,
            logger=logger,
            accelerator=args.accelerator,
            devices=args.devices,
            default_root_dir=str(output_dir),
            log_every_n_steps=args.log_every_n_steps,
        )

        trainer.fit(model, train_dataloaders=train_loader)
        test_results = trainer.test(model, dataloaders=test_loader)

        checkpoint_path = output_dir / f"{run_name}.ckpt"
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"Saved checkpoint to: {checkpoint_path}")
        print(f"Lightning metrics CSV: {Path(logger.log_dir) / 'metrics.csv'}")

        test_metrics = test_results[0] if test_results else {}
        test_accuracy = metric_to_float(test_metrics.get("test_acc"))
        if test_accuracy is None:
            for key, value in test_metrics.items():
                if "test_acc" in key:
                    test_accuracy = metric_to_float(value)
                    break

        if test_accuracy is not None:
            print(f"Test accuracy for {run_name}: {test_accuracy:.6f}")
            if best_accuracy is None or test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_experiment_name = experiment.name
                best_csv_path = inference_csv_path

    if len(args.experiments) > 1 and best_csv_path is not None:
        final_best_csv_path = output_dir / args.prediction_csv_name
        shutil.copy2(best_csv_path, final_best_csv_path)
        print(
            f"\nBest experiment: {best_experiment_name} "
            f"(test_acc={best_accuracy:.6f})"
        )
        print(f"Copied best inference CSV to: {final_best_csv_path}")


if __name__ == "__main__":
    main()
