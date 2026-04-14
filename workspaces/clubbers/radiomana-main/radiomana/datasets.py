"""loaders for public datasets"""
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .transforms import LogNoise

DSET_ENV_LUT = {
    # lookup table for environment variable to dataset source URL
    "DSET_FIOT_HIGHWAY2": "https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2"
}


def get_dataset_path(some_env_var: str) -> Path:
    """
    Given an environment variable name, return the Path object for the dataset directory.
    If the environment variable is not set, raise an error.
    """
    dataset_path = os.getenv(some_env_var)
    if not dataset_path:
        raise ValueError(
            f"Environment variable {some_env_var} is not set. Please clone the dataset from {DSET_ENV_LUT.get(some_env_var, 'the appropriate source')} and set the environment variable accordingly."
        )
    return Path(dataset_path)


class Highway2Dataset(Dataset):
    """
    Dataset class for the FIOT Highway2 dataset

    This 30GB dataset contains PSDs of shape (freq, time) = (512, 243).

    working @ dataset git rev 6246f6

    items are indexed by text files in the root directory and contain rows with "folder/file class_label"
    """

    def __init__(self, root_dir: str = "DSET_FIOT_HIGHWAY2", subset: str = "train", transform=None):
        if subset not in ["train", "test"]:
            raise ValueError("subset must be 'train' or 'test'")
        self.root_dir = get_dataset_path(root_dir)
        self.subset = subset
        self.items = self.load_items()
        self.transform = transform

        # metadata from dataset readme
        self.sample_rate_hz = 62.5e6
        self.sample_duration_s = 0.02
        self.class_labels = [
            "None",
            "None",
            "None",
            "None",
            "Chirp, high distance",
            "Chirp, medium distance",
            "Chirp, small distance",
            "Cigarette lighter 1",
            "Cigarette lighter 2",
        ]

    def load_items(self) -> list:
        labels_path = self.root_dir / f"{self.subset}.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"Items file not found at {labels_path}")
        items = []
        with open(labels_path, "r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) == 2:
                    items.append((parts[0], int(parts[1])))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        Get a single PSD and its label.

        PSDs seem to be in dB units, with noise floor around -90 dB.

        Note this function could be accelerated significantly with stochastic caching and shared memory between workers.
        https://charl-ai.github.io/blog/dataloaders/
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
        file_path, label = self.items[idx]
        # load the data from the file (implementation depends on file format)
        data = torch.from_numpy(np.load(self.root_dir / file_path))
        # apply any transforms
        if self.transform:
            data = self.transform(data)
        # stored as float64, but precision is overkill, so cast to float32
        data = data.type(torch.float32)
        return data, label


class HighwayDataModule(L.LightningDataModule):
    """
    Lightning datamodule for the FIOT Highway2 dataset

    Implements 5 key methods
    - prepare_data: things to do on 1 accelerator only (download, tokenize, etc)
    - setup: things to do on every accelerator (split dataset, etc)
    - train_dataloader: return the training dataloader
    - val_dataloader: return the validation dataloader
    - test_dataloader: return the test dataloader
    """

    def __init__(self, batch_size: int = 32, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        # this allows access to all hparams via self.hparams
        self.save_hyperparameters(logger=False)

    def setup(self, stage=None, root_dir: str = "DSET_FIOT_HIGHWAY2"):
        """
        called on every process in DDP

        We want to augment training data, not validation or test data.
        Since we want to split our training data into train/val, we need to create two
        Highway2Dataset instances and then pick specific indices for train/val splits.
        """
        data_train = Highway2Dataset(
            root_dir=root_dir,
            subset="train",
            transform=v2.Compose(
                [
                    LogNoise(noise_power_db=-90, p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                ]
            ),
        )
        data_val = Highway2Dataset(root_dir=root_dir, subset="train", transform=None)
        self.data_test = Highway2Dataset(root_dir=root_dir, subset="test", transform=None)

        # split train into train/val, stratified by class label
        train_indices, val_indices = train_test_split(
            np.arange(len(data_train)),
            test_size=0.1,
            stratify=[label for _, label in data_train.items],
            random_state=0xD00D1E,
        )
        self.data_train = torch.utils.data.Subset(data_train, train_indices)
        self.data_val = torch.utils.data.Subset(data_val, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
