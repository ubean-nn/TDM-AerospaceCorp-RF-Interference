#!/usr/bin/env python3

import lightning as L
import torch
import torchmetrics
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, resnet18


class HighwayBaselineModel(L.LightningModule):
    """Example model architecture for the FIOT datasets"""

    def __init__(self, num_classes: int = 9):
        super(HighwayBaselineModel, self).__init__()
        # add channels dimension and project to 3 channels
        self.reshape = Rearrange("batchsize height width -> batchsize 1 height width")
        # pointwise projection to 3 channels
        self.project = nn.Conv2d(1, 3, kernel_size=1)
        # submodel selection
        # self.submodel = resnet18()
        self.submodel = mobilenet_v3_large()
        self.head = nn.Linear(1000, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes, task="multiclass")
        self.test_f1 = torchmetrics.F1Score(num_classes=num_classes, average="macro", task="multiclass")

    def forward(self, x):
        x = self.reshape(x)
        x = self.project(x)
        x = self.submodel(x)
        x = self.head(x)
        return x

    def step(self, batch, batch_idx):
        x, y_true = batch
        y_hat = self(x)
        return self.criterion(y_hat, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """similar to step, but we also update confusion matrix and F1 score"""
        x, y_true = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.test_confmat.update(preds, y_true)
        self.test_f1.update(preds, y_true)
        loss = self.criterion(y_hat, y_true)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        self.confmat = self.test_confmat.compute()
        self.log("test_acc", torch.sum(torch.diagonal(self.confmat)) / torch.sum(self.confmat).item())
        self.log("test_f1", self.test_f1.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = HighwayBaselineModel()
    print(model)
    sample_input = torch.randn(16, 512, 243)
    output = model(sample_input)
    print(output.shape)
