import time
import pytorch_lightning as pl
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

class NN1(pl.LightningModule):
    def __init__(self, model, loss_fn, lr=1e-3):
        super().__init__()
        self.save_hyperparameters() #stores all inputs inside of self.hparams
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = self.loss_fn(logits, y)
    #     acc = (logits.argmax(dim=1) == y).float().mean()
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


