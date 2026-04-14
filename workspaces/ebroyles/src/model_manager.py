import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from constants import LOGS_ROOT




class ModelManager:

    def __init__(self, model_name, max_epochs, model_class, model_body, loss_fn, log_every_n_steps=10):
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.model = model_class(model_body, loss_fn)
        self.logger = CSVLogger(LOGS_ROOT, name=model_name)
        self.trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", log_every_n_steps=log_every_n_steps, logger=self.logger)

    def train(self, train_loader):
        start = time.time()
        self.trainer.fit(self.model, train_loader)
        print(f"Train Time (s): {time.time() - start}")
        
    def test(self, test_loader):
        start = time.time()
        self.trainer.test(self.model, test_loader)
        print(f"Test Time (s): {time.time() - start}")

    def confustion_matrix(self, test_loader, num_classes):
        y_true, y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = self.model(batch_X)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        cmd = ConfusionMatrixDisplay(cm, display_labels=range(num_classes))
        plt.figure(figsize=(16,16))
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def relabeled_confusion_table(self, test_loader, num_classes, relabel = {}):
        #test_loader has original labels 0 to 8 (probably) before relabel
        #model predicts somthing like 0 or 1.
        #y-axis is the true label (0 to 8)
        #x-axis is for that true label what did the model predict
        #relabel = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, ...}
        pred_classes = max(relabel.values()) + 1
        conf = np.zeros((num_classes, pred_classes), dtype=int)
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = self.model(batch_X)
                preds = torch.argmax(logits, dim=1)
                y = batch_y.cpu().numpy()
                preds = preds.cpu().numpy()
                for true_label, pred_label in zip(y, preds):
                    mapped_true = relabel[int(true_label)]
                    conf[int(true_label), int(pred_label)] += 1
        row_labels = [f"True {i}" for i in range(num_classes)]
        col_labels = [f"Pred {i}" for i in range(pred_classes)]
        df = pd.DataFrame(conf, index=row_labels, columns=col_labels)
        print(df)

    def plot_logger(self):
        logger_path = os.path.join(LOGS_ROOT, self.model_name)
        if not os.path.exists(logger_path): raise FileNotFoundError(f"What is this path bro {logger_path}")
        version_dirs = [d for d in os.listdir(logger_path) if re.match(r"version_\d+", d)]
        if not version_dirs: raise FileNotFoundError("No version_X directories found")
        latest_version = max(version_dirs, key=lambda x: int(x.split("_")[1]))
        metrics_path = os.path.join(logger_path, latest_version, "metrics.csv")
        if not os.path.exists(metrics_path): raise FileNotFoundError(f"metrics.csv not found in {latest_version}")
        df = pd.read_csv(metrics_path)
        def plot_metric(ax, step_col, value_col, title):
            mask = df[value_col].notna()
            ax.plot(df.loc[mask, step_col], df.loc[mask, value_col])
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(value_col)
        fig, axes = plt.subplots(1, 2, figsize=(20, 4))
        plot_metric(axes[0], "step", "train_loss", "Train Loss vs Step")
        plot_metric(axes[1], "step", "train_acc",  "Train Accuracy vs Step")
        print("Logger Metics: ", metrics_path) 
        plt.tight_layout()
        plt.show()

    def visualize_model(self):
        #model.parameters has weights and biases
        pass




 
class MyNN(pl.LightningModule):
    def __init__(self, model_body, loss_fn, lr=1e-3):
        super().__init__()
        self.model_body = model_body
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=["model_body", "loss_fn"])

    def forward(self, x):
        return self.model_body(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean() #this finds the idx of the label with smallest logits
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean() #this finds the idx of the label with smallest logits
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)




    
    


# class MyMultiCheckNN(pl.LightningModule):
#     def __init__(self, model_body, loss_fn, lr=1e-3):
#         super().__init__()
#         self.model_body = model_body
#         self.loss_fn = loss_fn
#         self.save_hyperparameters(ignore=["model_body", "loss_fn"])

#     def forward(self, x):
#         return self.model_body(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss_fn(logits, y)
#         acc = (logits.argmax(dim=1) == y).float().mean() #this finds the idx of the label with smallest logits
#         self.log("train_loss", loss)
#         self.log("train_acc", acc, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss_fn(logits, y)
#         acc = (logits.argmax(1) == y).float().mean() #this finds the idx of the label with smallest logits

        
#         self.log("test_loss", loss)
#         self.log("test_acc", acc)

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.hparams.lr)

# class MyMultiCheckNN(pl.LightningModule):
#     ## loss is number of checks needed to get the prediction correct (average this for each batch to get the reported loss)
#     ## FInd the label_prob = [...9] each idx corresponds to label 0,1,2... sum(label_prob) = 1.
#     ## checks: look at label_prob with highest prob and check if this label matches if not continue, count number of checks needed to be correct.

#     def __init__(self, model_body, loss_fn, lr=1e-3):
#         super().__init__()
#         self.model_body = model_body
#         self.loss_fn = loss_fn
#         self.save_hyperparameters(ignore=["model_body", "loss_fn"])

#     def forward(self, x):
#         return self.model_body(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x) #  logit for one sample [...9 labels]
#         #convert the logits into label_prob
#         probs = torch.softmax(logits, dim=1)
        
        
        
        
#         loss = self.loss_fn(logits, y)
#         acc = (logits.argmax(dim=1) == y).float().mean() #this finds the idx of the label with smallest logits
#         self.log("train_loss", loss)
#         self.log("train_acc", acc, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss_fn(logits, y)
#         acc = (logits.argmax(1) == y).float().mean() #this finds the idx of the label with smallest logits
#         self.log("test_loss", loss)
#         self.log("test_acc", acc)

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.hparams.lr)



# class MyMultiCheckLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, probs, y):
#         # probs: (B, num labels) 
#         # y: (B,)
#         # B is batch size
        
#         pred_labels = np.argsort(probs) #  [label0, label2, label7, ... label1] (label0 is lowest prob, label1 is highest)
#         np.reverse(pred_labels)
#         #get the idx y appears, add this idx to the loss

#         return loss.mean()
        


    









