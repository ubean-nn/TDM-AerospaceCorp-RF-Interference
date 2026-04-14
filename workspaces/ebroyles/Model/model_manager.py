## Terrible, to be remade

import os
import re
import time
import numpy as np
import torch
import torch.nn as torchnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nn import NN1


class ModelManager:
    LOGS_PATH = "logs"

    def show_confusion_matrix(self, test_loader, model, num_classes):
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        cm = confusion_matrix(y_true, y_pred)
        cmd = ConfusionMatrixDisplay(cm, display_labels=range(num_classes))
        
        plt.figure(figsize=(16,16))
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        

    def show_logger_data(self, logger_path):
        """
        Finds the latest Lightning CSVLogger version and plots:
        - train_loss vs step
        - train_acc vs step
        - val_loss vs step
        - val_acc vs step
        """
        version_dirs = [
            d for d in os.listdir(logger_path)
            if re.match(r"version_\d+", d)
        ]
    
        if not version_dirs:
            raise FileNotFoundError("No version_X directories found")
    
        latest_version = max(
            version_dirs,
            key=lambda x: int(x.split("_")[1])
        )
    
        metrics_path = os.path.join(
            logger_path, latest_version, "metrics.csv"
        )
    
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"metrics.csv not found in {latest_version}")
        df = pd.read_csv(metrics_path)
    
        def plot_metric(ax, step_col, value_col, title):
            mask = df[value_col].notna()
            ax.plot(df.loc[mask, step_col], df.loc[mask, value_col])
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(value_col)
    
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        plot_metric(axes[0], "step", "train_loss", "Train Loss vs Step")
        plot_metric(axes[1], "step", "train_acc",  "Train Accuracy vs Step")
        print(metrics_path)
        plt.tight_layout()
        plt.show()
            
    def run_model_v1(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v1"
        num_classes = 9
        max_epoch = 20
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

    def run_model_v2(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v2"
        num_classes = 9
        max_epoch = 25
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

    def run_model_v2_1(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v2_1"
        num_classes = 9
        max_epoch = 35
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn
        
    def run_model_v2_2(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v2_2"
        num_classes = 9
        max_epoch = 45
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

    def run_model_v2_3(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v2_3"
        num_classes = 9
        max_epoch = 80
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

    def run_model_v3(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v3"
        num_classes = 9
        max_epoch = 25
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 1024),
            torchnn.ReLU(),
            torchnn.Linear(1024, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, 128),
            torchnn.ReLU(),
            torchnn.Linear(128, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

    def run_model_v3_1(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v3_1"
        num_classes = 9
        max_epoch = 50
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 1024),
            torchnn.ReLU(),
            torchnn.Linear(1024, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, 128),
            torchnn.ReLU(),
            torchnn.Linear(128, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

    def run_model_v4(self, train_loader, test_loader, num_features):
        MODEL_NAME = "model_v4"
        num_classes = 9
        max_epoch = 60
        model_body = torchnn.Sequential(
            torchnn.Linear(num_features, 1024),
            torchnn.ReLU(),
            torchnn.Linear(1024, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 256),
            torchnn.ReLU(),
            torchnn.Linear(256, 128),
            torchnn.ReLU(),
            torchnn.Linear(128, 64),
            torchnn.ReLU(),
            torchnn.Linear(64, num_classes)
        )
        loss_fn = torchnn.CrossEntropyLoss()
        model = NN1(model_body, loss_fn)
        csv_logger = CSVLogger(self.LOGS_PATH, name=MODEL_NAME)  
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            logger=csv_logger  
        )
        start = time.time()
        # trainer.fit(model, train_loader, test_loader) (if you want a val step)
        trainer.fit(model, train_loader)
        trainer.test(model, test_loader)
        print(f"Train and Test Time: {time.time() - start} seconds")
        self.show_confusion_matrix(test_loader, model, num_classes)
        self.show_logger_data(os.path.join(self.LOGS_PATH, MODEL_NAME))
        return model, trainer, loss_fn

            
        





