import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# Dataset extractor
# ================================================================
def struct_extract(struct, classes):
    paths  = []
    labels = []
    valid = set(classes)
    shift = min(classes)

    for entry in struct:
        label = int(entry[1])
        if label in valid:
            path = entry[0].split('/')[1].split('.')[0]
            paths.append(path)
            labels.append(label - shift)  # shift to 0..N-1
    return paths, labels

# ================================================================
# Dataset class (generalizable)
# ================================================================
class HighwayDataset(Dataset):
    def __init__(self, paths, labels, data_root, resize,
                 norm_type=None, clip_range=(-140,0)):
        self.paths = paths
        self.labels = labels
        self.data_root = data_root
        self.resize = resize
        self.norm_type = norm_type
        self.clip_range = clip_range

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fname = self.paths[idx] + ".npy"
        arr = np.load(os.path.join(self.data_root, fname)) # 2D
        arr = np.clip(arr, *self.clip_range)

        if self.norm_type == "zero_one":
            mn, mx = self.clip_range
            arr = (arr - mn)/(mx-mn)
        elif self.norm_type == "zscore":
            arr = (arr - np.mean(arr)) / (np.std(arr)+1e-6)
        # else: raw

        img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        img = TF.resize(img, self.resize)
        label = int(self.labels[idx])
        return img, label

# ================================================================
# Flexible CNN
# ================================================================
class FlexibleCNN(pl.LightningModule):
    def __init__(self, conv_channels, num_classes,
                 lr, img_size, linear_width):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_c = 1
        h,w = img_size

        # Conv blocks
        for ch in conv_channels:
            layers += [
                nn.Conv2d(in_c, ch, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            h//=2; w//=2
            in_c = ch

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_c*h*w, linear_width))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(linear_width, num_classes))

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x,y = batch
        logits = self(x)
        loss = self.loss_fn(logits,y)
        acc = (logits.argmax(1)==y).float().mean()
        self.log("train_loss",loss); self.log("train_acc",acc)
        return loss

    def test_step(self, batch, _):
        x,y = batch
        logits = self(x)
        loss = self.loss_fn(logits,y)
        acc = (logits.argmax(1)==y).float().mean()
        self.log("test_loss",loss); self.log("test_acc",acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ================================================================
# Runner helper
# ================================================================
class ExperimentRunner:
    def __init__(self, data_root, train_txt, test_txt,
                 classes, resize, batch_size, lr, max_epochs,
                 normalization, conv_layers, linear_width,
                 model_name="experiment"):
        self.data_root=data_root
        self.train_txt=train_txt
        self.test_txt=test_txt
        self.classes=classes
        self.resize=resize
        self.batch_size=batch_size
        self.lr=lr
        self.max_epochs=max_epochs
        self.norm=normalization
        self.conv_layers=conv_layers
        self.linear_width=linear_width
        self.name=model_name

    def run(self):
        # load file structs
        train_struct = np.loadtxt(self.train_txt, dtype=str).tolist()
        test_struct  = np.loadtxt(self.test_txt,  dtype=str).tolist()

        tr_paths,tr_labels = struct_extract(train_struct, self.classes)
        te_paths,te_labels = struct_extract(test_struct,  self.classes)

        train_ds = HighwayDataset(tr_paths,tr_labels,self.data_root,
                                  self.resize,self.norm)
        test_ds  = HighwayDataset(te_paths,te_labels,self.data_root,
                                  self.resize,self.norm)

        train_ld = DataLoader(train_ds,batch_size=self.batch_size,
                              shuffle=True,num_workers=4)
        test_ld  = DataLoader(test_ds,batch_size=self.batch_size,
                              num_workers=4)

        model = FlexibleCNN(
            conv_channels=self.conv_layers,
            num_classes=len(self.classes),
            lr=self.lr,
            img_size=self.resize,
            linear_width=self.linear_width
        )

        logger = CSVLogger("logs", name=self.name)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            accelerator="auto"
        )

        start=time.time()
        trainer.fit(model,train_ld)
        trainer.test(model,test_ld)
        end=time.time()
        print(f"\nTotal train+test time = {end-start:.2f}s\n")

        # confusion matrix
        y_true=[]; y_pred=[]
        model.eval()
        with torch.no_grad():
            for x,y in test_ld:
                logits=model(x)
                preds=logits.argmax(1)
                y_true+=y.cpu().tolist()
                y_pred+=preds.cpu().tolist()

        cm = confusion_matrix(y_true,y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        # Accuracy/Loss curves
        metrics = np.loadtxt(f"logs/{self.name}/metrics.csv",delimiter=',',
                             skiprows=1,dtype=str)
        names   = np.loadtxt(f"logs/{self.name}/metrics.csv",delimiter=',',
                             max_rows=1,dtype=str)
        idx = {n:i for i,n in enumerate(names)}

        steps = metrics[:,idx["step"]].astype(float)
        tloss = metrics[:,idx["train_loss"]].astype(float)
        tacc  = metrics[:,idx.get("train_acc",idx.get("train_accuracy"))].astype(float)
        # test metrics only appear sparsely; just plot train curves
        plt.figure()
        plt.plot(steps,tloss,label="train_loss")
        plt.plot(steps,tacc, label="train_acc")
        plt.legend(); plt.title("Training Curves")
        plt.show()

        print(f"Final Train Loss: {tloss[-1]:.3f}")
        print(f"Final Train Acc : {tacc[-1]*100:.2f}%")
