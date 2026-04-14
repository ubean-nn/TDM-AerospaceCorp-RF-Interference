#!/usr/bin/env python3
import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
import radiomana 
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from radiomana.datasets import HighwayDataModule

# ---------- CUSTOM MODEL ARCHITECTURE ----------
class CustomInterferenceCNN(L.LightningModule):
    def __init__(self, num_classes, input_channels=2, input_length=1024, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        
        # 1. Feature Extractor (Convolutional Blocks)
        # We use Conv1d because Radio data is typically time-series I/Q data
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Forces output to (Batch, 256, 1) regardless of input length
        )

        # 2. Classifier (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Helps prevent overfitting
            nn.Linear(128, num_classes)
        )

        # 3. Metrics
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        # We keep this strictly for the final logging block in your script
        self.confmat_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        # x shape: [batch_size, 2, 1024]
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc_metric(preds, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Log standard metrics
        self.log("test_loss", loss)
        self.log("test_acc", self.acc_metric(preds, y))
        self.log("test_f1", self.f1_metric(preds, y))
        
        # Update confusion matrix for later retrieval
        self.confmat_metric.update(preds, y)

    @property
    def confmat(self):
        # This property ensures your existing code "print(best_model.confmat)" works
        # We move it to CPU to avoid print errors
        return self.confmat_metric.compute().cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------- RESTORED HELPER FUNCTION ----------
def get_data_module(dataset_path=None, batch_size=32):
    # Hardcode the path specifically for the Anvil cluster
    os.environ["DSET_FIOT_HIGHWAY2"] = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
    
    print(f"Dataset path set to: {os.environ['DSET_FIOT_HIGHWAY2']}")

    # Check if path actually exists to be safe
    if not os.path.exists(os.environ["DSET_FIOT_HIGHWAY2"]):
        print(f"WARNING: Path {os.environ['DSET_FIOT_HIGHWAY2']} not found!")

    dm = radiomana.HighwayDataModule(batch_size=batch_size)
    dm.setup()
    return dm

# ---------- MAIN TRAINING LOOP ----------
def train():
    shortname = "highway_custom_cnn" 
    
    # 1. Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="radiomana-highway",
        name=shortname,
        log_model=True
    )

    # 2. Setup Data
    dmodule = get_data_module(batch_size=32) 
    
    # --- FIX 1: Handle Subset Attribute Error ---
    if hasattr(dmodule.data_train, "dataset"):
        full_dataset = dmodule.data_train.dataset
    else:
        full_dataset = dmodule.data_train

    class_labels = full_dataset.class_labels
    num_classes = len(class_labels)
    # --------------------------------------------

    # --- FIX 2: Auto-Detect Input Channels ---
    # We grab one batch from the training loader to see the actual shape
    print("Inspecting data shape...")
    train_loader = dmodule.train_dataloader()
    first_batch = next(iter(train_loader))
    x_sample, _ = first_batch
    
    # Shape is usually [Batch, Channels, Length]
    detected_channels = x_sample.shape[1] 
    detected_length = x_sample.shape[2]
    
    print(f"Data Shape Detected: {x_sample.shape}")
    print(f"Auto-configuring model: input_channels={detected_channels}, input_length={detected_length}")
    # -----------------------------------------
    
    # 3. Setup Custom Model with DETECTED channels
    model = CustomInterferenceCNN(
        num_classes=num_classes,
        input_channels=detected_channels, # <--- uses 512 automatically now
        learning_rate=1e-3
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename=shortname + "-{epoch:03d}-{val_loss:05f}",
            save_top_k=1,
            mode="min",
        ),
        RichProgressBar(),
    ]

    # 4. Configure Trainer with WandB
    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=30, 
        precision=32, 
        callbacks=callbacks,
        logger=wandb_logger
    )
    
    wandb_logger.watch(model, log="all")

    # 5. Train
    trainer.fit(model, datamodule=dmodule)

    # 6. Test Best Model
    print(f"rewinding to best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    best_model = CustomInterferenceCNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    results = trainer.test(best_model, datamodule=dmodule)[0]

    print(f"test loss: {results['test_loss']:0.3f}")
    print(f"test f1: {results['test_f1']:0.3f}")
    print(f"test acc: {results['test_acc']:8.3%}")

    # 7. Manual Logging 
    confmat = best_model.confmat
    print("confusion matrix:\n", confmat)
    
    class_acc = torch.diagonal(confmat) / confmat.sum(dim=1)
    print("test per-class accuracy:")
    
    per_class_metrics = {}
    
    for cdx, class_label in enumerate(class_labels):
        acc = class_acc[cdx]
        print(f"  class {cdx} ({class_label:<22s}): {acc:8.3%}")
        per_class_metrics[f"acc_class_{class_label}"] = acc

    wandb.log(per_class_metrics)

    # 8. Save Weights
    save_filename = f"{shortname}-loss={results['test_loss']:0.3f}.pt"
    torch.save(best_model.state_dict(), save_filename)
    wandb.save(save_filename)
    
    wandb.finish()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        # This ensures the error is printed to the log even if it crashes
        print(f"\nCRITICAL ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()
        
        # Mark the run as failed in WandB explicitly so you know why
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise e