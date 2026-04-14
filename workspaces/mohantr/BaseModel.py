# --- Setup & Install radiomana ---
# Run this once per notebook session.

# --- Imports ---
import os
import torch
import radiomana
import matplotlib.pyplot as plt

# --- Set the dataset environment variable ---
folder_path = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
os.environ["DSET_FIOT_HIGHWAY2"] = folder_path

print("✅ radiomana installed and dataset path set!")
print("radiomana version:", radiomana.__version__)
print("CUDA available:", torch.cuda.is_available())

import radiomana

# Create dataset
dataset = radiomana.Highway2Dataset()

print("✅ Dataset loaded successfully!")
print("Number of samples:", len(dataset))

# Unpack one sample
psd, label = dataset[0]

print("PSD type:", type(psd))
print("Label type:", type(label))
print("PSD shape:", psd.shape if hasattr(psd, 'shape') else "Not a tensor/array")
print("Label:", label)

import matplotlib.pyplot as plt

psd, label = dataset[0]

plt.figure(figsize=(8, 4))
plt.imshow(psd, aspect='auto', origin='lower', cmap='viridis')
plt.title(f"Sample 0 — Label: {label}")
plt.xlabel("Time (index)")
plt.ylabel("Frequency (index)")
plt.colorbar(label="Power (dB)")
plt.show()

# --- Create DataModule for ML training (fixed) ---

# Create DataModule with 0 workers to avoid shared memory errors on Anvil
dmodule = radiomana.HighwayDataModule(num_workers=0)
dmodule.setup()

# Get dataloaders (no arguments!)
train_loader = dmodule.train_dataloader()
val_loader = dmodule.val_dataloader()

# Inspect one batch
batch = next(iter(train_loader))
print("✅ DataModule loaded!")
print("Train batch PSD shape:", batch[0].shape)
print("Train batch labels shape:", batch[1].shape)

import torch
import torch.nn as nn
import torch.optim as optim
from radiomana.models import HighwayBaselineModel

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of classes from the dataset
num_classes = len(set(dataset[i][1] for i in range(len(dataset))))
print("Number of classes:", num_classes)

# Initialize model
model = HighwayBaselineModel(num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
num_epochs = 3  # for demo

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for psd_batch, labels in dmodule.train_dataloader():
        psd_batch, labels = psd_batch.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(psd_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * psd_batch.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("✅ Training complete!")

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Evaluate on validation set ---
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for psd_batch, labels in dmodule.val_dataloader():
        psd_batch = psd_batch.to(device)
        labels = labels.to(device)

        outputs = model(psd_batch)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

# Concatenate batches
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# --- Confusion matrix ---
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:\n", cm)

# --- Plot ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Validation Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved to confusion_matrix.png")

# --- Per-class accuracy ---
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nPer-class accuracy:")
for i, acc in enumerate(per_class_acc):
    print(f"Class {i}: {acc:.4f}")