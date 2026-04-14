import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class PSDDataset(Dataset):
    def __init__(self, root_dir, index_file):
        self.items = []
        self.root = root_dir

        with open(index_file, "r") as f:
            for line in f:
                rel_path, lbl = line.strip().split()
                full = os.path.join(self.root, rel_path)
                self.items.append((full, int(lbl)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npy_path, label = self.items[idx]
        arr = np.load(npy_path).astype(np.float32)

        mn, mx = arr.min(), arr.max()
        arr = (arr - mn) / (mx - mn + 1e-6)

        arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        sample_id = os.path.basename(npy_path).replace(".npy", "")
        return arr, label, sample_id


class TinyCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def export_predictions_csv(model, loader, device, out_csv="val_predictions.csv"):
    model.eval()
    rows = []

    with torch.no_grad():
        for x, y, sample_ids in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            true_labels = y.cpu().numpy()

            for sid, true_y, pred_y in zip(sample_ids, true_labels, preds):
                rows.append({
                    "sample_id": sid,
                    "true_label": int(true_y),
                    "prediction": int(pred_y),
                    "correct": int(true_y == pred_y)
                })

    df = pd.DataFrame(rows)
    df = df.sort_values("sample_id").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved predictions to {out_csv}")
    print(df.head())
    return df


def run_training():
    base = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
    index_file = os.path.join(base, "train.txt")

    batch = 8
    epochs = 10
    lr = 1e-3
    n_classes = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = PSDDataset(base, index_file)
    if len(data) == 0:
        print("No samples found. Check file paths.")
        return

    first_x, _, _ = data[0]
    print("Detected input shape:", first_x.shape)

    labels = np.array([label for _, label in data.items])
    indices = np.arange(len(data))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_set = Subset(data, train_idx)
    val_set = Subset(data, val_idx)

    train_labels = labels[train_idx]
    class_counts = Counter(train_labels)

    print("\nTrain split class counts:")
    for c in range(n_classes):
        print(f"  class {c}: {class_counts.get(c, 0)}")

    counts = torch.tensor(
        [class_counts.get(c, 0) for c in range(n_classes)],
        dtype=torch.float32
    )
    class_weights = counts.sum() / torch.clamp(counts, min=1)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    print("\nClass weights used in loss:")
    for c in range(n_classes):
        print(f"  class {c}: {class_weights[c].item():.4f}")

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False)

    model = TinyCNN(num_classes=n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimzr = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        hits = 0
        samples = 0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} (Training)"):
            x = x.to(device)
            y = y.to(device)

            optimzr.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimzr.step()

            preds = out.argmax(1)

            total_loss += loss.item() * x.size(0)
            hits += (preds == y).sum().item()
            samples += y.size(0)

        train_loss = total_loss / samples
        train_acc = hits / samples

        model.eval()
        val_loss = 0.0
        val_hits = 0
        val_samples = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = loss_fn(out, y)
                preds = out.argmax(1)

                val_loss += loss.item() * x.size(0)
                val_hits += (preds == y).sum().item()
                val_samples += y.size(0)

                all_true.extend(y.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())

        v_loss = val_loss / val_samples
        v_acc = val_hits / val_samples

        cm = confusion_matrix(all_true, all_pred, labels=np.arange(n_classes))
        per_class_recall = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)
        mean_per_class_recall = per_class_recall.mean()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"\n┌──────────── Epoch {ep+1}/{epochs} ────────────┐\n"
            f"│   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}   │\n"
            f"│   Val   → Loss: {v_loss:.4f} | Acc: {v_acc:.4f}   │\n"
            f"│   Mean Per Class Recall: {mean_per_class_recall:.4f}          │\n"
            f"└──────────────────────────────────────────────┘\n"
        )

        print("Confusion Matrix:")
        print(cm)

        print("Per-class recall:")
        for c in range(n_classes):
            print(f"  class {c}: {per_class_recall[c]:.4f}")

    print("Finished training.")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model by validation accuracy: {best_val_acc:.4f}")

    export_predictions_csv(model, val_loader, device, out_csv="val_predictions.csv")


if __name__ == "__main__":
    run_training()