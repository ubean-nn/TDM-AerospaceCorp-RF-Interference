import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import gc
 
# ------------------------------------------------------
# GPU SETUP
# ------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("WARNING: CUDA not available, using CPU")
 
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
 
 
# ------------------------------------------------------
# Label mappings
#
# Fine labels:  0, 1, 2, 3, 4, 5, 6, 7, 8
# Coarse groups:
#   Group 0 <- labels 0, 1, 2   ("low")
#   Group 1 <- labels 3, 4, 5, 6 ("mid")
#   Group 2 <- labels 7, 8       ("high")
# ------------------------------------------------------
FINE_TO_COARSE = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2,
}
NUM_COARSE = 3
NUM_FINE   = 9
 
# Within each coarse group, what are the fine label indices?
# Used to remap fine labels to local indices (0-based within group)
COARSE_TO_FINE_LABELS = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6],
    2: [7, 8],
}
 
def fine_to_coarse(label):
    return FINE_TO_COARSE[label]
 
def fine_to_local(label):
    """Map fine label to its 0-based index within its coarse group."""
    coarse = FINE_TO_COARSE[label]
    return COARSE_TO_FINE_LABELS[coarse].index(label)
 
 
# ------------------------------------------------------
# Downsampling utility
#
# Balances the dataset so no class dominates training.
# max_per_class: hard cap per fine label.
# If a class has fewer samples it keeps all of them.
# ------------------------------------------------------
def downsample_indices(samples, max_per_class=2000, seed=42):
    rng = random.Random(seed)
    label_to_indices = {}
    for i, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(i)
 
    kept = []
    for label, idxs in sorted(label_to_indices.items()):
        if len(idxs) > max_per_class:
            idxs = rng.sample(idxs, max_per_class)
        kept.extend(idxs)
        print(f"  Fine label {label}: keeping {len(idxs)} samples")
 
    return kept
 
 
# ------------------------------------------------------
# Dataset  (stores BOTH fine and coarse labels)
# ------------------------------------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, folder_path, txt_file, mean=None, std=None,
                 augment=False, max_per_class=None):
        self.folder_path = folder_path
        self.augment = augment
        self.samples = []  # (path, fine_label)
 
        with open(txt_file, "r") as f:
            for line in f:
                file_path, label = line.strip().split()
                full_path = os.path.join(folder_path, file_path)
                self.samples.append((full_path, int(label)))
 
        # Optional downsampling
        if max_per_class is not None:
            print(f"\nDownsampling to max {max_per_class} per class...")
            kept_idx = downsample_indices(self.samples, max_per_class)
            self.samples = [self.samples[i] for i in kept_idx]
            print(f"Total after downsample: {len(self.samples)}\n")
 
        # Compute or reuse mean/std
        if mean is None or std is None:
            print("Calculating dataset mean/std (streaming)...")
            total_sum = 0.0
            total_sq  = 0.0
            total_n   = 0
            for path, _ in tqdm(self.samples):
                arr = np.load(path).astype(np.float32)
                total_sum += arr.sum()
                total_sq  += (arr ** 2).sum()
                total_n   += arr.size
            self.mean = total_sum / total_n
            var       = total_sq / total_n - self.mean ** 2
            self.std  = float(np.sqrt(max(var, 1e-12)))
        else:
            self.mean = float(mean)
            self.std  = float(std)
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        file_path, fine_label = self.samples[idx]
        data = np.load(file_path).astype(np.float32)
 
        # Normalise
        data = (data - self.mean) / (self.std + 1e-6)
        tensor = torch.from_numpy(data).unsqueeze(0)   # (1, H, W)
 
        # Augmentation (training only)
        if self.augment:
            H, W = tensor.size(1), tensor.size(2)
            if random.random() < 0.5 and H >= 12:
                f0 = random.randint(0, H - 10)
                tensor[:, f0:f0+10, :] = 0          # frequency masking
            if random.random() < 0.5 and W >= 12:
                t0 = random.randint(0, W - 10)
                tensor[:, :, t0:t0+10] = 0          # time masking
            if random.random() < 0.5:
                tensor += 0.02 * torch.randn_like(tensor)
 
        coarse_label = fine_to_coarse(fine_label)
        local_label  = fine_to_local(fine_label)   # 0-based within group
 
        return tensor, coarse_label, fine_label, local_label
 
 
# ------------------------------------------------------
# Model: Shared backbone + dual heads
#
# "School" design:
#   - One backbone learns shared RF features
#   - Coarse head: 3-class (macro groups)  — trains first / guides backbone
#   - Fine head  : 9-class (all labels)    — trains jointly but with less weight early
#
# Pruning is applied to the backbone conv layers at the end via
# torch.nn.utils.prune (structured L1 on output channels).
# ------------------------------------------------------
class HierarchicalCNN(nn.Module):
    def __init__(self, input_shape=(1, 64, 64),
                 num_coarse=NUM_COARSE, num_fine=NUM_FINE):
        super().__init__()
 
        # ---- Shared backbone (slightly wider than original) ----
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
 
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
 
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
 
            # Block 4 (extra depth for fine discrimination)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # fixed spatial size regardless of input
        )
 
        flatten_size = 256 * 4 * 4   # 4096
 
        # ---- Coarse head (macro groups: 0, 1, 2) ----
        self.coarse_head = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_coarse),
        )
 
        # ---- Fine head (all 9 classes) ----
        # Gets both backbone features AND a stop-gradient coarse embedding
        # so it can "read" the coarse decision without polluting coarse gradients.
        self.coarse_embed = nn.Linear(num_coarse, 32)   # coarse logits -> embedding
 
        self.fine_head = nn.Sequential(
            nn.Linear(flatten_size + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_fine),
        )
 
    def forward(self, x):
        feats         = self.backbone(x)
        feats_flat    = feats.flatten(1)                # (B, 4096)
 
        coarse_logits = self.coarse_head(feats_flat)    # (B, 3)
 
        # Stop gradient: fine head sees coarse signal but can't backprop through it
        coarse_emb    = self.coarse_embed(coarse_logits.detach())  # (B, 32)
        fine_input    = torch.cat([feats_flat, coarse_emb], dim=1) # (B, 4096+32)
        fine_logits   = self.fine_head(fine_input)                 # (B, 9)
 
        return coarse_logits, fine_logits
 
 
# ------------------------------------------------------
# Hierarchical loss
#
# Total loss = w_coarse * L_coarse + w_fine * L_fine
#
# "School" schedule: early epochs emphasise coarse loss so the
# backbone first learns macro structure, then fine weight ramps up.
# ------------------------------------------------------
def hierarchical_loss(coarse_logits, fine_logits,
                      coarse_labels, fine_labels,
                      epoch, total_epochs,
                      label_smoothing=0.1):
 
    # Weight schedule: coarse starts at 0.8, fine ramps from 0.2 -> 0.7
    progress   = epoch / max(total_epochs - 1, 1)
    w_coarse   = 0.8 - 0.3 * progress   # 0.8 -> 0.5
    w_fine     = 0.2 + 0.5 * progress   # 0.2 -> 0.7
 
    ce_coarse = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ce_fine   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
 
    loss_c = ce_coarse(coarse_logits, coarse_labels)
    loss_f = ce_fine(fine_logits, fine_labels)
 
    total = w_coarse * loss_c + w_fine * loss_f
    return total, loss_c, loss_f, w_coarse, w_fine
 
 
# ------------------------------------------------------
# Evaluation
# ------------------------------------------------------
def evaluate(model, loader):
    model.eval()
 
    coarse_correct = fine_correct = total = 0
    coarse_class_correct = np.zeros(NUM_COARSE)
    coarse_class_total   = np.zeros(NUM_COARSE)
    fine_class_correct   = np.zeros(NUM_FINE)
    fine_class_total     = np.zeros(NUM_FINE)
 
    with torch.no_grad():
        for inputs, coarse_labels, fine_labels, _ in loader:
            inputs        = inputs.to(device, non_blocking=True)
            coarse_labels = coarse_labels.to(device, non_blocking=True)
            fine_labels   = fine_labels.to(device, non_blocking=True)
 
            coarse_logits, fine_logits = model(inputs)
 
            coarse_preds = coarse_logits.argmax(1)
            fine_preds   = fine_logits.argmax(1)
 
            coarse_correct += (coarse_preds == coarse_labels).sum().item()
            fine_correct   += (fine_preds   == fine_labels).sum().item()
            total          += inputs.size(0)
 
            for c in range(NUM_COARSE):
                mask = (coarse_labels == c)
                coarse_class_total[c]   += mask.sum().item()
                coarse_class_correct[c] += (coarse_preds[mask] == c).sum().item()
 
            for c in range(NUM_FINE):
                mask = (fine_labels == c)
                fine_class_total[c]   += mask.sum().item()
                fine_class_correct[c] += (fine_preds[mask] == c).sum().item()
 
    coarse_acc = coarse_correct / max(total, 1)
    fine_acc   = fine_correct   / max(total, 1)
 
    per_coarse = [
        coarse_class_correct[c] / coarse_class_total[c]
        if coarse_class_total[c] > 0 else float("nan")
        for c in range(NUM_COARSE)
    ]
    per_fine = [
        fine_class_correct[c] / fine_class_total[c]
        if fine_class_total[c] > 0 else float("nan")
        for c in range(NUM_FINE)
    ]
 
    return coarse_acc, fine_acc, per_coarse, per_fine
 
 
# ------------------------------------------------------
# Structured pruning
#
# Removes the weakest output channels from each Conv2d in
# the backbone based on L1 norm of their weights.
# amount=0.2 removes 20% of channels per layer.
# Call AFTER training; then fine-tune for a few epochs.
# ------------------------------------------------------
def prune_backbone(model, amount=0.2):
    import torch.nn.utils.prune as prune
    print(f"\nApplying structured L1 pruning ({amount*100:.0f}% per conv layer)...")
    for module in model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight",
                                amount=amount, n=1, dim=0)
            prune.remove(module, "weight")   # make permanent
    print("Pruning applied.\n")
    return model
 
 
# ------------------------------------------------------
# Training
# ------------------------------------------------------
def train_model():
    folder_path = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
    txt_file    = os.path.join(folder_path, "train.txt")
 
    # ---- Hyperparameters ----
    BATCH_SIZE      = 32
    EPOCHS          = 20       # more epochs; scheduler handles early stop
    LR              = 5e-4
    WEIGHT_DECAY    = 1e-4
    MAX_PER_CLASS   = 2000     # downsampling cap per fine label
    PRUNE_AMOUNT    = 0.20     # 20% channel pruning after main training
    FINETUNE_EPOCHS = 5        # epochs to fine-tune after pruning
    PATIENCE        = 5        # early stopping patience
 
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
 
    # ---- Build dataset (with downsampling) ----
    base_ds = SpectrogramDataset(
        folder_path, txt_file,
        max_per_class=MAX_PER_CLASS
    )
    mean, std = base_ds.mean, base_ds.std
    print(f"Dataset mean={mean:.6f}  std={std:.6f}")
 
    labels      = [lbl for _, lbl in base_ds.samples]
    all_indices = np.arange(len(base_ds))
 
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=np.array(labels),
        random_state=42
    )
 
    # Augmented training set, clean val set
    train_ds_aug = SpectrogramDataset(
        folder_path, txt_file,
        mean=mean, std=std,
        augment=True,
        max_per_class=MAX_PER_CLASS
    )
    train_ds = Subset(train_ds_aug, train_idx)
    val_ds   = Subset(base_ds, val_idx)
 
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda")
    )
 
    # ---- Model ----
    sample, _, _, _ = base_ds[0]
    model = HierarchicalCNN(input_shape=sample.shape).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
 
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
 
    best_fine_acc  = 0.0
    patience_count = 0
 
    # ================================================================
    # Phase 1: Main training (coarse -> fine school schedule)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: Hierarchical Training")
    print("="*60)
 
    for epoch in range(EPOCHS):
        model.train()
        running_loss = running_lc = running_lf = 0.0
        coarse_correct = fine_correct = total = 0
 
        for inputs, coarse_labels, fine_labels, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):
            inputs        = inputs.to(device, non_blocking=True)
            coarse_labels = coarse_labels.to(device, non_blocking=True)
            fine_labels   = fine_labels.to(device, non_blocking=True)
 
            optimizer.zero_grad(set_to_none=True)
            coarse_logits, fine_logits = model(inputs)
 
            loss, lc, lf, wc, wf = hierarchical_loss(
                coarse_logits, fine_logits,
                coarse_labels, fine_labels,
                epoch, EPOCHS
            )
 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
 
            running_loss += loss.item() * inputs.size(0)
            running_lc   += lc.item()  * inputs.size(0)
            running_lf   += lf.item()  * inputs.size(0)
 
            coarse_correct += (coarse_logits.argmax(1) == coarse_labels).sum().item()
            fine_correct   += (fine_logits.argmax(1)   == fine_labels).sum().item()
            total          += inputs.size(0)
 
        scheduler.step()
 
        train_loss  = running_loss / max(total, 1)
        train_lc    = running_lc   / max(total, 1)
        train_lf    = running_lf   / max(total, 1)
        train_cacc  = coarse_correct / max(total, 1)
        train_facc  = fine_correct   / max(total, 1)
 
        val_cacc, val_facc, per_coarse, per_fine = evaluate(model, val_loader)
 
        print("\n" + "-"*60)
        print(f"Epoch {epoch+1:02d}  |  w_coarse={wc:.2f}  w_fine={wf:.2f}")
        print(f"  Train -> Loss={train_loss:.4f}  Coarse={train_cacc:.4f}  Fine={train_facc:.4f}")
        print(f"  Val   -> Coarse={val_cacc:.4f}  Fine={val_facc:.4f}")
        print(f"  Val coarse per-group : {[f'{x:.3f}' for x in per_coarse]}")
        print(f"  Val fine   per-class : {[f'{x:.3f}' for x in per_fine]}")
 
        if val_facc > best_fine_acc:
            best_fine_acc  = val_facc
            patience_count = 0
            torch.save({
                "epoch":        epoch + 1,
                "model_state":  model.state_dict(),
                "opt_state":    optimizer.state_dict(),
                "mean":         mean,
                "std":          std,
                "val_fine_acc": val_facc,
                "val_coarse_acc": val_cacc,
            }, "checkpoint_best.pt")
            print("  ✓ Saved best checkpoint")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {PATIENCE} epochs).")
                break
 
        gc.collect()
 
    # ================================================================
    # Phase 2: Structured pruning + fine-tune
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: Pruning + Fine-Tune")
    print("="*60)
 
    # Load best weights before pruning
    ckpt  = torch.load("checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = prune_backbone(model, amount=PRUNE_AMOUNT)
    model = model.to(device)
 
    pruned_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters after pruning: {pruned_params:,}  "
          f"({100*(1-pruned_params/total_params):.1f}% reduction)")
 
    ft_optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
    ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-7
    )
    best_ft_acc = 0.0
 
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        running_loss = coarse_correct = fine_correct = total = 0
 
        for inputs, coarse_labels, fine_labels, _ in tqdm(
            train_loader, desc=f"FineTune {epoch+1}/{FINETUNE_EPOCHS}"
        ):
            inputs        = inputs.to(device, non_blocking=True)
            coarse_labels = coarse_labels.to(device, non_blocking=True)
            fine_labels   = fine_labels.to(device, non_blocking=True)
 
            ft_optimizer.zero_grad(set_to_none=True)
            coarse_logits, fine_logits = model(inputs)
 
            # Use final-epoch weights (fully fine-focused)
            loss, lc, lf, wc, wf = hierarchical_loss(
                coarse_logits, fine_logits,
                coarse_labels, fine_labels,
                EPOCHS - 1, EPOCHS          # pin to last epoch weights
            )
 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            ft_optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
 
        ft_scheduler.step()
 
        val_cacc, val_facc, per_coarse, per_fine = evaluate(model, val_loader)
        print(f"\nFineTune {epoch+1} | Val Coarse={val_cacc:.4f}  Fine={val_facc:.4f}")
        print(f"  Per coarse: {[f'{x:.3f}' for x in per_coarse]}")
        print(f"  Per fine  : {[f'{x:.3f}' for x in per_fine]}")
 
        if val_facc > best_ft_acc:
            best_ft_acc = val_facc
            torch.save({
                "epoch":          EPOCHS + epoch + 1,
                "model_state":    model.state_dict(),
                "mean":           mean,
                "std":            std,
                "val_fine_acc":   val_facc,
                "val_coarse_acc": val_cacc,
                "pruned":         True,
                "prune_amount":   PRUNE_AMOUNT,
            }, "checkpoint_pruned_best.pt")
            print("  ✓ Saved pruned checkpoint")
 
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Best pre-pruning  val fine acc : {best_fine_acc:.4f}")
    print(f"  Best post-pruning val fine acc : {best_ft_acc:.4f}")
    print("="*60)
 
 
# ------------------------------------------------------
# Inference helper  (loads checkpoint and predicts)
# ------------------------------------------------------
def predict(checkpoint_path, spectrogram_path):
    """
    Returns (coarse_pred, fine_pred) for a single .npy spectrogram file.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = HierarchicalCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
 
    mean, std = ckpt["mean"], ckpt["std"]
    arr  = np.load(spectrogram_path).astype(np.float32)
    data = (arr - mean) / (std + 1e-6)
    x    = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
 
    with torch.no_grad():
        coarse_logits, fine_logits = model(x)
 
    coarse_pred = coarse_logits.argmax(1).item()
    fine_pred   = fine_logits.argmax(1).item()
    return coarse_pred, fine_pred
 
 
if __name__ == "__main__":
    train_model()