import os
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt


# -----------------
# Config (edit if needed)
# -----------------
ROOT_DIR = "/anvil/projects/x-cis220051/corporate/aerospace-rf/fiot_highway2-main"
TRAIN_TXT = os.path.join(ROOT_DIR, "train.txt")
TEST_TXT  = os.path.join(ROOT_DIR, "test.txt")

VAL_SPLIT = 0.15
SEED = 42

PLOT_DIR = "imbalance_plots"   # where PNGs will be saved


def load_txt(txt_path):
    paths, labels = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            paths.append(parts[0])
            labels.append(int(parts[1]))
    return paths, labels


def print_distribution(name, labels):
    counts = Counter(labels)
    total = len(labels)

    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"Total samples: {total}")

    classes = sorted(counts.keys())
    print("\nClass counts:")
    for c in classes:
        print(f"  class {c}: {counts[c]}")

    print("\nClass percentages:")
    for c in classes:
        print(f"  class {c}: {counts[c] / total * 100:.2f}%")

    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"\nImbalance ratio (max/min): {ratio:.2f}x")

    majority_class, majority_count = counts.most_common(1)[0]
    majority_baseline = majority_count / total
    print(f"Majority-class baseline accuracy: {majority_baseline:.4f} "
          f"(always predict class {majority_class})")

    if ratio <= 2:
        severity = "mild or balanced"
    elif ratio <= 5:
        severity = "moderate imbalance"
    elif ratio <= 10:
        severity = "strong imbalance"
    else:
        severity = "severe imbalance"
    print(f"Imbalance severity (rough): {severity}")


def _safe_name(name: str) -> str:
    # file-friendly name
    return (
        name.replace(" ", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace("=", "")
            .replace("/", "_")
    )


def plot_distribution(name, labels, save_dir=PLOT_DIR):
    """
    Saves:
      - bar chart of raw counts
      - bar chart of counts on log scale (useful for extreme imbalance)
      - bar chart of percentages
    """
    os.makedirs(save_dir, exist_ok=True)

    counts = Counter(labels)
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]
    total = len(labels)
    perc = [v / total * 100.0 for v in values]

    base = _safe_name(name)

    # 1) Raw counts
    plt.figure()
    plt.bar(classes, values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"{name} - Class Counts")
    out1 = os.path.join(save_dir, f"{base}_counts.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Log-scale counts
    plt.figure()
    plt.bar(classes, values)
    plt.yscale("log")
    plt.xlabel("Class")
    plt.ylabel("Count (log scale)")
    plt.title(f"{name} - Class Counts (Log Scale)")
    out2 = os.path.join(save_dir, f"{base}_counts_log.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Percentages
    plt.figure()
    plt.bar(classes, perc)
    plt.xlabel("Class")
    plt.ylabel("Percent (%)")
    plt.title(f"{name} - Class Percentages")
    out3 = os.path.join(save_dir, f"{base}_percentages.png")
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plots:\n  {out1}\n  {out2}\n  {out3}")


def make_train_val_split(labels, val_split=0.15, seed=42):
    """Deterministic split indices similar to random_split."""
    n = len(labels)
    val_size = int(n * val_split)
    train_size = n - val_size

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    return train_idx, val_idx


def main():
    if not os.path.exists(TRAIN_TXT):
        raise FileNotFoundError(f"TRAIN_TXT not found: {TRAIN_TXT}")

    # Train.txt distribution (full, before split)
    _, train_labels = load_txt(TRAIN_TXT)
    print_distribution("TRAIN.TXT (full)", train_labels)
    plot_distribution("TRAIN.TXT (full)", train_labels)

    # Train/Val split distributions
    train_idx, val_idx = make_train_val_split(train_labels, VAL_SPLIT, SEED)
    train_split_labels = [train_labels[i] for i in train_idx]
    val_split_labels = [train_labels[i] for i in val_idx]

    train_name = f"TRAIN SPLIT (val_split={VAL_SPLIT}, seed={SEED})"
    val_name = f"VAL SPLIT (val_split={VAL_SPLIT}, seed={SEED})"

    print_distribution(train_name, train_split_labels)
    plot_distribution(train_name, train_split_labels)

    print_distribution(val_name, val_split_labels)
    plot_distribution(val_name, val_split_labels)

    # Test.txt distribution (if exists)
    if os.path.exists(TEST_TXT):
        _, test_labels = load_txt(TEST_TXT)
        print_distribution("TEST.TXT (full)", test_labels)
        plot_distribution("TEST.TXT (full)", test_labels)
    else:
        print("\nNOTE: test.txt not found, skipping test distribution.")

    print(f"\nDone. Plots saved in: {PLOT_DIR}/")


if __name__ == "__main__":
    main()