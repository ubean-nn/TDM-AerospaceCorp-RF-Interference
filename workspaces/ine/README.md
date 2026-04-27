# `ine` workspace

This folder holds RF interference experiments for the Highway2 / GNSS jamming project: custom CNNs, an **Audio Spectrogram Transformer (AST)** pipeline, earlier **WandB + Lightning** CNN prototypes (“gurt” / smoothing variants), and small utilities. Most scripts assume **PyTorch** and either **`pytorch-lightning`** or **`lightning`**, plus dataset layout from the project’s `train.txt` / `val.txt` / `test.txt` splits.

**Paths:** Several files default to **Anvil cluster** locations (e.g. `fiot_highway2-main`, `ine/sbatch`). Before running locally, search for `DEFAULT_CLUSTER_` / `ROOT_DIR` / hardcoded paths and point them at your copy of the dataset.

---

## Main training entry points

### `final_model.py` — cluster-oriented CNN + optional SMOTE

- **`CNNV6`:** From-scratch **2D** CNN on tensors shaped like **`(1, 512, 243)`** (`INPUT_SHAPE`): conv blocks, pooling, **DropPath**-style stochastic depth, bilinear downscale early, then two linear layers for **9 classes**.
- **`LightningCNN`:** PyTorch Lightning module wrapping `CNNV6` with train/val/test steps, **CSV test predictions**, confusion matrix, and classification report on the test epoch.
- **`HighwayDataset`:** Reads split lines (`path` + label), loads `.npy`, optional **Gaussian noise** augmentation, returns `(x, y, path)` for traceability.
- **`apply_smote` / `random_oversample`:** Optional **class balancing** via `imblearn`’s SMOTE (or random oversample fallback) on flattened features, then reshaped back to `INPUT_SHAPE` for a `TensorDataset`.
- **CLI:** `argparse` for `--dataset-root`, `--train-txt`, `--val-txt`, `--test-txt`, batch size, epochs, LR, SMOTE target size, optional noise std, output CSV path, etc. Intended as the **“final”** self-contained CNN training script for the cluster.

### `AST_training.py` — AST (transformer on spectrograms)

- Treats each sample as a **2D spectrogram** (time × frequency), with **global mean/std** normalization (constants in the file should match your dataset statistics if you change splits).
- **`RFSpectrogramDataset`:** Load `.npy`, transpose to time-major, pad/trim to **`TARGET_TIME_LEN`** (e.g. 240) and **`FREQ_BINS`** (512).
- **`RFDataModule`:** Builds train/val from `train.txt` with a **random val split**, and test from `test.txt` if present.
- **`ASTClassifier`:** HuggingFace **`ASTForAudioClassification`** from **`MIT/ast-finetuned-audioset-...`** with `ASTConfig` aligned to mel bins / max length / patch size / heads; **focal loss** with optional class weights; Lightning training with checkpointing and test-time prediction collection.
- Dependencies include **`transformers`**, **`torchmetrics`**, Lightning, and **`sklearn`** for reports. Adjust **`ROOT_DIR`**, `TRAIN_TXT`, `TEST_TXT`, and **`GLOBAL_MEAN` / `GLOBAL_STD`** for your environment.

### `CNN0to8.py` — multi-run CNN experiments (0–8 classes, highway-style)

- **`CNNClassifier`:** Another Lightning CNN path, with **DB-scaled** augmentation options (`MIN_DB` / `MAX_DB`) and torchvision-style transforms where used.
- **`ExperimentConfig` / `DEFAULT_EXPERIMENTS`:** Named presets (**shallow/deep/deeper**, with/without augmentation) for batching different training runs (aligned with ideas from the boosted / hierarchical work under `ebroyles`).
- Reads split files with **label ranges** (`min_label` / `max_label`) so you can target subsets of the 9 classes. Default output roots reference **`ine/sbatch`** and optionally **`ebroyles/BoostedHierarchialModel`** on the cluster—change for local use.

---

## Earlier prototypes and experiments

### `smooth.py`

- **`RFDataset`** loads `.npy`, **per-sample** normalize, applies **`scipy.ndimage.gaussian_filter`** (Gaussian smoothing on the 2D slice), adds a channel dim **`(1, 512, 243)`**.
- **`RFLitModel`:** CNN classifier trained with Lightning, **WandB** logging, early stopping—used to study whether **spectral smoothing** helped stability or accuracy before settling on other pipelines.

### `yo-gurt-v2.py`, `model-gurt.py`, `gurt-model.py`

- Iterations of the same general idea: **`RFDataset`** (load + normalize + channel dim), **`RFDataModule`**, **`RFLitModel`** (small conv stack + classifier), **WandB** + Lightning.
- **`yo-gurt-v2`** is the closest sibling to the smoothing experiment line; **`model-gurt`** / **`gurt-model`** are naming variants as the prototype evolved. Prefer **`final_model.py`** or **`AST_training.py`** for current documentation unless you are reproducing an old run.

### `timm-test.py`

- **Environment smoke test:** prints Python/torch/`timm` versions, CUDA availability, and runs **`nvidia-smi`**. Use on a new node or venv to confirm GPU and **`timm`** before ViT or other timm-backed runs (ViT itself lives mainly under other workspaces, e.g. `wurex`).

### `bean-model.ipynb`

- Exploratory **notebook** for model ideas (“bean” naming); open in Jupyter/VS Code to inspect cells. Not a batch CLI entry point.

### `myprogram.py`

- Minimal **command-line demo** (sum of integers)—not part of the RF pipeline; safe to ignore for ML reproduction.

---

## Quick dependency sketch

| Area | Typical packages |
|------|-------------------|
| CNN (`final_model.py`, `CNN0to8.py`) | `torch`, Lightning, `numpy`, `pandas`, `sklearn`; optional `imblearn` for SMOTE |
| AST (`AST_training.py`) | `torch`, Lightning, `transformers`, `torchmetrics`, `sklearn` |
| Gurt / smooth prototypes | `torch`, Lightning, `wandb`, `numpy`, `scipy` (for `smooth.py`) |

---

## Related docs

- Repo-wide model families: **`../model-arch.md`**
- Root closure / overview: **`../../README.md`**
