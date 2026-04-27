# Model architectures in this repository

This document summarizes the **four main architecture families** the team used for classifying RF interference from PSD (spectrogram-style) data. The ensemble described in the project closure combines many trained checkpoints; these four families are the main conceptual pillars.

> **Input convention:** Most pipelines treat loaded `.npy` arrays as a **2D time–frequency image** (often resized or normalized to a fixed shape such as 224×224 for ViT, or kept near native dimensions such as 512×243 for some conv models). Always check the corresponding script for exact tensor layout.

---

## 1. Convolutional neural network (CNN)

**Role:** Convolutional stacks on PSD/spectrogram tensors (mostly **2D** `Conv2d` paths; occasionally **1D** convs on time-series-style inputs) feeding classification heads. The team spread CNN experiments across several member workspaces—not a single canonical file.

**Where it appears (examples)**

- `workspaces/ebroyles/` — large share of CNN work: e.g. `GPU CNN/CNNClassify4To8.py`, `CombinationModel/` notebooks and helpers, `Model/nn.py` (Lightning wrapper around a conv backbone), and many `CNN*.ipynb` / hierarchical runs.
- `workspaces/tbhat/sbatch/training.py` — `CustomInterferenceCNN`: **Conv1d** blocks, pooling, and dense layers for multi-class output (see that file for tensor shapes).
- `workspaces/vrao/gpu_testing/sbatch/mini_CNN.py` — compact CNN-style classifier used in GPU/sbatch experiments.
- `workspaces/clubbers/datamodel_clubber.py` (and related notebooks in the same folder) — CNN-oriented datamodel and training experiments.
- `workspaces/arangayy/Model.py` (and `Model.ipynb`) — member pipeline with conv-style training logged via Lightning.

**Also in `workspaces/ine/`**

- `workspaces/ine/final_model.py` — `CNNV6` (`Conv2d` + `BatchNorm2d` + pooling, DropPath-style regularization) wrapped as `LightningCNN`: a concrete from-scratch 2D CNN reference. **AST** for this repo is only under `ine/` as well, but in `AST_training.py` (Section 4), not this file.

---

## 2. 2D ResNet-style network (`VisionSignalNet`)

**Role:** **Deeper, residual** feature extraction on 2D PSD data—strong inductive bias for local patterns while allowing deeper stacks than a tiny custom CNN.

**Primary implementation**

- `workspaces/tbhat/sbatch/weeper.py`  
  - `ResNetBlock2D` — standard two-layer residual blocks with a shortcut.  
  - `VisionSignalNet` — stem (`Conv2d` + `BatchNorm` + `GELU` + pooling), then **three** residual stages (documented in-code as *ResNet18-style* with two blocks per stage: 32→64→128→256 channels), global pooling, dropout, and a linear head.

**Also related**

- `workspaces/mohantr/README.md` and early work refer to a **ResNet-18** baseline for exploration; the shared `radiomana` package in `workspaces/clubbers/radiomana-main` defines `HighwayBaselineModel` using **ImageNet-style** backbones (currently wired to **MobileNet V3 Large** in code, with a commented `resnet18` option). Treat that as a *baseline / alternate backbone* path, distinct from the custom `VisionSignalNet` above.

---

## 3. Vision Transformer (ViT)

**Role:** **Patch-based** image model (transformer encoder) on the PSD, often with **ImageNet pretraining** via `timm`, after repeating or projecting to 3 channels to match pretrained expectations.

**Primary implementation**

- `workspaces/wurex/newmodelAttempt.py`  
  - Class: `LitViT` (PyTorch Lightning).  
  - Uses `timm.create_model(..., pretrained=True, num_classes=...)`; examples in the file include `vit_base_patch16_224` and `vit_base_patch8_224` (and similar variants can be tried by changing the model name string).

**Data flow (typical in this file)**

- Load PSD → resize/normalize to **224×224** → **repeat to 3 channels** for compatible ViT inputs → train with Lightning.

---

## 4. Audio Spectrogram Transformer (AST)

**Role:** A **transformer** stack designed for **log-mel / spectrogram** inputs, adapted here to RF spectrograms. Uses **patch embedding + self-attention** (HuggingFace `AST*` implementation) rather than convolutions in the main trunk.

**Primary implementation**

- `workspaces/ine/AST_training.py`  
  - `transformers.ASTForAudioClassification` with `ASTConfig` (e.g. patch size, hidden size, attention heads, trimmed time length).  
  - Expects 2D spectrogram tensors shaped for AST-style processing (the script resizes/trim to a fixed time length and `FREQ_BINS` and applies global mean/std normalization).

**Relation to “two transformers”**

- **ViT** (Section 3) and **AST** (Section 4) are the two main **transformer** lineages in this repo: ViT operates on a visual patch grid with `timm`/vision pretraining; AST uses the **audio** AST head and config tailored to spectrogram-like inputs.

---

## Quick reference

| Family   | Paradigm        | Example entry point |
|----------|-----------------|----------------------|
| CNN      | 2D (or 1D) convs + head | See §1 — e.g. `workspaces/ebroyles/`, `workspaces/tbhat/sbatch/training.py`, `workspaces/ine/final_model.py` |
| ResNet   | 2D residual CNN | `workspaces/tbhat/sbatch/weeper.py` (`VisionSignalNet`) |
| ViT      | Patch ViT (timm)| `workspaces/wurex/newmodelAttempt.py` (`LitViT`) |
| AST      | AST (transformers) | `workspaces/ine/AST_training.py` |

For training orchestration, logging (e.g. WandB), and the shared dataset API, see also the **radiomana**-based scripts under `workspaces/clubbers/`, `workspaces/tbhat/`, and `workspaces/mohantr/`.
