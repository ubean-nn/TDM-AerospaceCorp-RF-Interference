## RF Interference Classification — Hierarchical CNN

This project trains a hierarchical CNN to classify RF interference signals from spectrogram data into 9 fine-grained classes (0–8), grouped into 3 macro categories: low (0–2), mid (3–6), and high (7–8). The model uses a shared backbone encoder that feeds into two output heads simultaneously, a coarse head that learns the macro groups first, and a fine head that uses the coarse prediction as a hint to discriminate between all 9 classes. Training follows a scheduled loss weighting strategy where the coarse head dominates early epochs and the fine head takes over gradually, forcing the backbone to build general RF features before specializing. Per-class downsampling caps each class at 2,000 samples to address class imbalance, and spectrogram augmentation (frequency masking, time masking, Gaussian noise) is applied during training to improve generalization. After main training, structured L1 channel pruning removes the weakest 20% of convolutional filters from the backbone to reduce model size, followed by a short fine-tune phase to recover accuracy. The best pre-pruning checkpoint is saved as checkpoint_best.pt and the best post-pruning checkpoint as checkpoint_pruned_best.pt. Trained on the Spectrum Highway Dataset 2 using Purdue's Anvil A100 GPU cluster, the model achieved 96.3% coarse and 78.1% fine validation accuracy, improving over a 76% ResNet18 baseline.

## Files

pycode.py —> full training, pruning, and inference pipeline
checkpoint_best.pt —> best model weights before pruning
checkpoint_pruned_best.pt —> best model weights after pruning

## Usage
bashpython pycode.py

To run inference on a single spectrogram:
pythonfrom pycode import predict
coarse, fine = predict("checkpoint_best.pt", "path/to/spectrogram.npy")
