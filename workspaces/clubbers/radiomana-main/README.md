# Radio Mana

*radiomana* is an open-source PyTorch library developed by The Aerospace Corporation for manipulating radio signals.

## Installation

### Python Package

```bash
pip install --editable .
```
### Datasets

1. Download the [Spectrum Highway Dataset 2](https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2)
2. Set the `DSET_FIOT_HIGHWAY2` environment variable in your `.bashrc` file to point to the location of the dataset.

```bash
# example
export DSET_FIOT_HIGHWAY2=/tmp/highway2
```

## Usage Example

### Loading the Highway2 Dataset

```python
import radiomana

# inspect a single item
dset = radiomana.Highway2Dataset()
some_psd, some_label = dset[0]

# inspect a whole batch from loader
dmodule = radiomana.HighwayDataModule()
dmodule.setup()
some_batch = next(iter(dmodule.train_datamodule()))
```

### Training Example

```bash
./examples/train_baseline.py
```

### Highway2 Model Performance

With provided basic models and augmentations, we achieve the following performance.
Observe that when un-augmented we overfit rapidly to the training set and our model doesn't generalize well.

| Model                | Submodel           | Augmentations | # Params (M) | Memory (Mb) |  GFlops | Test Loss | F1    | Acc% |
|----------------------|--------------------|---------------|--------------|-------------|---------|-----------|-------|------|
| HighwayBaselineModel | resnet18           | None          |         11.7 |          46 |   1.81  | 0.535     | 0.634 | 79.2 |
| HighwayBaselineModel | resnet18           | VerticalFlip  |         11.7 |          46 |   1.81  | 0.506     | 0.694 | 80.1 |
| HighwayBaselineModel | resnet18           | Noise @ -90dB |         11.7 |          46 |   1.81  | 0.521     | 0.633 | 79.6 |
| HighwayBaselineModel | resnet18           | VFlip & Noise |         11.7 |          46 |   1.81  | 0.507     | 0.662 | 80.2 |
| HighwayBaselineModel | mobilenet_v3_large | None          |          5.5 |          12 |   0.22  | 0.617     | 0.582 | 75.6 |
| HighwayBaselineModel | mobilenet_v3_large | VerticalFlip  |          5.5 |          12 |   0.22  | 0.578     | 0.625 | 77.2 |
| HighwayBaselineModel | mobilenet_v3_large | Noise @ -90dB |          5.5 |          12 |   0.22  | 0.593     | 0.607 | 78.4 |
| HighwayBaselineModel | mobilenet_v3_large | VFlip & Noise |          5.5 |          12 |   0.22  | 0.527     | 0.611 | 79.5 |

## Open Source

### Release

This project is approved for public release with unlimited distribution by Aerospace under OSS Project Ref #OSS25-0006.

### Contributing

Do you have code you would like to contribute to this Aerospace project?

We are excited to work with you. We are able to accept small changes
immediately and require a Contributor License Agreement (CLA) for larger
changesets. Generally documentation and other minor changes less than 10 lines
do not require a CLA. The Aerospace Corporation CLA is based on the well-known
[Harmony Agreements CLA](http://harmonyagreements.org/) created by Canonical,
and protects the rights of The Aerospace Corporation, our customers, and you as
the contributor. [You can find our CLA here](https://aerospace.org/sites/default/files/2020-12/Aerospace-CLA-2020final.pdf).

Please complete the CLA and send us the executed copy. Once a CLA is on file we
can accept pull requests on GitHub or GitLab. If you have any questions, please
e-mail us at [oss@aero.org](mailto:oss@aero.org).

### Licensing

The Aerospace Corporation supports Free & Open Source Software and we publish
our work with GPL-compatible licenses. If the license attached to the project
is not suitable for your needs, our projects are also available under an
alternative license. An alternative license can allow you to create proprietary
applications around Aerospace products without being required to meet the
obligations of the GPL. To inquire about an alternative license, please get in
touch with us at [oss@aero.org](mailto:oss@aero.org).
