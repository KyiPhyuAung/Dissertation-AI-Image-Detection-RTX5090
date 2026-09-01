# AI-Generated Image Detection using Deep Learning

### Multi-Dataset Comparative Evaluation of ResNet18, EfficientNet-B0 and ConvNeXt-Tiny

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green?logo=nvidia)
![RTX5090](https://img.shields.io/badge/GPU-RTX%205090-success)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-blue)

------------------------------------------------------------------------

## Overview

This repository contains the experimental implementation for an MSc
dissertation on the detection of AI-generated images using deep
learning.

The research compares three convolutional neural network (CNN)
architectures:

-   ResNet18
-   EfficientNet-B0
-   ConvNeXt-Tiny

The models are evaluated across three image datasets:

-   TIGAS
-   Real and Fake (AI-Generated) Art Images (`ART_IMAGES`)
-   CIFAKE

The main purpose of the study is to compare model performance across
different image distributions rather than relying on the results of a
single dataset. The experiments use a consistent training and evaluation
pipeline where possible, while preserving dataset-specific test-set
requirements.

This repository contains the RTX 5090 training and evaluation pipeline
used for the controlled experiments. A separate real-world evaluation
application is maintained independently and is not the main focus of
this repository.

------------------------------------------------------------------------

## Research Design

The controlled benchmark contains nine model-dataset experiments:

                      TIGAS   ART_IMAGES   CIFAKE
  ----------------- ------- ------------ --------
  ResNet18                ✓            ✓        ✓
  EfficientNet-B0         ✓            ✓        ✓
  ConvNeXt-Tiny           ✓            ✓        ✓

The experiments are within-dataset evaluations. They provide a broader
multi-dataset comparison, but they should not be interpreted as direct
cross-dataset generalisation experiments.

------------------------------------------------------------------------

## Datasets

### TIGAS

The local TIGAS dataset used in the official experiments contains
**169,584 images**.

  Split               Images
  ------------ -------------
  Train              141,290
  Validation          14,167
  Test                14,127
  **Total**      **169,584**

The dataset contains real and AI-generated images from multiple
generator sources. The existing TIGAS split is preserved for the
official experiments.

### ART_IMAGES

`ART_IMAGES` contains **21,642 images**, balanced between real and
AI-generated artwork.

  Class             Images
  ----------- ------------
  REAL              10,821
  FAKE              10,821
  **Total**     **21,642**

A deterministic class-stratified split with seed 42 is used:

  Split          Images
  ------------ --------
  Train          17,312
  Validation      2,164
  Test            2,166

### CIFAKE

CIFAKE contains **120,000 images**.

The official 20,000-image test set is kept untouched. The official
100,000-image training set is divided into a 90,000-image training set
and a deterministic 10,000-image validation holdout.

  Split               Images
  ------------ -------------
  Train               90,000
  Validation          10,000
  Test                20,000
  **Total**      **120,000**

For the binary folder datasets, labels are assigned explicitly as:

``` text
REAL = 0
FAKE = 1
```

------------------------------------------------------------------------

## Models

### ResNet18

A residual CNN used as the lightweight baseline architecture.

### EfficientNet-B0

A compact CNN based on compound scaling of network depth, width and
input resolution.

### ConvNeXt-Tiny

A modern convolutional architecture that adopts several design ideas
associated with newer vision architectures while retaining
convolution-based operations.

All three models use ImageNet-pretrained weights before being adapted to
binary real-versus-AI image classification.

------------------------------------------------------------------------

## Training Configuration

The controlled experiments use the following main configuration:

  Parameter                                      Value
  ---------------------------------------------- -------------------------
  Input size                                     384 × 384
  Batch size                                     64
  Epochs                                         20
  Optimizer                                      Adam
  Learning rate                                  1e-4
  Loss function                                  Cross Entropy
  Random seed for deterministic dataset splits   42
  Mixed precision (AMP)                          Enabled on CUDA
  TF32                                           Enabled on CUDA
  GPU                                            NVIDIA GeForce RTX 5090

### Image preprocessing

Training images are:

1.  resized to 384 × 384,
2.  randomly horizontally flipped,
3.  randomly rotated by up to ±10 degrees,
4.  converted to tensors, and
5.  normalised using ImageNet mean and standard deviation.

Validation and test images are resized, converted to tensors and
normalised without random augmentation.

### Checkpoint selection

Each official run is trained for 20 epochs. The final test evaluation
does **not** automatically use the epoch-20 model.

The best checkpoint is selected using the **lowest validation loss**
observed during training. The selected checkpoint is then evaluated on
the corresponding test set.

------------------------------------------------------------------------

## Evaluation Metrics

The evaluation pipeline reports:

-   Accuracy
-   Precision
-   Recall
-   Macro F1-score
-   Weighted F1-score
-   Balanced Accuracy
-   Matthews Correlation Coefficient (MCC)
-   ROC-AUC
-   PR-AUC

The pipeline also stores supporting outputs such as classification
reports, confusion matrices, training histories, validation curves,
JSON/CSV summaries and experiment timing information.

TIGAS-specific experiments additionally support generator-level analysis
where applicable.

------------------------------------------------------------------------

## Official Multi-Dataset Results

### Test Accuracy

  Model                    TIGAS   ART_IMAGES       CIFAKE
  ----------------- ------------ ------------ ------------
  ResNet18                97.40%       99.72%       98.12%
  EfficientNet-B0         98.41%   **99.91%**   **98.62%**
  ConvNeXt-Tiny       **99.05%**       99.72%       98.33%

The results show that the highest-performing architecture is not the
same for every dataset. ConvNeXt-Tiny achieved the highest TIGAS
accuracy, while EfficientNet-B0 achieved the highest accuracy on
ART_IMAGES and CIFAKE.

These results are treated as a multi-dataset comparative evaluation.
Differences between datasets may reflect differences in image content,
source distributions and dataset characteristics, so the table should
not be interpreted as evidence of cross-dataset generalisation.

### ART_IMAGES detailed results

  ----------------------------------------------------------------------------------
  Model                 Accuracy     Macro F1          MCC   Best Epoch     Training
                                                                                Time
  ----------------- ------------ ------------ ------------ ------------ ------------
  ResNet18                99.72%       99.72%       0.9945           12    16.45 min

  EfficientNet-B0     **99.91%**   **99.91%**   **0.9982**           16    48.33 min

  ConvNeXt-Tiny           99.72%       99.72%       0.9945           14      **11.05
                                                                               min**
  ----------------------------------------------------------------------------------

### CIFAKE detailed results

  ----------------------------------------------------------------------------------
  Model                 Accuracy     Macro F1          MCC   Best Epoch     Training
                                                                                Time
  ----------------- ------------ ------------ ------------ ------------ ------------
  ResNet18                98.12%       98.12%       0.9625           15    81.02 min

  EfficientNet-B0     **98.62%**   **98.62%**   **0.9724**           10   242.41 min

  ConvNeXt-Tiny           98.33%       98.33%       0.9667            7      **52.76
                                                                               min**
  ----------------------------------------------------------------------------------

### TIGAS detailed results

  -----------------------------------------------------------------------------
  Model                   Accuracy       Macro F1        ROC-AUC  Training Time
  ----------------- -------------- -------------- -------------- --------------
  ResNet18                  97.40%         97.40%         0.9966     145.22 min

  EfficientNet-B0           98.41%         98.41%         0.9986     392.71 min

  ConvNeXt-Tiny         **99.05%**     **99.05%**     **0.9992** **138.69 min**
  -----------------------------------------------------------------------------

The legacy TIGAS timing values above are recorded training times. They
are kept separate from total experiment time because equivalent
machine-readable total experiment timing is not available for those
earlier runs.

------------------------------------------------------------------------

## Comparison Outputs

The multi-dataset comparison script can be run with:

``` powershell
python -m src.plot_official_comparisons
```

It reads the official experiment outputs and creates comparison tables
and figures.

Main outputs:

``` text
results/
└── official_comparisons/
    └── official_model_dataset_comparison.csv

figures/
└── official_comparisons/
    └── ...
```

For ART_IMAGES and CIFAKE, the comparison script reads the
machine-readable official run files. For the earlier TIGAS experiments,
recorded legacy accuracy and training-time values are used when
equivalent machine-readable files are not available.

------------------------------------------------------------------------

## Project Structure

``` text
Dissertation-AI-Image-Detection/

├── checkpoints/
│   ├── official/
│   │   ├── art_images/
│   │   ├── cifake/
│   │   └── ...
│   └── experimental/
│
├── datasets/
│   ├── ART_IMAGES/
│   ├── CIFAKE/
│   └── TIGAS/
│
├── figures/
│   ├── official/
│   │   ├── art_images/
│   │   └── cifake/
│   ├── official_comparisons/
│   └── experimental/
│
├── results/
│   ├── official/
│   │   ├── art_images/
│   │   └── cifake/
│   ├── official_comparisons/
│   └── experimental/
│
├── src/
│   ├── binary_folder_dataset.py
│   ├── config.py
│   ├── eda.py
│   ├── experiment.py
│   ├── plot_official_comparisons.py
│   └── ...
│
├── README.md
└── requirements.txt
```

Large datasets and model checkpoints should normally remain outside
version control unless they are intentionally managed using an
appropriate large-file storage solution.

------------------------------------------------------------------------

## Running the Experiments

Activate the project environment before running experiments.

### TIGAS

``` powershell
python -m src.experiment --model resnet18 --dataset tigas
python -m src.experiment --model efficientnet_b0 --dataset tigas
python -m src.experiment --model convnext_tiny --dataset tigas
```

### ART_IMAGES

``` powershell
python -m src.experiment --model resnet18 --dataset art_images
python -m src.experiment --model efficientnet_b0 --dataset art_images
python -m src.experiment --model convnext_tiny --dataset art_images
```

### CIFAKE

``` powershell
python -m src.experiment --model resnet18 --dataset cifake
python -m src.experiment --model efficientnet_b0 --dataset cifake
python -m src.experiment --model convnext_tiny --dataset cifake
```

Official runs use the configured 20 epochs. Shorter runs should be
treated as experimental or sanity checks and should not be reported as
final dissertation results.

------------------------------------------------------------------------

## Reproducibility Notes

-   ART_IMAGES uses a deterministic class-stratified split with seed 42.
-   CIFAKE keeps the official test set untouched and creates a
    deterministic validation holdout from the official training set.
-   The explicit binary label mapping is `REAL = 0` and `FAKE = 1`.
-   Dataset split reproducibility does not imply bit-for-bit
    deterministic GPU training.
-   Official and experimental outputs are stored separately to reduce
    the risk of a short sanity run overwriting an official checkpoint.
-   Final testing uses the best validation-loss checkpoint.

------------------------------------------------------------------------

## Hardware and Software Environment

Official RTX 5090 experiments were conducted with:

-   Windows 11
-   Python 3.12.10
-   PyTorch 2.12.1+cu130
-   Torchvision 0.27.1+cu130
-   CUDA 13.0
-   NVIDIA GeForce RTX 5090
-   64 GB RAM
-   AMD Ryzen 9 9900X3D
-   Automatic Mixed Precision (AMP)

Main Python libraries include PyTorch, Torchvision, NumPy, Pandas,
Matplotlib, Scikit-learn and Pillow.

------------------------------------------------------------------------

## Research Scope

This repository focuses on controlled model training and evaluation
across TIGAS, ART_IMAGES and CIFAKE.

A separate desktop application is used as a supporting practical
evaluation environment for real-world images. That application deploys
the previously trained TIGAS checkpoints and should not be interpreted
as deploying the ART_IMAGES- or CIFAKE-trained checkpoints.

The benchmark results presented here and the practical application
therefore represent related but distinct parts of the wider dissertation
work.

------------------------------------------------------------------------

## Author

**Ko Kyi Phyu Aung**

MSc Computing Dissertation\
University of Wolverhampton, UK
