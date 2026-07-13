# AI-Generated Image Detection using Deep Learning
### Comparative Analysis of ResNet18, EfficientNet-B0 and ConvNeXt-Tiny on the TIGAS Dataset

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?logo=nvidia)
![RTX5090](https://img.shields.io/badge/GPU-RTX%205090-success)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This repository contains the complete implementation for my MSc dissertation project:

> **AI-Generated Image Detection using Deep Learning: A Comparative Study of Modern Convolutional Neural Networks**

The objective of this research is to investigate the effectiveness of modern convolutional neural network (CNN) architectures in distinguishing real images from AI-generated images.

Three state-of-the-art CNN architectures were evaluated under identical experimental conditions:

- ResNet18
- EfficientNet-B0
- ConvNeXt-Tiny

Experiments were conducted using the TIGAS benchmark dataset containing **169,584 images** generated from **19 different AI generators**.

---

# Research Objectives

The objectives of this dissertation are:

- Detect AI-generated images using deep learning.
- Compare multiple CNN architectures.
- Evaluate cross-generator performance.
- Investigate the effect of image resolution.
- Provide a reproducible evaluation pipeline.

---

# Dataset

## TIGAS Dataset

Total Images

- **169,584**

Generators

- ADM
- BigGAN
- DALLE2
- GauGAN
- Glide
- Midjourney
- Stable Diffusion
- SDXL
- VQDM
- Wukong
- Stargan
- Face
- Art001
- Art002 (4 variants)

Dataset Split

| Split | Images |
|-------|--------:|
| Train | 141,290 |
| Validation | 14,167 |
| Test | 14,127 |

Classes

- Real Images
- AI-generated Images

---

# Dataset Analysis

The project includes a complete exploratory data analysis (EDA), including:

- Image resolution analysis
- Width and height distribution
- Megapixel statistics
- Generator distribution
- Class distribution
- Image format statistics
- High-resolution image analysis
- Dataset summary reports

---

# Deep Learning Models

The following CNN architectures were evaluated.

## ResNet18

A lightweight residual convolutional neural network serving as the baseline architecture.

## EfficientNet-B0

A compound-scaled CNN designed for improved accuracy and computational efficiency.

## ConvNeXt-Tiny

A modern convolutional architecture inspired by Vision Transformers while maintaining convolution-based operations.

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 384 × 384 |
| Batch Size | 64 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | Cross Entropy |
| Mixed Precision | Enabled |
| CUDA | Enabled |
| GPU | NVIDIA RTX 5090 |

---

# Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- PR-AUC

The project also generates:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Generator-wise Accuracy
- Generator-wise Fake Recall
- Training History
- Validation Curves

---

# Official Experimental Results

| Model | Accuracy | Macro F1 | ROC-AUC | Training Time |
|------|---------:|---------:|---------:|--------------:|
| ResNet18 | **97.40%** | **0.9740** | **0.9966** | **145.22 min** |
| EfficientNet-B0 | **98.41%** | **0.9841** | **0.9986** | **392.71 min** |
| ConvNeXt-Tiny | **99.05%** | **0.9905** | **0.9992** | **138.69 min** |

---

# Project Structure

```text
Dissertation-AI-Image-Detection/

├── checkpoints/
├── datasets/
├── figures/
│
│── official/
│     ├── resnet18
│     ├── efficientnet_b0
│     └── convnext_tiny
│
├── results/
│
│── official/
│     ├── resnet18
│     ├── efficientnet_b0
│     └── convnext_tiny
│
├── src/
│
├── notebooks/
│
├── README.md
└── requirements.txt
```

---

# Running Experiments

Train ResNet18

```bash
python -m src.experiment --dataset tigas --model resnet18
```

Train EfficientNet-B0

```bash
python -m src.experiment --dataset tigas --model efficientnet_b0
```

Train ConvNeXt-Tiny

```bash
python -m src.experiment --dataset tigas --model convnext_tiny
```

---

# Hardware

Experiments were conducted on

- NVIDIA RTX 5090
- 64 GB RAM
- AMD Ryzen 9 9900X3D
- Windows 11
- CUDA acceleration
- Mixed Precision (AMP)

---

# Technologies

- Python
- PyTorch
- Torchvision
- CUDA
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

---

# Future Work

Potential future extensions include:

- Vision Transformer (ViT)
- Swin Transformer
- CLIP-based detection
- Explainable AI (Grad-CAM)
- Real-world social media image evaluation

---

# Author

**Kyi Phyu Aung**

MSc Computing Dissertation

University of Greenwich (NCC Education)

---

# Acknowledgements

This work utilizes the TIGAS benchmark dataset and open-source deep learning libraries including PyTorch and Torchvision for AI-generated image detection research.