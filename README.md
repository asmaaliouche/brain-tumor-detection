# BrainScanAI - Brain Tumor Detection with Semi-Supervised Learning

BrainScanAI is a Research & Development initiative focused on automating the detection of brain tumors from Magnetic Resonance Imaging (MRI) scans. 

The dataset comprises a small subset of "strongly labeled" MRI scans (expert-annotated as normal or cancerous) and a larger set of "unlabeled" MRI scans. This repository implements an end-to-end computer vision pipeline combining feature extraction, unsupervised clustering, weak labeling, and semi-supervised deep learning.

---

## 🎯 Project Objectives

1. **Exploration & Preprocessing**: Perform systematic visual and technical verification of MRI scans (resolution, aspect ratio, color channel, and quality consistency).
2. **Feature Extraction**: Leverage pre-trained Deep Learning models (such as ResNet) to extract high-dimensional visual embeddings.
3. **Weak Labeling (Clustering)**: Project high-dimensional features (via PCA/t-SNE) and apply clustering algorithms (K-Means, DBSCAN) to group unlabeled scans. Clustering performance is validated against the labeled subset using the Adjusted Rand Index (ARI).
4. **Semi-Supervised Deep Learning**:
   - Train a Convolutional Neural Network (CNN) on weakly labeled data.
   - Refine and fine-tune the network on the strongly labeled dataset.
   - Benchmark performance against a baseline model trained strictly under supervised conditions on the labeled subset.
5. **Scalability Analysis**: Provide technical recommendations and cost projections for scaling the pipeline to 4 million images under a specified budget.

---

## 📂 Repository Structure

The codebase is organized modularly to ensure clean separation of concerns, testability, and reproducibility:

```text
6. Brain tumor detection/
├── data/
│   ├── raw/               # Location for raw brain scan images and metadata
│   └── processed/         # Location for extracted embeddings and weak labels
├── notebooks/
│   ├── 01_unsupervised_analysis.ipynb   # Exploration, feature extraction, and clustering
│   └── 02_semi_supervised_approach.ipynb # Supervised and semi-supervised model training
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # PyTorch Dataset definitions and image inspection tools
│   ├── features.py        # Preprocessing and feature extraction pipelines
│   └── modeling.py        # CNN architectures and training/evaluation loops
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   # Automated unit tests for data loading and preprocessing
├── pyproject.toml         # Poetry dependency configuration
└── README.md              # Project documentation
```

---

## ⚙️ Environment Setup

This project uses **Poetry** for deterministic dependency management. To set up the virtual environment:

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

---

## 🚀 Getting Started

1. **Dataset Placement**: Extract the raw MRI dataset into the `data/raw/` directory.
2. **Analysis**: Execute `notebooks/01_unsupervised_analysis.ipynb` to perform visual exploration, run feature extraction, and perform clustering.
3. **Model Training**: Execute `notebooks/02_semi_supervised_approach.ipynb` to train and evaluate the supervised and semi-supervised CNN models.
