import pandas as pd
import numpy as np
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from src.training import MRIDataset, evaluate_model
import torch.nn as nn
from PIL import Image

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
df = pd.read_csv("data/processed/metadata_catalog.csv")

# 1. K-Means misclassifications
df_labeled = df[df["true_label"] != -1].copy()
misclassified_kmeans = df_labeled[df_labeled["true_label"] != df_labeled["weak_label"]]
kmeans_error_files = set(misclassified_kmeans["image_name"].tolist())
print(f"K-Means misclassified {len(kmeans_error_files)} images.")

# To find what CNN misclassifies, we would need to run the test set through the CNN.
# However, the CNN in notebook 2 uses a holdout set. Let's see what the holdout set is.
# The holdout set is defined in Notebook 2 by train_test_split.
# Wait, Notebook 2 splits the data randomly in the cell. So we don't have a fixed test set?
