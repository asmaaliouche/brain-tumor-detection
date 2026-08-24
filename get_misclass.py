import pandas as pd
import numpy as np

# Load metadata
df = pd.read_csv("data/processed/metadata_catalog.csv")
df_labeled = df[df["true_label"] != -1].copy()
misclassified_kmeans = df_labeled[df_labeled["true_label"] != df_labeled["weak_label"]]
print(f"K-Means misclassified {len(misclassified_kmeans)} out of {len(df_labeled)} labeled images.")
print(misclassified_kmeans[['image_name', 'true_label', 'weak_label']])

