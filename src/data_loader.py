"""
BrainScanAI - Brain MRI Data Loader and Inspector Module

This module provides the core data loading and programmatic analysis components for the
BrainScanAI pipeline. It implements custom PyTorch Dataset configurations
and helper functions to analyze dataset physical characteristics under strict production
and MLOps standards.

Classes:
    - BrainMRIDataset: Custom PyTorch Dataset that loads MRI brain scans from specified folders.
      Handles both labeled (supervised/fine-tuning) and unlabeled (clustering/weak-labeling) modes.
      Ensures automatic conversion to 3-channel RGB to maintain compatibility with pre-trained models.

Functions:
    - inspect_image_properties: Programmatically extracts metadata from all JPEG/PNG scans
      in a specified directory, logging and organizing image width, height, aspect ratio,
      color mode, file format, and file size (KB).

Implementation Details:
    - Built using PIL (Pillow), pandas, and PyTorch.
    - All image conversions are standardized to RGB.
    - Supports custom torchvision transforms for downstream augmentation and preprocessing.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing Brain MRI scan images.
    """

    def __init__(self, data_dir: str, labels_df: pd.DataFrame = None, transform=None):
        """
        Initializes the dataset loader.

        Args:
            data_dir (str): Path to the directory containing MRI images.
            labels_df (pd.DataFrame, optional): DataFrame containing image file names and labels.
                                               If None, the dataset runs in unlabeled/inference mode.
            transform (callable, optional): PyTorch transforms to be applied to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = labels_df

        # If a list of labels is provided, use it; otherwise, load all images in the folder
        if self.labels_df is not None:
            self.image_names = self.labels_df["image_name"].values
            self.labels = self.labels_df["label"].values
        else:
            self.image_names = [
                f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            self.labels = None

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure 3 channels for pre-trained CNNs

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label, img_name

        return image, img_name


def inspect_image_properties(data_dir: str) -> pd.DataFrame:
    """
    Helper function to programmatically analyze the image characteristics in a directory.

    Args:
        data_dir (str): Directory containing the images.

    Returns:
        pd.DataFrame: A DataFrame containing dimensions, color modes, and formats of each image.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    properties = []

    for img_name in image_files:
        path = os.path.join(data_dir, img_name)
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode  # e.g., 'RGB', 'L' (grayscale)
                img_format = img.format
                properties.append({
                    "image_name": img_name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "mode": mode,
                    "format": img_format,
                    "file_size_kb": os.path.getsize(path) / 1024.0
                })
        except Exception as e:
            properties.append({
                "image_name": img_name,
                "error": str(e)
            })

    return pd.DataFrame(properties)
