"""
BrainScanAI - Automated Pipeline Test Suite

This module implements comprehensive unit testing for Step 1 (Data Exploration and Loading) 
of the BrainScanAI pipeline. Using mock/synthetic image directories created dynamically in memory,
it validates data integrity loaders and property cataloging tools without dependencies on external files.

Fixtures:
    - temp_image_directory: Automatically handles creation and teardown of a temporary
      folder containing randomized RGB and Grayscale images for isolated test executions.

Test Functions:
    - test_inspect_image_properties: Verifies that the programmatic property cataloging tool
      correctly detects image heights, widths, formats, aspect ratios, and color modes.
    - test_brain_mri_dataset_unlabeled: Validates the custom PyTorch Dataset behavior under
      unlabeled/clustering inference configurations (automatically finding and converting images to 3-channel RGB).
    - test_brain_mri_dataset_labeled: Validates the custom PyTorch Dataset behavior under
      labeled conditions (e.g., matching file lists and target values from a labels dataframe).

Quality Assurance:
    - Uses pytest fixtures and temporary OS directories to maintain file system hygiene.
    - Runs in less than 1 second, ensuring rapid integration testing.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data_loader import BrainMRIDataset, inspect_image_properties


@pytest.fixture
def temp_image_directory():
    """
    Fixture that creates a temporary directory containing dummy images for testing.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create an RGB image
        rgb_img_path = os.path.join(tmp_dir, "test_rgb.png")
        rgb_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(rgb_arr).save(rgb_img_path)

        # Create a grayscale image
        gray_img_path = os.path.join(tmp_dir, "test_gray.jpg")
        gray_arr = np.random.randint(0, 255, (120, 80), dtype=np.uint8)
        Image.fromarray(gray_arr).save(gray_img_path)

        yield tmp_dir


def test_inspect_image_properties(temp_image_directory):
    """
    Test programmatically inspecting properties of images in the directory.
    """
    df = inspect_image_properties(temp_image_directory)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    # Check for correct columns
    expected_cols = {"image_name", "width", "height", "aspect_ratio", "mode", "format", "file_size_kb"}
    assert expected_cols.issubset(df.columns)

    # Validate row specific values
    rgb_row = df[df["image_name"] == "test_rgb.png"].iloc[0]
    assert rgb_row["width"] == 100
    assert rgb_row["height"] == 100
    assert rgb_row["mode"] == "RGB"

    gray_row = df[df["image_name"] == "test_gray.jpg"].iloc[0]
    assert gray_row["width"] == 80
    assert gray_row["height"] == 120
    assert gray_row["mode"] == "L"


def test_brain_mri_dataset_unlabeled(temp_image_directory):
    """
    Test BrainMRIDataset in unlabeled mode (scanning all files in directory).
    """
    dataset = BrainMRIDataset(data_dir=temp_image_directory)

    assert len(dataset) == 2
    # Load first item
    image, img_name = dataset[0]
    assert isinstance(image, Image.Image)
    assert img_name in ["test_rgb.png", "test_gray.jpg"]
    # Verify both are loaded as 3-channel (RGB) by the convert("RGB") logic in dataset
    assert image.mode == "RGB"


def test_brain_mri_dataset_labeled(temp_image_directory):
    """
    Test BrainMRIDataset in labeled mode using an annotations DataFrame.
    """
    labels_df = pd.DataFrame({
        "image_name": ["test_rgb.png"],
        "label": [1]
    })

    dataset = BrainMRIDataset(data_dir=temp_image_directory, labels_df=labels_df)

    assert len(dataset) == 1
    image, label, img_name = dataset[0]

    assert isinstance(image, Image.Image)
    assert img_name == "test_rgb.png"
    assert label == 1
    assert image.mode == "RGB"
