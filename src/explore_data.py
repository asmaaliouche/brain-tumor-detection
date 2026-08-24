"""
BrainScanAI - Dataset Programmatic Exploration Script

This script executes an automated technical exploration and validation of the raw brain
MRI scan dataset for BrainScanAI. It scans the specified subdirectories,
verifies file counts, formats, resolutions, color modes, and statistics, and catalogs
all image properties into a centralized CSV for downstream analysis and reproduction.

Functions:
    - run_exploration: Scans raw directories ('avec_labels/cancer', 'avec_labels/normal',
      and 'sans_label'), programmatically extracts properties for all 1506 images, validates
      their integrity (checking for corrupted files), logs distributions, and saves the final
      catalog to 'data/processed/image_properties.csv'.

MLOps Standards Implemented:
    - Implemented a unified formatting schema with timestamps, log levels, and module tags.
    - Preserves data immutability by treating 'data/raw/' as read-only and logging/saving metrics
      exclusively under the 'data/processed/' output directory.
"""

import os
import logging
import pandas as pd
from PIL import Image

# Configure logging to match production MLOps standards
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("explore_data")

def run_exploration():
    """
    Executes a comprehensive programmatic exploration of the Brain MRI dataset.
    Verifies image counts, file formats, dimensions, color modes, and file size statistics.
    Logs warnings for missing paths or corrupted files.
    """
    base_raw_dir = "data/raw/mri_dataset_brain_cancer_oc"
    
    subdirs = {
        "labeled_cancer": os.path.join(base_raw_dir, "avec_labels/cancer"),
        "labeled_normal": os.path.join(base_raw_dir, "avec_labels/normal"),
        "unlabeled": os.path.join(base_raw_dir, "sans_label")
    }
    
    logger.info("Starting brain MRI dataset exploration.")
    
    total_files = 0
    all_properties = []
    
    for category, path in subdirs.items():
        if not os.path.exists(path):
            logger.warning("Category path '%s' does not exist.", path)
            continue
            
        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        count = len(files)
        total_files += count
        logger.info("Category: %-16s | Path: %-45s | Image Count: %d", category, path, count)
        
        # Analyze properties for each image in this category
        for f in files:
            full_path = os.path.join(path, f)
            try:
                with Image.open(full_path) as img:
                    w, h = img.size
                    mode = img.mode
                    img_format = img.format
                    size_kb = os.path.getsize(full_path) / 1024.0
                    all_properties.append({
                        "category": category,
                        "file_name": f,
                        "width": w,
                        "height": h,
                        "mode": mode,
                        "format": img_format,
                        "size_kb": size_kb
                    })
            except Exception as e:
                logger.error("Failed to load image %s: %s", full_path, str(e))
                all_properties.append({
                    "category": category,
                    "file_name": f,
                    "error": str(e)
                })

    df = pd.DataFrame(all_properties)
    logger.info("Total Images Scanned: %d", len(df))
    
    if len(df) == 0:
        logger.error("No images found. Dataset structure should be verified.")
        return
        
    # Check for any errors during reading
    if "error" in df.columns:
        errors = df[df["error"].notna()]
        if len(errors) > 0:
            logger.error("Corrupted files identified: %d files could not be loaded.", len(errors))
            return
            
    logger.info("All scanned images successfully parsed with zero corrupted files.")

    # 1. Image Resolutions and Aspect Ratios
    resolutions = df.groupby(["width", "height"]).size().reset_index(name="count")
    for _, row in resolutions.iterrows():
        pct = (row['count'] / len(df)) * 100
        logger.info("Resolution: %dx%d | Count: %d (%.1f%%)", row['width'], row['height'], row['count'], pct)

    # 2. Color Mode Distribution
    modes = df.groupby("mode").size().reset_index(name="count")
    for _, row in modes.iterrows():
        pct = (row['count'] / len(df)) * 100
        mode_desc = "Grayscale (L)" if row['mode'] == 'L' else ("RGB Color" if row['mode'] == 'RGB' else row['mode'])
        logger.info("Color Mode: %-5s (%-13s) | Count: %d (%.1f%%)", row['mode'], mode_desc, row['count'], pct)

    # 3. File Formats
    formats = df.groupby("format").size().reset_index(name="count")
    for _, row in formats.iterrows():
        pct = (row['count'] / len(df)) * 100
        logger.info("File Format: %-5s | Count: %d (%.1f%%)", row['format'], row['count'], pct)

    # 4. File Size Statistics
    desc = df["size_kb"].describe()
    logger.info("File Size Statistics (KB) - Count: %d, Mean: %.2f, Std: %.2f, Min: %.2f, Max: %.2f", 
                desc["count"], desc["mean"], desc["std"], desc["min"], desc["max"])
    
    # Save the scanned properties for downstream analysis
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/image_properties.csv"
    df.to_csv(output_path, index=False)
    logger.info("Technical properties cataloged and saved to: %s", output_path)

if __name__ == "__main__":
    run_exploration()
