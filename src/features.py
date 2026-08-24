import os
import ssl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Temporary bypass for SSL certificate verification issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Define Image Transformations for ResNet (ImageNet standard resolution 224x224)
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Custom Dataset to load images from paths
class MRIPathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, path, 0
        except Exception as e:
            # Return dummy tensor in case of load failure
            logger.error(f"Failed to load image at {path}: {e}")
            return torch.zeros(3, 224, 224), path, 1


def extract_resnet_features(all_image_paths, batch_size=32, num_workers=0):
    """
    Extracts dense image embeddings using a pre-trained ResNet-18 model.
    """
    logger.info(f"Total images provided for feature extraction: {len(all_image_paths)}")
    
    # Set up DataLoader (num_workers=0 is safer across OS platforms)
    mri_dataset = MRIPathDataset(all_image_paths, transform=resnet_transform)
    mri_dataloader = DataLoader(mri_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load Pre-trained ResNet-18 and strip classification head
    logger.info("Loading pre-trained ResNet-18 model...")
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet_model.fc = nn.Identity()  # Strip fully connected layer

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    # Run Feature Extraction loop
    features_list = []
    logger.info("Extracting dense image embeddings...")

    with torch.no_grad():
        for imgs, paths, errors in mri_dataloader:
            imgs = imgs.to(device)
            embeddings = resnet_model(imgs)
            features_list.append(embeddings.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    logger.info(f"Extracted feature matrix shape: {features.shape}")
    
    return features


def generate_metadata_catalog(all_image_paths):
    """
    Creates a metadata catalog tracking paths, filenames, and true labels.
    """
    image_names = [os.path.basename(p) for p in all_image_paths]
    true_labels = []
    for p in all_image_paths:
        if "avec_labels/cancer" in p:
            true_labels.append(1)  # 1 for Cancer
        elif "avec_labels/normal" in p:
            true_labels.append(0)  # 0 for Normal
        else:
            true_labels.append(-1) # -1 for Unlabeled

    metadata_df = pd.DataFrame({
        "image_name": image_names,
        "path": all_image_paths,
        "true_label": true_labels
    })
    return metadata_df

