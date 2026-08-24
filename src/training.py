import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

class MRIDataset(Dataset):
    """
    Custom Dataset for loading MRI images from file paths.
    """
    def __init__(self, dataframe, label_col="true_label", transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["path"]
        label = row[self.label_col]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
            label = 0
            
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    """
    Standard training loop for PyTorch models.
    """
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

    return train_losses, val_losses

def evaluate_model(model, loader, device):
    """
    Evaluates a trained model on a dataloader and returns true labels and predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds
