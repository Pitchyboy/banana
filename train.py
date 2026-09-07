import os
import argparse
import json
import time
import copy
import matplotlib
# Use Agg backend to prevent GUI-related errors when running on headless servers or sandboxes
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN to classify banana leaf diseases.")
    parser.add_argument("--data_dir", type=str, default="Dataset", help="Path to the dataset directory containing train/val folders.")
    parser.add_argument("--model_name", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"], help="CNN architecture to use.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization.")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save checkpoints.")
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU (Apple Silicon MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_transforms():
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    # Validation transform (only resizing and cropping, no random changes)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def initialize_model(model_name, num_classes):
    print(f"Initializing {model_name} with pre-trained ImageNet weights...")
    if model_name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        
        # Modify the classifier (last layer) for our target number of classes
        # EfficientNet-B0 has a classifier layer that is sequential:
        # classifier[1] is the Linear(in_features=1280, out_features=1000)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "mobilenet":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        
        # MobileNetV3-Large has a classifier layer that is sequential:
        # classifier[3] is the Linear(in_features=1280, out_features=1000)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Store history for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward & optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Deep copy the model weights if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def save_plots(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved training curves to {plot_path}")

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Device configuration
    device = get_device()
    
    # 2. Setup Data Transformers & Load Datasets
    train_transform, val_transform = get_transforms()
    
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing train/val directories under {args.data_dir}. Please verify dataset directory structure.")
        
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Save the mapping of classes to indexes for prediction script
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    mapping_path = os.path.join(args.save_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Class mapping saved to {mapping_path}")
    print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
    
    # 3. Create Data Loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }
    
    # 4. Initialize model
    num_classes = len(train_dataset.classes)
    model = initialize_model(args.model_name, num_classes)
    model = model.to(device)
    
    # 5. Define loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    
    # AdamW works very well for fine-tuning pre-trained CNNs
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler decays learning rate smoothly over training epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 6. Train and evaluate
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    # 7. Save model checkpoints
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    torch.save(model.state_dict(), best_model_path)
    print(f"Saved best model weights to {best_model_path}")
    
    # 8. Plot loss and accuracy curves
    save_plots(history, args.save_dir)

if __name__ == '__main__':
    main()
