import os
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained banana leaf model on the entire validation dataset.")
    parser.add_argument("--data_dir", type=str, default="Dataset", help="Path to the dataset directory containing val folder.")
    parser.add_argument("--model_name", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"], help="Model architecture used during training.")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Path to the trained model weights file.")
    parser.add_argument("--mapping_path", type=str, default="models/class_mapping.json", help="Path to the class mapping JSON file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save evaluation results.")
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_class_mapping(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at {mapping_path}. Run training first.")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}

def initialize_and_load_model(model_name, num_classes, model_path, device):
    print(f"Loading {model_name} model architecture...")
    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained weights not found at {model_path}. Run training first.")
        
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load class mapping and setup classes list
    class_mapping = load_class_mapping(args.mapping_path)
    num_classes = len(class_mapping)
    # Sort class names by their label index (0, 1, 2...)
    class_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
    
    # 2. Setup Validation Transform and Data Loader
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dir = os.path.join(args.data_dir, 'val')
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")
        
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Loaded validation set: {len(val_dataset)} images across {len(class_names)} classes.")
    
    # 3. Initialize and Load model
    model = initialize_and_load_model(args.model_name, num_classes, args.model_path, device)
    
    # 4. Run model evaluation
    all_preds = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 5. Compute and print metrics
    acc = np.mean(all_preds == all_targets) * 100
    print("\n" + "="*50)
    print(f"Overall Validation Accuracy: {acc:.2f}%")
    print("="*50)
    
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save text report to file
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Validation Accuracy: {acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")
    
    # 6. Compute and save confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Greens, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix of Banana Leaf Disease Classification")
    plt.tight_layout()
    
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == '__main__':
    main()
