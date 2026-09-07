import os
import argparse
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

def parse_args():
    parser = argparse.ArgumentParser(description="Predict disease class for a single banana leaf image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--model_name", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"], help="Model architecture used during training.")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Path to the trained model weights file.")
    parser.add_argument("--mapping_path", type=str, default="models/class_mapping.json", help="Path to the class mapping JSON file.")
    parser.add_argument("--output_path", type=str, default="prediction_result.png", help="Path to save the output labeled image.")
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
    # Convert keys to integer indices
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
    # Load model weights (map to CPU first, then transfer to device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    # Same transforms as validation phase
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return image, tensor

def predict():
    args = parse_args()
    device = get_device()
    
    # 1. Load class mapping
    class_mapping = load_class_mapping(args.mapping_path)
    num_classes = len(class_mapping)
    
    # 2. Load model
    model = initialize_and_load_model(args.model_name, num_classes, args.model_path, device)
    
    # 3. Preprocess image
    pil_image, input_tensor = preprocess_image(args.image_path)
    input_tensor = input_tensor.to(device)
    
    # 4. Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get highest probability class
    confidence, predicted_idx = torch.max(probabilities, dim=0)
    confidence_pct = confidence.item() * 100
    predicted_class = class_mapping[predicted_idx.item()]
    
    # Print results to terminal
    print("\n" + "="*30)
    print(f"Prediction Result:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print("="*30)
    
    # 5. Save visualized output image
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.title(f"Prediction: {predicted_class} ({confidence_pct:.1f}%)", color='green', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches='tight')
    print(f"Saved prediction visualization to {args.output_path}")

if __name__ == '__main__':
    predict()
