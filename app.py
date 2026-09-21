import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import socket
import gdown
FILE_ID = "10Jlf6WZmaPGRD0djlakhFnnplwOiQ9Iz"
MODEL_PATH = "best.pth"

if not os.path.exists(MODEL_PATH):
    print("กำลังดาวน์โหลดโมเดล ...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
# Initialize FastAPI application
app = FastAPI(
    title="Banana Leaf Disease Classifier API",
    description="API to predict banana leaf diseases from images using a fine-tuned EfficientNet-B0 model.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (essential for local HTML file testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = "best_model.pth"
MAPPING_PATH = "class_mapping.json"
MODEL_NAME = "efficientnet"  # Set to "mobilenet" if MobileNet was trained instead

# Global model and mapping storage
model = None
class_mapping = None
device = None

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@app.on_event("startup")
def load_resources():
    global model, class_mapping, device
    
    # 1. Select device
    device = get_device()
    print(f"Server is starting. Using device: {device}")
    
    # 2. Load class mapping
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Class mapping file not found at {MAPPING_PATH}. Please run training first.")
        
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    # Convert keys to integer indices
    class_mapping = {int(k): v for k, v in mapping.items()}
    num_classes = len(class_mapping)
    print(f"Loaded class mapping with {num_classes} classes.")
    
    # 3. Initialize model architecture and load weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model weights not found at {MODEL_PATH}. Please run training first.")
        
    print(f"Initializing {MODEL_NAME} architecture...")
    if MODEL_NAME == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif MODEL_NAME == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")
        
    print(f"Loading weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("🍌 BananaGuard AI is running and ready for Mobile & Desktop! 🍌")
    print(f"To open on this computer: http://localhost:8000")
    if local_ip != "127.0.0.1":
        print(f"To open on your MOBILE PHONE: http://{local_ip}:8000")
        print("Make sure your phone and computer are on the same Wi-Fi network!")
    print("="*60 + "\n")

# Same transform as validation phase
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=FileResponse)
def read_root():
    # Serve index.html directly at the root URL
    return FileResponse("index.html")

@app.get("/info")
def read_info():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "classes": list(class_mapping.values()) if class_mapping else []
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    global model, class_mapping, device
    
    if model is None or class_mapping is None:
        raise HTTPException(status_code=503, detail="Model resources are not loaded yet.")
        
    # Read uploaded file content
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image format.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded image. Error: {str(e)}")
        
    # Preprocess image
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preprocess image. Error: {str(e)}")
        
    # Run inference
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        confidence_pct = float(confidence.item()) * 100
        predicted_class = class_mapping[predicted_idx.item()]
        
        # Build list of class probabilities for details
        probabilities_dict = {}
        for idx, prob in enumerate(probabilities):
            cls_name = class_mapping[idx]
            probabilities_dict[cls_name] = float(prob.item()) * 100
            
        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": confidence_pct,
            "probabilities": probabilities_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run model inference. Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
