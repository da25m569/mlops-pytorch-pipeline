import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.model import create_model


app = FastAPI(title="CIFAR-10 ResNet-18 API")

MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoints/best_model.pth")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    num_classes = checkpoint.get("num_classes", 10)

    model = create_model(
        num_classes=num_classes,
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = Image.open(file.file).convert("RGB")

        image = image.resize((224, 224))

        image_tensor = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32,
        ).reshape(224, 224, 3)

        image_tensor = image_tensor / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        return {
            "class_id": int(predicted.item()),
            "confidence": float(confidence.item()),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
