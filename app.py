from fastapi import FastAPI
from ultralytics import YOLO

from services.download_models import download_models

app = FastAPI()

download_models()

model = YOLO("models/best.pt")

@app.get("/")
def home():
    return {"message": "OncoVision API Running"}

@app.get("/test-model")
def test_model():
    return {"status": "YOLO Loaded Successfully"}