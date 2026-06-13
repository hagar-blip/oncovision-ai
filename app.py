from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

model = YOLO("models/best.pt")

@app.get("/")
def home():
    return {"message": "OncoVision API Running"}

@app.get("/test-model")
def test_model():
    return {"status": "YOLO Loaded Successfully"}