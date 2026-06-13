from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)

def detect_tumor(img_path):

    results = model(img_path)

    boxes = results[0].boxes

    if len(boxes) == 0:
        return None

    box = boxes[0]

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    confidence = float(box.conf[0])

    return {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "confidence": round(confidence * 100, 2)
    }