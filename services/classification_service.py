import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms
from PIL import Image

MODEL_PATH = "models/best_breast_classifier.pt"

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = models.resnet18(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict_classification(img_path):

    img = Image.open(img_path).convert("RGB")

    x = transform(img)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, dim=1)

    return {
        "prediction": CLASS_NAMES[int(pred)],
        "confidence": round(float(conf) * 100, 2)
    }