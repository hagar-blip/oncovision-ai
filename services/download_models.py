from huggingface_hub import hf_hub_download
import os

REPO_ID = "Hagar500/oncovision-models"

MODELS = [
    "best.pt",
    "best_breast_classifier.pt",
    "quality_gate_best.h5",
    "unet_breast_model.pth"
]

def download_models():
    os.makedirs("models", exist_ok=True)

    token = os.getenv("HF_TOKEN")

    for model_name in MODELS:
        local_path = os.path.join("models", model_name)

        if not os.path.exists(local_path):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=model_name,
                local_dir="models",
                token=token
            )