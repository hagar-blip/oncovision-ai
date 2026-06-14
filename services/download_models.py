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

    for model_name in MODELS:
        local_path = os.path.join("models", model_name)

        if not os.path.exists(local_path):
            print(f"Downloading {model_name}...")

            hf_hub_download(
                repo_id=REPO_ID,
                filename=model_name,
                local_dir="models",
                local_dir_use_symlinks=False
            )

            print(f"{model_name} downloaded.")