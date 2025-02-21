import os
import requests
from pathlib import Path
from tqdm import tqdm

MODEL_URL = "https://huggingface.co/TheBloke/dolphin-2.0-mistral-7B-GGUF/resolve/main/dolphin-2.0-mistral-7b.Q5_K_S.gguf"
MODEL_PATH = Path("models/dolphin-2.0-mistral-7b.Q5_K_S.gguf")


def download_model():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading Q5_K_M model for optimal GPU performance...")

    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
        with open(MODEL_PATH, "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
