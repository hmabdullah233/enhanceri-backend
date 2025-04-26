import os
import time
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from utils.image_utils import full_enhance

# --- Weight Downloading Setup ---
def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {os.path.basename(destination)} successfully.")
    else:
        print(f"{os.path.basename(destination)} already exists. Skipping download.")

# CodeFormer model weight
download_file(
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "models/codeformer/weights/codeformer/codeformer.pth"
)

# FaceLib ArcFace model weight
download_file(
    "https://huggingface.co/henry/arcface-resnet50/resolve/main/arcface_resnet50.pth",
    "models/codeformer/weights/facelib/arcface.pth"
)
# --- Weight Downloading Setup Ends ---

app = FastAPI()

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    input_path = "input.jpg"
    output_path = "output.jpg"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    start_time = time.time()
    full_enhance(input_path, output_path)
    end_time = time.time()

    print(f"Total Enhancement Time: {end_time - start_time:.2f} sec")
    return FileResponse(output_path, media_type="image/jpeg")
