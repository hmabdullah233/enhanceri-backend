import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from utils.image_utils import full_enhance

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
