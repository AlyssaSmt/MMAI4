import io
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from clip_caption import caption_from_base64


# =========================
# Parameter
# =========================
IMG_SIZE = 64

# =========================
# App
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Modell laden
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "quickdraw_cnn.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    index_to_class = json.load(f)

# =========================
# Preprocessing
# =========================
def smart_crop_and_resize(img, out_size, pad=12, threshold=245):
    arr = np.array(img)
    ys, xs = np.where(arr < threshold)

    if len(xs) == 0:
        return Image.new("L", (out_size, out_size), 255)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1], x1 + pad)
    y1 = min(arr.shape[0], y1 + pad)

    cropped = arr[y0:y1, x0:x1]
    h, w = cropped.shape
    side = max(h, w)

    square = np.full((side, side), 255, dtype=np.uint8)
    square[(side-h)//2:(side-h)//2+h,
           (side-w)//2:(side-w)//2+w] = cropped

    return Image.fromarray(square).resize(
        (out_size, out_size),
        Image.Resampling.BILINEAR
    )

def preprocess_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    img = Image.open(
        io.BytesIO(base64.b64decode(base64_string))
    ).convert("L")

    img = smart_crop_and_resize(img, IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0

    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# =========================
# API
# =========================
@app.post("/predict")
async def predict(image_base64: str = Form(...)):
    x = preprocess_image(image_base64)
    preds = model.predict(x, verbose=0)[0]

    top_indices = preds.argsort()[-3:][::-1]

    def clean_label(label: str) -> str:
        return label.replace("full_simplified_", "")

    results = [
        {
            "label": clean_label(index_to_class[str(i)]),
            "confidence": round(float(preds[i]), 3)
        }
        for i in top_indices
    ]


    return {
        "prediction": results[0]["label"],
        "confidence": results[0]["confidence"],
        "top": results
    }

from fastapi import Form

@app.post("/caption")
async def caption(image_base64: str = Form(...)):
    top = caption_from_base64(image_base64, top_k=3)
    best = top[0]
    return {
        "caption": best["caption"],
        "confidence": round(best["confidence"], 3),
        "top": [
            {"caption": r["caption"], "confidence": round(r["confidence"], 3)}
            for r in top
        ],
    }
