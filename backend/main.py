import io
import base64
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from backend.openclip_model import OpenVocabCLIP

import random

# =====================================================
# Pfade & Initialisierung
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
VOCAB_PATH = BASE_DIR.parent / "categories.txt"  # <- deine Open-Vocab-Liste

# ⚠️ Tipp: für Stabilität erstmal begrenzen
# später kannst du vocab_limit=None setzen
clip_model = OpenVocabCLIP(
    vocab_path=VOCAB_PATH,
    vocab_limit=500 # Startwert
)

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI(
    title="Open Vocabulary Drawing AI",
    description="Open-Vocabulary Image Recognition using CLIP",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Hilfsfunktionen
# =====================================================

def decode_base64_image(image_base64: str) -> Image.Image:
    """
    Erwartet ein Base64-encoded PNG (optional mit data:image/... Prefix)
    """
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def make_caption(label: str) -> str:
    """
    Einfache Caption aus Label
    """
    return f"a simple sketch of a {label}"


# =====================================================
# Health Check
# =====================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "vocab_size": len(clip_model.words),
        "open_vocabulary": True
    }


# =====================================================
# Prediction (Open Vocabulary)
# =====================================================

@app.post("/predict")
async def predict(
    image_base64: str = Form(...),
    top_k: int = Form(5)
):
    """
    Liefert:
    {
      "prediction": "cat",
      "confidence": 0.12,
      "top": [
        {"label": "cat", "confidence": 0.12},
        {"label": "dog", "confidence": 0.08}
      ]
    }
    """

    img = decode_base64_image(image_base64)
    image_tensor = clip_model.preprocess(img).unsqueeze(0)

    results = clip_model.predict(image_tensor, top_k=int(top_k))

    if not results:
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "top": []
        }

    best = results[0]

    return {
        "prediction": best["label"],
        "confidence": float(best["confidence"]),
        "top": [
            {"label": r["label"], "confidence": float(r["confidence"])}
            for r in results
        ]
    }


# =====================================================
# Caption Endpoint (für dein Frontend)
# =====================================================

@app.post("/caption")
async def caption(
    image_base64: str = Form(...),
    top_k: int = Form(5)
):
    """
    Liefert:
    {
      "caption": "a simple sketch of a cat",
      "confidence": 0.12,
      "top": [
        {"caption": "...", "confidence": 0.12}
      ]
    }
    """

    img = decode_base64_image(image_base64)
    image_tensor = clip_model.preprocess(img).unsqueeze(0)

    results = clip_model.predict(image_tensor, top_k=int(top_k))

    if not results:
        return {
            "caption": "I am not sure what this is",
            "confidence": 0.0,
            "top": []
        }

    best = results[0]

    # 🔍 Unsicherheits-Schwelle (sehr wichtig für Open Vocabulary!)
    if best["confidence"] < 0.04:
        return {
            "caption": "I am not sure what this is",
            "confidence": float(best["confidence"]),
            "top": []
        }

    return {
        "caption": make_caption(best["label"]),
        "confidence": float(best["confidence"]),
        "top": [
            {
                "caption": make_caption(r["label"]),
                "confidence": float(r["confidence"])
            }
            for r in results
        ]
    }


# =====================================================
# Optional: Alias für Debug / Tests
# =====================================================

@app.post("/open_vocab")
async def open_vocab(
    image_base64: str = Form(...),
    top_k: int = Form(5)
):
    pred = await predict(image_base64=image_base64, top_k=top_k)
    pred["vocab_size"] = len(clip_model.words)
    return pred



@app.get("/random_prompt")
def random_prompt():
    return {
        "word": random.choice(clip_model.words)
    }
