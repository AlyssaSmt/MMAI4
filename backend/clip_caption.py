import io
import base64
from PIL import Image
import torch
import clip

# 1) Captions, die CLIP auswählen darf
CAPTIONS = [
    "a simple sketch of a cat",
    "a simple sketch of a car",
    "a simple sketch of a cup",
    "a simple sketch of a house",
    "a simple sketch of a sun",
    "a simple sketch of an eye",
    "a simple sketch of a tree",
    "a simple sketch of pants",
    "a simple sketch of a strawberry",
    "a simple sketch of a wine glass",
# 2) Neue Captions
    "a simple sketch of a cow",
    "a simple sketch of a person",
    "a simple sketch of a phone",
    "a simple sketch of a bicycle",
    "a simple sketch of a fish",
    "a simple sketch of a guitar",
    "a simple sketch of a chair",
]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _preprocess = clip.load("ViT-B/32", device=_device)
_text_tokens = clip.tokenize(CAPTIONS).to(_device)

@torch.no_grad()
def caption_from_base64(image_base64: str, top_k: int = 3):
    # base64 cleanup (data:image/png;base64,...)
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_input = _preprocess(img).unsqueeze(0).to(_device)

    image_features = _model.encode_image(image_input)
    text_features = _model.encode_text(_text_tokens)

    # cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (image_features @ text_features.T).squeeze(0)  # (num_captions,)
    probs = torch.softmax(sims * 100.0, dim=0)            # sharpened softmax

    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(CAPTIONS)))
    results = [
        {"caption": CAPTIONS[i], "confidence": float(p)}
        for p, i in zip(top_probs.tolist(), top_idx.tolist())
    ]

    return results
