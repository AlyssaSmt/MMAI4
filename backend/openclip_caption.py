import io
import base64
from pathlib import Path
from PIL import Image
import torch
import open_clip

# =========================
# Konfiguration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
TOP_K = 5

# =========================
# Vocabulary laden
# =========================
VOCAB_PATH = Path(__file__).parent / "vocab.txt"
VOCAB = [line.strip() for line in VOCAB_PATH.read_text().splitlines() if line.strip()]

# =========================
# Prompt Templates
# =========================
TEMPLATES = [
    "a hand drawn sketch of a {}",
    "a simple doodle of a {}",
    "a black and white line drawing of a {}",
    "a childlike drawing of a {}",
    "a rough sketch of a {}",
]

CAPTIONS = [
    template.format(word)
    for word in VOCAB
    for template in TEMPLATES
]

# =========================
# Modell laden
# =========================
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(DEVICE)
model.eval()

# Texte einmal embeddieren (SEHR WICHTIG für Speed!)
with torch.no_grad():
    text_tokens = tokenizer(CAPTIONS).to(DEVICE)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# =========================
# Hauptfunktion
# =========================
@torch.no_grad()
def caption_from_base64(image_base64: str, top_k: int = TOP_K):
    # Base64 Cleanup
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Cosine Similarity
    similarities = (image_features @ text_features.T).squeeze(0)

    probs = torch.softmax(similarities * 100.0, dim=0)

    top_probs, top_indices = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append({
            "caption": CAPTIONS[idx],
            "confidence": float(prob)
        })

    return {
        "best": results[0],
        "top": results
    }
