import torch
import open_clip
import json
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE).eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Load prompts
with open(Path(__file__).parent / "prompts.json", "r") as f:
    data = json.load(f)

CAPTIONS = [
    t.format(c)
    for c in data["concepts"]
    for t in data["templates"]
]

with torch.no_grad():
    tokens = tokenizer(CAPTIONS).to(DEVICE)
    text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
