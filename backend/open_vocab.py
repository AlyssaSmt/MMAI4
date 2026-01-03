import io, base64, torch
from PIL import Image
from openclip_model import model, preprocess, text_features, CAPTIONS

@torch.no_grad()
def recognize(image_base64, top_k=5):
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(next(model.parameters()).device)

    image_features = model.encode_image(img)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    sims = (image_features @ text_features.T).squeeze(0)
    probs = torch.softmax(sims * 100, dim=0)

    top = torch.topk(probs, top_k)
    return [
        {
            "label": CAPTIONS[i],
            "confidence": float(p)
        }
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
