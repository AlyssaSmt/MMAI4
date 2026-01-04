from __future__ import annotations
from pathlib import Path
import torch
import open_clip

from .openclip_vocab import load_vocab, build_prompts, DEFAULT_TEMPLATES

class OpenVocabCLIP:
    def __init__(
        self,
        vocab_path: Path,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        templates: list[str] = DEFAULT_TEMPLATES,
        vocab_limit: int | None = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

        self.words = load_vocab(vocab_path, limit=vocab_limit)
        self.templates = templates
        self.prompts = build_prompts(self.words, templates=self.templates)

        # Build mapping prompt_index -> word_index
        self.prompt_to_word = []
        for wi in range(len(self.words)):
            for _ in self.templates:
                self.prompt_to_word.append(wi)

        # Cache text embeddings once
        with torch.no_grad():
            tokens = self.tokenizer(self.prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features  # (num_prompts, dim)

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, top_k: int = 5):
        # image_tensor: (1, 3, H, W) already preprocessed
        image_features = self.model.encode_image(image_tensor.to(self.device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        sims = (image_features @ self.text_features.T).squeeze(0)  # (num_prompts,)

        # Aggregate prompt scores -> per word
        # take max over templates per word (often better than mean)
        word_scores = torch.full((len(self.words),), -1e9, device=self.device)
        for pi, wi in enumerate(self.prompt_to_word):
            word_scores[wi] = torch.maximum(word_scores[wi], sims[pi])

        probs = torch.softmax(word_scores * 100.0, dim=0)
        top_probs, top_idx = torch.topk(probs, k=min(top_k, len(self.words)))

        results = []
        for p, i in zip(top_probs.tolist(), top_idx.tolist()):
            results.append({"label": self.words[i], "confidence": float(p)})

        return results
