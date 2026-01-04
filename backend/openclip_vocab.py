# for loading and processing vocabularies for OpenCLIP
# not used anymore in the current version, but kept for reference

from pathlib import Path

DEFAULT_TEMPLATES = [
    "a simple sketch of a {}",
    "a black and white doodle of a {}",
    "a hand drawn {}",
]

def load_vocab(vocab_path: Path, min_len: int = 2, max_len: int = 30, limit: int | None = None):
    words = []
    for line in vocab_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = line.strip()
        if not w:
            continue
        # basic cleanup
        w = w.replace("_", " ").strip()
        if len(w) < min_len or len(w) > max_len:
            continue
        words.append(w)
    # unique, stable order
    seen = set()
    unique = []
    for w in words:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        unique.append(w)
    if limit:
        unique = unique[:limit]
    return unique

def build_prompts(words: list[str], templates: list[str] = DEFAULT_TEMPLATES):
    prompts = []
    for w in words:
        for t in templates:
            prompts.append(t.format(w))
    return prompts
