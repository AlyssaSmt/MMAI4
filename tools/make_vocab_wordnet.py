# make_vocab_wordnet.py
# wird nicht aktiv genutzt, da WordNet-Vokabular schlechter ist als das aus make_vocab_common_words.py!!!!!
import re
import random
import nltk
from collections import defaultdict
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
nltk.download("omw-1.4")

TARGET_SIZE = 10_000
OUTFILE = "vocab.txt"
SEED = 42

random.seed(SEED)

def clean_term(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9 \-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def is_good_term(s: str) -> bool:
    if len(s) < 2 or len(s) > 32:
        return False
    if s.isdigit():
        return False
    if s.count(" ") > 3:
        return False
    return True

def main():
    buckets = defaultdict(set)

    # Alle Nomen sammeln
    for syn in wn.all_synsets(pos=wn.NOUN):
        for lemma in syn.lemma_names():
            term = clean_term(lemma)
            if is_good_term(term):
                first = term[0]
                if first.isalpha():
                    buckets[first].add(term)

    letters = sorted(buckets.keys())
    per_letter = TARGET_SIZE // len(letters)

    vocab = []

    for letter in letters:
        words = list(buckets[letter])
        random.shuffle(words)
        vocab.extend(words[:per_letter])

    # Falls noch Platz ist (Rundungsfehler)
    if len(vocab) < TARGET_SIZE:
        rest = []
        for letter in letters:
            rest.extend(buckets[letter])
        rest = list(set(rest) - set(vocab))
        random.shuffle(rest)
        vocab.extend(rest[: TARGET_SIZE - len(vocab)])

    random.shuffle(vocab)

    with open(OUTFILE, "w", encoding="utf-8") as f:
        for w in vocab[:TARGET_SIZE]:
            f.write(w + "\n")

    print(f"✅ vocab.txt geschrieben ({TARGET_SIZE} Wörter)")
    print("Beispiel:", vocab[:30])

if __name__ == "__main__":
    main()
