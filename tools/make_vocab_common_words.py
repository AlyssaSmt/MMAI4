# make_vocab_common_words.py
import re

INFILE = "google-10000-english-no-swears.txt"
OUTFILE = "vocab.txt"

def is_good_word(w):
    if len(w) < 3 or len(w) > 20:
        return False
    if not w.isalpha():
        return False
    return True

with open(INFILE) as f:
    words = [w.strip().lower() for w in f if is_good_word(w.strip())]

print("Total words:", len(words))

with open(OUTFILE, "w") as f:
    for w in words:
        f.write(w + "\n")

print("✅ vocab.txt geschrieben")
