"""Proposed method: normalise text BEFORE the moderation model sees it.

Written from general rules of how people disguise words online, not from our own
obfuscation code (src/perturbations.py):
  1. Unicode NFKC + lower case            ("ＩＤＩＯＴ", "IdIoT"   -> "idiot")
  2. look-alike Cyrillic/Greek letters     ("іdіоt"              -> "idiot")
  3. separators inside a word are removed  ("id.iot", "st*upid"  -> "idiot", "stupid")
  4. digits/symbols used as letters        ("1d10t", "$tupid"    -> "idiot", "stupid")
  5. a letter repeated 3+ times -> once    ("traaaash"           -> "trash")
Words without any letter (numbers), @mentions, #hashtags and links are left alone.
"""

import re
import unicodedata

LOOKALIKE = dict(zip("аеорсхуіјкмнтвѕһӏοαεικνρτυχ", "aeopcxyijkmhtbshloaeiknptux"))
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}


def normalize_word(word: str) -> str:
    if word.startswith(("@", "#", "http")) or not re.search(r"[a-z]", word):
        return word
    word = re.sub(r"(?<=\w)[.*_\-](?=\w)", "", word)       # rule 3
    word = "".join(LEET.get(ch, ch) for ch in word)          # rule 4
    return re.sub(r"(.)\1{2,}", r"\1", word)                 # rule 5


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()                # rule 1
    text = "".join(LOOKALIKE.get(ch, ch) for ch in text)              # rule 2
    return " ".join(normalize_word(w) for w in text.split())
