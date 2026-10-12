"""Character-level obfuscation of text (the manipulated independent variable).

Each perturbation imitates a human-written evasion strategy described in the
literature (e.g. NoisyHate, Ye et al. 2023): leetspeak, inserted separators,
look-alike Unicode characters (homoglyphs) and typo-like character swaps.

All perturbations are deterministic: the same (seed, perturbation, item id)
always produces the same output, so every run sees identical test inputs.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

# Latin -> visually identical Cyrillic / other Unicode letters.
HOMOGLYPH_MAP = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у", "i": "і",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х",
}

SEPARATORS = [".", "*", "-", "_"]

_TOKEN_CORE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")
_SPLIT_KEEP_WS = re.compile(r"(\s+)")


def _leetspeak(word: str, rng: random.Random) -> str:
    out = "".join(LEET_MAP.get(ch.lower(), ch) for ch in word)
    if out == word:  # no mappable letters -> fall back to one separator
        return _char_insert(word, rng)
    return out


def _char_insert(word: str, rng: random.Random) -> str:
    pos = rng.randint(1, len(word) - 1)
    return word[:pos] + rng.choice(SEPARATORS) + word[pos:]


def _homoglyph(word: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(word) if ch in HOMOGLYPH_MAP]
    if not idx:
        return _char_insert(word, rng)
    # replace about half of the replaceable letters (at least one)
    k = max(1, len(idx) // 2)
    chosen = set(rng.sample(idx, k))
    return "".join(HOMOGLYPH_MAP[ch] if i in chosen else ch for i, ch in enumerate(word))


def _char_swap(word: str, rng: random.Random) -> str:
    # swap two adjacent interior characters, keep first and last letter
    if len(word) < 4:
        return word[::-1] if len(word) == 2 else word
    pos = rng.randint(1, len(word) - 3)
    chars = list(word)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    swapped = "".join(chars)
    if swapped == word:  # e.g. "letter": swapped identical letters
        return _char_insert(word, rng)
    return swapped


PERTURBATIONS: Dict[str, Callable[[str, random.Random], str]] = {
    "leetspeak": _leetspeak,
    "char_insert": _char_insert,
    "homoglyph": _homoglyph,
    "char_swap": _char_swap,
}

AVAILABLE = ["none", *PERTURBATIONS.keys()]


@dataclass(frozen=True)
class PerturbationResult:
    text: str
    n_eligible: int
    n_perturbed: int


def _eligible(token: str, min_len: int) -> bool:
    if not token or token[0] in "@#" or token.lower().startswith(("http", "www.")):
        return False
    m = _TOKEN_CORE.match(token)
    return bool(m) and len(m.group(2)) >= min_len


def perturb_text(
    text: str,
    kind: str,
    word_rate: float,
    min_word_length: int,
    seed: int,
    item_id: str,
) -> PerturbationResult:
    """Perturb ``ceil(word_rate * n_eligible)`` randomly chosen words of ``text``."""
    if kind == "none":
        return PerturbationResult(text, 0, 0)
    if kind not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation '{kind}'. Available: {AVAILABLE}")
    if not 0.0 < word_rate <= 1.0:
        raise ValueError("word_rate must be in (0, 1]")

    rng = random.Random(f"{seed}|{kind}|{item_id}")
    parts: List[str] = _SPLIT_KEEP_WS.split(text)
    eligible = [i for i, p in enumerate(parts) if not p.isspace() and _eligible(p, min_word_length)]
    if not eligible:
        return PerturbationResult(text, 0, 0)

    k = min(len(eligible), max(1, math.ceil(word_rate * len(eligible))))
    fn = PERTURBATIONS[kind]
    for i in rng.sample(eligible, k):
        prefix, core, suffix = _TOKEN_CORE.match(parts[i]).groups()  # type: ignore[union-attr]
        parts[i] = prefix + fn(core, rng) + suffix
    return PerturbationResult("".join(parts), len(eligible), k)


def perturb_many(
    texts: List[str], ids: List[str], kind: str, word_rate: float, min_word_length: int, seed: int
) -> Tuple[List[str], List[int]]:
    out, counts = [], []
    for text, item_id in zip(texts, ids):
        res = perturb_text(text, kind, word_rate, min_word_length, seed, str(item_id))
        out.append(res.text)
        counts.append(res.n_perturbed)
    return out, counts
