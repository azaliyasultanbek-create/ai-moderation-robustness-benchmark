"""Download the TweetEval 'hate' task (SemEval-2019 Task 5, HatEval, English).

Source : https://github.com/cardiffnlp/tweeteval  (Barbieri et al., 2020)
Output : data/raw/tweeteval_hate/{train,val,test}.csv  with columns id,text,label
         label 1 = hateful, 0 = not hateful.

The raw data is NOT committed to Git (see .gitignore); this script makes the
download reproducible and prints SHA-256 checksums for the run log.

Usage: python scripts/download_data.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/hate"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "tweeteval_hate"
SPLITS = {"train": "train", "val": "val", "test": "test"}


def fetch(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        return resp.read().decode("utf-8").splitlines()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, prefix in SPLITS.items():
        texts = fetch(f"{BASE}/{prefix}_text.txt")
        labels = fetch(f"{BASE}/{prefix}_labels.txt")
        if len(texts) != len(labels):
            print(f"[error] {split}: {len(texts)} texts vs {len(labels)} labels", file=sys.stderr)
            return 1
        path = OUT / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "text", "label"])
            for i, (t, y) in enumerate(zip(texts, labels)):
                writer.writerow([f"{split}_{i:05d}", t.strip(), int(y.strip())])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        n_pos = sum(int(y) for y in labels)
        print(f"{split:5s}: {len(texts):5d} rows ({n_pos} hateful) -> {path.relative_to(OUT.parents[2])}  sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
