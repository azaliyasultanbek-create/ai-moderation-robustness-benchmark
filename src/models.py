"""Detectors under test (levels of the 'model' independent variable).

Every detector exposes the same interface:
    fit(texts, labels)        -> None   (no-op for pre-trained models)
    predict_proba(texts)      -> np.ndarray of P(harmful) in [0, 1]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


class Detector:
    name: str = "detector"
    trainable: bool = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:  # pragma: no cover
        return None

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class TfidfLogReg(Detector):
    """Classical ML moderation model: TF-IDF features + logistic regression.

    analyzer="word"    -> word n-grams (the honest, un-optimised baseline)
    analyzer="char_wb" -> character n-grams inside word boundaries
    """

    trainable = True

    def __init__(
        self,
        name: str,
        analyzer: str = "word",
        ngram_range: Sequence[int] = (1, 2),
        max_features: int | None = 50000,
        min_df: int = 1,
        lowercase: bool = True,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        max_iter: int = 1000,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=analyzer,
                        ngram_range=tuple(ngram_range),
                        max_features=max_features,
                        min_df=min_df,
                        lowercase=lowercase,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C, class_weight=class_weight, max_iter=max_iter, random_state=seed
                    ),
                ),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if len(set(labels)) < 2:
            raise ValueError(f"[{self.name}] training data must contain both classes")
        self.pipeline.fit(list(texts), list(labels))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]


class HFTextClassifier(Detector):
    """Off-the-shelf pre-trained transformer moderation model (Hugging Face Hub).

    Requires the optional dependencies in requirements-transformer.txt.
    Runs on CPU; the model is downloaded once into the Hugging Face cache.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        positive_labels: Sequence[str],
        max_length: int = 128,
        num_threads: int | None = None,
        revision: str | None = None,
        seed: int = 42,
    ) -> None:
        try:
            import torch
            from transformers import pipeline, set_seed
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                f"[{name}] needs torch + transformers. Install requirements-transformer.txt "
                "or build the Docker image with --build-arg WITH_TRANSFORMER=1"
            ) from exc

        self.name = name
        self.max_length = max_length
        self.positive = {lbl.lower() for lbl in positive_labels}
        if num_threads:
            torch.set_num_threads(int(num_threads))
        set_seed(seed)
        log.info("[%s] loading %s (revision=%s) on CPU", name, model_id, revision or "main")
        self.pipe = pipeline(
            "text-classification", model=model_id, tokenizer=model_id, revision=revision, device=-1
        )
        # exact model snapshot used (controlled variable) -> written to the log
        self.model_commit = getattr(self.pipe.model.config, "_commit_hash", None)
        log.info("[%s] model snapshot commit: %s", name, self.model_commit)
        available = {str(v).lower() for v in self.pipe.model.config.id2label.values()}
        missing = self.positive - available
        if missing:
            raise ValueError(
                f"[{name}] positive_labels {sorted(missing)} not in model labels {sorted(available)}"
            )

    @staticmethod
    def _harm_score(item: Any, positive: set[str]) -> float:
        rows: List[Dict[str, Any]] = [item] if isinstance(item, dict) else list(item)
        scores = [float(r["score"]) for r in rows if str(r["label"]).lower() in positive]
        if not scores:
            raise ValueError(f"no positive label in model output: {rows}")
        return max(scores)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        outputs = self.pipe(
            list(texts),
            batch_size=len(texts),
            top_k=None,
            truncation=True,
            max_length=self.max_length,
        )
        return np.array([self._harm_score(o, self.positive) for o in outputs], dtype=float)


def build_detector(spec: Dict[str, Any], seed: int, num_threads: int | None) -> Detector:
    kind = spec.get("type")
    params = dict(spec.get("params") or {})
    name = spec["name"]
    if kind == "tfidf_lr":
        return TfidfLogReg(name=name, seed=seed, **params)
    if kind == "hf_pipeline":
        params.setdefault("num_threads", num_threads)
        return HFTextClassifier(name=name, seed=seed, **params)
    raise ValueError(f"Unknown model type '{kind}' for model '{name}'")
