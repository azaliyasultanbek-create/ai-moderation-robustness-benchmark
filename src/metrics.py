"""Dependent variables and the statistical test of H0.

Primary metric : recall (detection rate) on the harmful class.
Test of H0     : one-sided exact McNemar test (paired, same items clean vs.
                 perturbed), Holm-corrected across all comparisons, plus a
                 paired bootstrap 95% CI for the recall drop.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "roc_auc": auc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def latency_stats(per_item_ms: Sequence[float], wall_s: float, n: int) -> Dict[str, float]:
    arr = np.asarray(per_item_ms, dtype=float)
    if arr.size == 0:
        return {"latency_ms_mean": float("nan"), "latency_ms_p50": float("nan"),
                "latency_ms_p95": float("nan"), "throughput_items_s": float("nan")}
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "throughput_items_s": float(n / wall_s) if wall_s > 0 else float("nan"),
    }


def mcnemar_one_sided(correct_clean: Sequence[bool], correct_pert: Sequence[bool]) -> Dict[str, float]:
    """H1: the perturbed condition misses MORE harmful items than the clean one.

    b = detected when clean, missed when perturbed
    c = missed when clean, detected when perturbed
    Under H0, b ~ Binomial(b + c, 0.5); we test P(B >= b).
    """
    cc = np.asarray(correct_clean, dtype=bool)
    cp = np.asarray(correct_pert, dtype=bool)
    b = int(np.sum(cc & ~cp))
    c = int(np.sum(~cc & cp))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5, alternative="greater").pvalue)
    return {"b_lost": b, "c_gained": c, "p_value": p}


def paired_bootstrap_delta(
    correct_clean: Sequence[bool], correct_pert: Sequence[bool], iterations: int, seed: int
) -> Dict[str, float]:
    """95% CI for recall_clean - recall_perturbed (percentage points)."""
    cc = np.asarray(correct_clean, dtype=float)
    cp = np.asarray(correct_pert, dtype=float)
    n = cc.size
    if n == 0:
        return {"delta_ci_low_pp": float("nan"), "delta_ci_high_pp": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    deltas = (cc[idx].mean(axis=1) - cp[idx].mean(axis=1)) * 100
    return {
        "delta_ci_low_pp": float(np.percentile(deltas, 2.5)),
        "delta_ci_high_pp": float(np.percentile(deltas, 97.5)),
    }


def holm_correction(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()
