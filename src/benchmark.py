"""Benchmark runner: AI moderation robustness to character-level obfuscation.

Usage:
    python -m src.benchmark --config configs/config.yaml

For every model x perturbation condition the runner measures detection quality
(recall, precision, F1, macro-F1, FPR, ROC-AUC), latency (ms/text), throughput
and peak RAM, then tests H0 with a paired McNemar test against the clean
condition of the same model. All outputs go to results/<run_id>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):  # allow `python src/benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psutil
import yaml

from src.metrics import (
    classification_metrics,
    holm_correction,
    latency_stats,
    mcnemar_one_sided,
    paired_bootstrap_delta,
)
from src.models import build_detector
from src.perturbations import AVAILABLE, perturb_many

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("benchmark")

TRACKED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "psutil",
    "torch", "transformers", "tokenizers",
]


# --------------------------------------------------------------------------- utils
class PeakRSS:
    """Samples the process resident memory in a background thread."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.proc = psutil.Process()
        self.interval = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRSS":
        self.peak = self.proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20


def resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(path: Path, cfg: Dict[str, Any], max_rows: int | None, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. For the full experiment run "
            "`python scripts/download_data.py` first (see README)."
        )
    df = pd.read_csv(path)
    tcol, lcol = cfg["text_column"], cfg["label_column"]
    missing = {tcol, lcol} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df = df.rename(columns={tcol: "text", lcol: "label"})[["id", "text", "label"]]
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError(f"{path.name}: labels must be 0 (benign) / 1 (harmful)")
    if max_rows and len(df) > max_rows:
        # stratified subsample keeps the class ratio
        frac = max_rows / len(df)
        parts = [df[df["label"] == y].sample(frac=frac, random_state=seed) for y in (0, 1)]
        df = pd.concat(parts).sort_index().reset_index(drop=True)
    return df


def environment_info() -> Dict[str, Any]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    vm = psutil.virtual_memory()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_mb": round(vm.total / 2**20),
        "packages": versions,
        "git_commit": commit,
    }


def setup_logging(run_dir: Path, level: str) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "benchmark.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


# ------------------------------------------------------------------------ core
def run_condition(detector, texts: List[str], batch_size: int, warmup: int, timeout_s: float):
    """Returns scores, per-item latency (ms), wall time, peak RAM (MB), n_errors."""
    for b in range(min(warmup, max(1, len(texts) // batch_size))):  # warm-up, not timed
        detector.predict_proba(texts[b * batch_size:(b + 1) * batch_size] or texts[:1])

    scores = np.full(len(texts), np.nan)
    per_item_ms: List[float] = []
    errors = 0
    t_start = time.perf_counter()
    with PeakRSS() as mem:
        for start in range(0, len(texts), batch_size):
            if time.perf_counter() - t_start > timeout_s:
                log.error("timeout after %.0fs: %d items not scored", timeout_s, len(texts) - start)
                errors += len(texts) - start
                break
            batch = texts[start:start + batch_size]
            t0 = time.perf_counter()
            try:
                scores[start:start + len(batch)] = detector.predict_proba(batch)
            except Exception:  # noqa: BLE001 - every failure is counted, run continues
                log.exception("inference failed for batch starting at %d", start)
                errors += len(batch)
                continue
            per_item_ms.extend([(time.perf_counter() - t0) * 1000 / len(batch)] * len(batch))
    wall = time.perf_counter() - t_start
    return scores, per_item_ms, wall, mem.peak_mb, errors


def main(argv: List[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yaml", help="path to YAML config")
    args = ap.parse_args(argv)

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    exp, dcfg, pcfg, ecfg, gcfg = (cfg[k] for k in ("experiment", "data", "perturbations", "evaluation", "guardrails"))
    seed = int(exp.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    run_id = f"{exp['name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = resolve(exp.get("output_dir", "results")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, exp.get("log_level", "INFO"))
    log.info("run %s | config %s", run_id, cfg_path.relative_to(ROOT) if cfg_path.is_relative_to(ROOT) else cfg_path)

    unknown = set(pcfg["types"]) - set(AVAILABLE)
    if unknown:
        raise ValueError(f"unknown perturbations {unknown}; available: {AVAILABLE}")
    if "none" not in pcfg["types"]:
        raise ValueError("perturbations.types must include 'none' (the control condition)")

    train_path, test_path = resolve(dcfg.get("train_path")), resolve(dcfg["test_path"])
    test = load_split(test_path, dcfg, dcfg.get("max_test_samples"), seed)
    train = load_split(train_path, dcfg, dcfg.get("max_train_samples"), seed) if train_path else None
    log.info("test: %d items (%d harmful) | train: %s", len(test), int(test["label"].sum()),
             f"{len(train)} items" if train is not None else "none")

    meta = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "data_sha256": {p.name: sha256(p) for p in (train_path, test_path) if p is not None},
        "environment": environment_info(),
    }
    runtime = cfg.get("runtime", {})
    threshold = float(ecfg.get("threshold", 0.5))
    bs = int(ecfg.get("batch_size", 32))

    # Perturbed test sets are generated once and shared by all models.
    test_sets: Dict[str, List[str]] = {}
    n_changed: Dict[str, List[int]] = {}
    harmful_mask = test["label"].to_numpy() == 1
    for kind in pcfg["types"]:
        texts, counts = perturb_many(
            test["text"].tolist(), test["id"].astype(str).tolist(), kind,
            float(pcfg.get("word_rate", 0.5)), int(pcfg.get("min_word_length", 4)), seed,
        )
        if pcfg.get("apply_to", "all") == "harmful_only":
            texts = [t if h else o for t, o, h in zip(texts, test["text"], harmful_mask)]
            counts = [c if h else 0 for c, h in zip(counts, harmful_mask)]
        test_sets[kind], n_changed[kind] = texts, counts

    metric_rows, pred_frames = [], []
    for spec in cfg["models"]:
        if not spec.get("enabled", True):
            log.info("[%s] disabled in config, skipped", spec["name"])
            continue
        detector = build_detector(spec, seed, runtime.get("num_threads"))
        train_time = 0.0
        if getattr(detector, "trainable", False):
            if train is None:
                raise ValueError(f"model {spec['name']} needs data.train_path")
            t0 = time.perf_counter()
            detector.fit(train["text"].tolist(), train["label"].tolist())  # trained on CLEAN text only
            train_time = time.perf_counter() - t0
            log.info("[%s] trained on %d clean texts in %.2fs", spec["name"], len(train), train_time)

        for kind, texts in test_sets.items():
            scores, per_item, wall, peak_mb, errors = run_condition(
                detector, texts, bs, int(ecfg.get("warmup_batches", 1)), float(ecfg.get("timeout_s_per_condition", 600))
            )
            ok = ~np.isnan(scores)
            preds = np.where(ok, (np.nan_to_num(scores) >= threshold).astype(int), -1)
            row = {"model": spec["name"], "perturbation": kind}
            row.update(classification_metrics(test["label"][ok], preds[ok], scores[ok]))
            row.update(latency_stats(per_item, wall, int(ok.sum())))
            row.update({
                "peak_ram_mb": round(peak_mb, 1),
                "errors": errors,
                "error_rate": errors / len(texts),
                "train_time_s": round(train_time, 3),
                "mean_words_perturbed": float(np.mean(n_changed[kind])),
            })
            row["guardrail_fpr_ok"] = bool(row["fpr"] <= gcfg["max_fpr"])
            row["guardrail_latency_ok"] = bool(row["latency_ms_p95"] <= gcfg["max_latency_p95_ms"])
            row["guardrail_ram_ok"] = bool(row["peak_ram_mb"] <= gcfg["max_peak_ram_mb"])
            row["guardrail_errors_ok"] = bool(row["error_rate"] <= gcfg["max_error_rate"])
            metric_rows.append(row)
            log.info(
                "[%s | %-11s] recall=%.3f macroF1=%.3f FPR=%.3f p95=%.2fms RAM=%.0fMB errors=%d",
                spec["name"], kind, row["recall"], row["macro_f1"], row["fpr"],
                row["latency_ms_p95"], row["peak_ram_mb"], errors,
            )
            pred_frames.append(pd.DataFrame({
                "id": test["id"], "model": spec["name"], "perturbation": kind,
                "label": test["label"], "score": scores, "pred": preds,
                "correct": preds == test["label"].to_numpy(),
                "n_words_perturbed": n_changed[kind], "text": texts,
            }))

    metrics = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_frames, ignore_index=True)

    # ---------------------------------------------------------------- H0 test
    stats_rows = []
    harm = preds_df[preds_df["label"] == 1]
    for model, g in harm.groupby("model", sort=False):
        clean = g[g["perturbation"] == "none"].set_index("id")["correct"]
        for kind in [k for k in pcfg["types"] if k != "none"]:
            pert = g[g["perturbation"] == kind].set_index("id")["correct"].reindex(clean.index)
            res = mcnemar_one_sided(clean.to_numpy(), pert.to_numpy())
            res.update(paired_bootstrap_delta(clean.to_numpy(), pert.to_numpy(),
                                              int(ecfg.get("bootstrap_iterations", 1000)), seed))
            stats_rows.append({
                "model": model, "perturbation": kind, "n_harmful": int(len(clean)),
                "recall_clean": float(clean.mean()), "recall_perturbed": float(pert.mean()),
                "delta_recall_pp": float((clean.mean() - pert.mean()) * 100), **res,
            })
    stats = pd.DataFrame(stats_rows)
    if not stats.empty:
        stats["p_holm"] = holm_correction(stats["p_value"].tolist())
        stats["h0_rejected"] = (stats["p_holm"] < float(ecfg.get("alpha", 0.05))) & (
            stats["delta_recall_pp"] >= float(ecfg.get("min_effect_pp", 5.0))
        )

    # ---------------------------------------------------------------- outputs
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    stats.to_csv(run_dir / "hypothesis_tests.csv", index=False)
    preds_df.to_csv(run_dir / "predictions.csv", index=False)
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        cols = ["model", "perturbation", "recall", "macro_f1", "fpr", "latency_ms_p95", "peak_ram_mb", "errors"]
        log.info("SUMMARY\n%s", metrics[cols].round(3).to_string(index=False))
        if not stats.empty:
            scols = ["model", "perturbation", "recall_clean", "recall_perturbed",
                     "delta_recall_pp", "p_holm", "h0_rejected"]
            log.info("HYPOTHESIS TESTS (H1: recall drops under perturbation)\n%s",
                     stats[scols].round(4).to_string(index=False))
    log.info("done -> %s", run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir)
    return run_dir


if __name__ == "__main__":
    main()
