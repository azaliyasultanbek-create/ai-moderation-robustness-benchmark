"""TASK 3, step 1: run the experiment and write the raw log  data/raw/test_run.jsonl

For every model and two pipelines - "raw" (text as is) and "normalized" (src/normalize.py first):
  quality  lines: which posts were flagged as harmful, as a string of 0/1 per post, for
                  TweetEval test (clean + 4 obfuscations), TweetEval val, NoisyHate (clean / human-perturbed)
  latency  lines: time of every model call on 128 clean test posts, batch sizes 1/8/64, 5 repeats;
                  the first calls of each repeat are marked phase="warmup" and are dropped later
  memory   lines: peak RAM of the process while the pipeline ran

Usage:  python scripts/collect_raw_log.py --config configs/task3.yaml
"""

import argparse, json, logging, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.benchmark import PeakRSS, environment_info
from src.models import build_detector
from src.normalize import normalize
from src.perturbations import perturb_many

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("collect")
PIPELINES = {"raw": lambda t: t, "normalized": normalize}


def read(path):
    return pd.read_csv(ROOT / path).fillna({"text": ""})


def flag_posts(det, texts, prep, threshold, batch=32):
    """Returns a 0/1 array (1 = flagged as harmful) and the number of failed posts."""
    flags, errors = np.zeros(len(texts), dtype=int), 0
    for s in range(0, len(texts), batch):
        try:
            scores = det.predict_proba([PIPELINES[prep](t) for t in texts[s:s + batch]])
            flags[s:s + batch] = scores >= threshold
        except Exception:  # a failed batch is counted, not hidden
            errors += len(texts[s:s + batch])
    return flags, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    cfg = yaml.safe_load((ROOT / ap.parse_args().config).read_text(encoding="utf-8"))
    seed, thr, t3, p = cfg["experiment"]["seed"], cfg["evaluation"]["threshold"], cfg["task3"], cfg["perturbations"]
    np.random.seed(seed)

    train, test, val = read(cfg["data"]["train_path"]), read(cfg["data"]["test_path"]), read(t3["val_path"])
    nh = pd.read_csv(ROOT / t3["noisyhate_path"])           # every NoisyHate post is toxic (toxicity >= 0.5)
    sets = {("tweeteval", k): perturb_many(test.text.tolist(), test.id.astype(str).tolist(), k,
                                           p["word_rate"], p["min_word_length"], seed)[0] for k in p["types"]}
    sets[("val", "none")] = val.text.tolist()
    sets[("noisyhate", "clean")] = nh.clean_version.tolist()
    sets[("noisyhate", "human_perturbed")] = nh.perturbed_version.tolist()
    labels = {"tweeteval": test.label.to_numpy(), "val": val.label.to_numpy(), "noisyhate": np.ones(len(nh), int)}
    sweep = [test.text[i] for i in np.random.default_rng(seed).choice(len(test), t3["sweep_n_texts"], replace=False)]

    out = open(ROOT / t3["output_path"], "w", encoding="utf-8")
    write = lambda rec: out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write({"type": "meta", "seed": seed, "threshold": thr, "n_test": len(test), "n_noisyhate": len(nh),
           "batch_sizes": t3["batch_sizes"], "repeats": t3["repeats"], "environment": environment_info()})

    for spec in [m for m in cfg["models"] if m.get("enabled", True)]:
        for prep in PIPELINES:
            try:
                det = build_detector(spec, seed, cfg["runtime"]["num_threads"])
            except Exception as e:  # e.g. no internet / no torch for the transformer
                log.warning("[%s] skipped: %s", spec["name"], e)
                break
            name = spec["name"]
            with PeakRSS() as mem:
                if getattr(det, "trainable", False):  # TF-IDF: train on CLEAN (and, if needed, normalised) text
                    det.fit([PIPELINES[prep](t) for t in train.text], train.label.tolist())
                for (ds, cond), texts in sets.items():
                    f, err = flag_posts(det, texts, prep, thr)
                    y = labels[ds]
                    write({"type": "quality", "model": name, "prep": prep, "dataset": ds, "condition": cond,
                           "harm": "".join(map(str, f[y == 1])), "benign": "".join(map(str, f[y == 0])), "errors": err})
                for bs in t3["batch_sizes"]:
                    for rep in range(t3["repeats"]):
                        calls = [("warmup", sweep[:bs])] * t3["warmup_batches"]          # cold start, dropped later
                        calls += [("eval", sweep[s:s + bs]) for s in range(0, len(sweep), bs)]
                        for phase, batch in calls:
                            t0 = time.perf_counter()
                            det.predict_proba([PIPELINES[prep](t) for t in batch])
                            write({"type": "latency", "model": name, "prep": prep, "batch_size": bs, "repeat": rep,
                                   "phase": phase, "n_items": len(batch), "ms": round((time.perf_counter() - t0) * 1000, 4)})
            write({"type": "memory", "model": name, "prep": prep, "peak_rss_mb": round(mem.peak_mb, 1)})
            log.info("[%s | %s] done", name, prep)
    out.close()


if __name__ == "__main__":
    main()
