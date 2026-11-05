"""TASK 3, step 2: from the raw log to tables.

    load_log()        read data/raw/test_run.jsonl
    drop_warmup()     remove cold-start calls (phase = "warmup")
    latency_table()   median, p25, p75, p95, p99, mean, SD of ms per post; throughput (posts/s); 95% CI
    quality_table()   recall and false-positive rate with 95% CI
    hypothesis_tests()  H1: does obfuscation lower recall?  H2: does normalisation bring it back?
Run:  python -m src.process_data   -> data/processed/metrics_summary.csv, latency_summary.csv, hypothesis_tests.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import holm_correction, mcnemar_one_sided, paired_bootstrap_delta

ROOT = Path(__file__).resolve().parent.parent
SEED, N_BOOT = 42, 2000


def load_log(path=ROOT / "data/raw/test_run.jsonl"):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    by_type = lambda t: pd.DataFrame([r for r in rows if r["type"] == t])
    return rows[0], by_type("latency"), by_type("quality"), by_type("memory")


def drop_warmup(lat):
    lat = lat.assign(ms_per_item=lat["ms"] / lat["n_items"])
    return lat[lat.phase == "eval"], lat[lat.phase == "warmup"]


def boot_ci(x, stat=np.median):
    """95% bootstrap confidence interval: resample the data 2000 times with a fixed seed."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    stats = [stat(rng.choice(x, len(x))) for _ in range(N_BOOT)]
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def latency_table(ev):
    rows = []
    for (m, prep, bs), g in ev.groupby(["model", "prep", "batch_size"], sort=False):
        x = g.ms_per_item.to_numpy()
        tput = g.groupby("repeat").apply(lambda r: r.n_items.sum() / r.ms.sum() * 1000, include_groups=False)
        rows.append(dict(model=m, prep=prep, batch_size=bs, n=len(x), median_ms=np.median(x),
                         median_ci=boot_ci(x), p25_ms=np.percentile(x, 25), p75_ms=np.percentile(x, 75),
                         p95_ms=np.percentile(x, 95), p99_ms=np.percentile(x, 99), mean_ms=x.mean(), sd_ms=x.std(ddof=1),
                         throughput=tput.mean(), throughput_ci=boot_ci(tput, np.mean)))
    t = pd.DataFrame(rows)
    for c in ("median_ci", "throughput_ci"):  # split (low, high) into two columns
        t[[c + "_low", c + "_high"]] = pd.DataFrame(t.pop(c).tolist(), index=t.index)
    return t


def bits(s):
    return np.array([c == "1" for c in s], bool)


def quality_table(q):
    rows = []
    for r in q.itertuples():
        h, b = bits(r.harm), bits(r.benign)
        rows.append(dict(model=r.model, prep=r.prep, dataset=r.dataset, condition=r.condition,
                         recall=h.mean(), recall_ci_low=boot_ci(h, np.mean)[0], recall_ci_high=boot_ci(h, np.mean)[1],
                         fpr=b.mean() if len(b) else np.nan, n_harmful=len(h), n_benign=len(b),
                         error_rate=r.errors / (len(h) + len(b))))
    return pd.DataFrame(rows)


def paired_test(family, model, label, first, second):
    """Same harmful posts, two versions. One-sided McNemar: does `first` catch MORE posts than `second`?"""
    a, b = bits(first), bits(second)
    return dict(family=family, model=model, comparison=label, recall_first=a.mean(), recall_second=b.mean(),
                delta_pp=(a.mean() - b.mean()) * 100, **mcnemar_one_sided(a, b), **paired_bootstrap_delta(a, b, N_BOOT, SEED))


def hypothesis_tests(q, min_effect_pp=5.0, alpha=0.05):
    get = lambda m, prep, ds, c: q[(q.model == m) & (q.prep == prep) & (q.dataset == ds) & (q.condition == c)].harm.iloc[0]
    rows = []
    for m in q.model.unique():
        cases = [("tweeteval", "none", c) for c in q[(q.dataset == "tweeteval") & (q.condition != "none")].condition.unique()]
        cases += [("noisyhate", "clean", "human_perturbed")]
        for ds, clean, obf in cases:
            # H1 (Task 2): the clean post is caught more often than its obfuscated version (raw pipeline)
            rows.append(paired_test("H1", m, f"{ds}: clean vs {obf}", get(m, "raw", ds, clean), get(m, "raw", ds, obf)))
            # H2 (Task 3): on the obfuscated post, normalisation catches more than the raw pipeline
            rows.append(paired_test("H2", m, f"{ds} {obf}: normalized vs raw",
                                    get(m, "normalized", ds, obf), get(m, "raw", ds, obf)))
    t = pd.DataFrame(rows)
    for fam in ("H1", "H2"):  # Holm correction inside each family of tests
        f = t.family == fam
        t.loc[f, "p_holm"] = holm_correction(t.loc[f, "p_value"].tolist())
    t["supported"] = (t.p_holm < alpha) & (t.delta_pp >= min_effect_pp)  # decision rule of the methodology card
    return t


if __name__ == "__main__":
    meta, lat, q, mem = load_log()
    ev, wu = drop_warmup(lat)
    out = ROOT / "data/processed"
    out.mkdir(exist_ok=True)
    quality_table(q).to_csv(out / "metrics_summary.csv", index=False, float_format="%.4f")
    latency_table(ev).to_csv(out / "latency_summary.csv", index=False, float_format="%.4f")
    hypothesis_tests(q).to_csv(out / "hypothesis_tests.csv", index=False, float_format="%.6g")
    print(f"dropped {len(wu)} warm-up calls, kept {len(ev)}; tables written to {out}")
