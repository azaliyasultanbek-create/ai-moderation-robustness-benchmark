# Robustness of AI-Based Content Moderation to Character-Level Obfuscation

[![CI](https://github.com/zeinepka07/ai-moderation-robustness-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/zeinepka07/ai-moderation-robustness-benchmark/actions/workflows/ci.yml)

**Authors:** Azaliya Sultanbek, Zeinep Egemkul (Narxoz University, 2026)
**Course project:** *The Effectiveness of AI-Based Content Moderation in Detecting Harmful Content on Social Media*: Task 2, measurement tool
**Methodology card:** [`docs/methodology_card.md`](docs/methodology_card.md) · **License:** MIT

---

## Overview

Social media users often evade automated moderation by obfuscating harmful words:
`idiot → 1d10t`, `trash → t*rash`, Latin letters swapped for look-alike Cyrillic
ones, deliberate typos. The literature reports that detectors break on rare,
noisy and out-of-distribution text (Zhang & Luo, 2019; Nejadgholi & Kiritchenko, 2020;
Ye et al., 2023), but these effects are seldom measured for different model families
under one controlled protocol.

This repository contains a **reproducible benchmark** that applies four
deterministic, human-like obfuscation strategies to a fixed hate-speech test set
and measures how much the **detection rate (recall) of harmful posts** drops for
three moderation models, together with false-positive rate, latency and memory.

## Research Question

> **RQ (refined RQ3 of Task 1):** How does character-level obfuscation of harmful
> posts (leetspeak, inserted separators, homoglyphs, typos) affect the recall
> of AI-based content moderation models, and does the effect differ between a
> word-level baseline, a character-level model and a pre-trained transformer?

| | Hypothesis |
|---|---|
| **H0** | For a given model, recall on harmful posts under obfuscation is **not lower** than on the original posts (no significant paired difference, McNemar one-sided, Holm-adjusted *p* ≥ 0.05). |
| **H1** | Obfuscation **reduces** recall on harmful posts by at least **5 percentage points** (Holm-adjusted *p* < 0.05). |

## Repository structure

```text
ai-moderation-robustness-benchmark/
├── .github/workflows/ci.yml      # CI: docker build + smoke test on every push
├── configs/
│   ├── config.yaml               # smoke test on data/sample (default)
│   └── experiment.yaml           # full Week-7 experiment on TweetEval-hate
├── data/
│   ├── README.md                 # data card (sources, schema, ethics)
│   ├── sample/                   # 8-row synthetic train/test files (versioned)
│   ├── raw/                      # downloaded datasets (git-ignored)
│   └── processed/                # derived files (git-ignored)
├── docs/
│   ├── methodology_card.md       # Methodology Passport Card
│   └── references.bib            # bibliography (Zotero export)
├── notebooks/                    # Week-8 analysis notebooks
├── scripts/
│   └── download_data.py          # reproducible download of TweetEval-hate
├── src/
│   ├── benchmark.py              # experiment runner (entry point)
│   ├── models.py                 # detectors: TF-IDF + LogReg, HF transformer
│   ├── perturbations.py          # obfuscation strategies (independent variable)
│   └── metrics.py                # metrics, McNemar test, bootstrap CI, Holm
├── tests/test_pipeline.py        # unit + end-to-end smoke tests
├── results/                      # run outputs (git-ignored)
├── Dockerfile
├── requirements.txt              # core deps, exact pins
├── requirements-transformer.txt  # optional transformer deps, exact pins
├── CITATION.cff
└── LICENSE
```

## Planned benchmarks (Week 7)

Full factorial design: **3 models × 5 text conditions = 15 runs**, each on the same
2,970 test tweets (TweetEval-hate test split), seed 42.

| Model (IV 1) | Type | Training | Role |
|---|---|---|---|
| `tfidf_word_lr` | TF-IDF word 1–2-grams + logistic regression | 9,000 clean train tweets | **Baseline** |
| `tfidf_char_lr` | TF-IDF char 2–5-grams + logistic regression | 9,000 clean train tweets | Classical ML, sub-word features |
| `roberta_hate` | [`cardiffnlp/twitter-roberta-base-hate-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest) | pre-trained, used as-is | Off-the-shelf AI moderator |

| Text condition (IV 2) | Example (`stupid idiot`) |
|---|---|
| `none` (control) | stupid idiot |
| `leetspeak` | 57up1d 1d107 |
| `char_insert` | st-upid id.iot |
| `homoglyph` | stupіd іdіot *(Cyrillic і)* |
| `char_swap` | sutpid idoit |

`word_rate = 0.5`: half of all eligible words (≥ 4 letters, not `@mentions`/URLs) are obfuscated.

| Measured per run | Unit |
|---|---|
| **Recall on harmful class (primary)** and Δ recall vs. `none` | %, percentage points |
| Precision, F1, macro-F1, ROC-AUC | 0–1 |
| False-positive rate on benign posts *(guardrail ≤ 0.20)* | 0–1 |
| Latency per text: mean, p50, p95 *(guardrail p95 ≤ 100 ms)* | ms |
| Throughput | texts/s |
| Peak resident memory *(guardrail ≤ 4096 MB)* | MB |
| Failed or timed-out inferences *(guardrail = 0)* | count |

## System requirements

- **Docker** ≥ 24 (recommended), or **Python 3.13** with `pip`
- CPU only, no GPU needed. ≥ 4 GB RAM (8 GB for `roberta_hate`)
- Disk: ~0.5 GB (core image), ~2.5 GB (image with transformer + model cache)
- Internet only for the full experiment (dataset + model download)

## Quickstart Guide (smoke test, ~1 minute)

```bash
git clone https://github.com/zeinepka07/ai-moderation-robustness-benchmark.git && cd ai-moderation-robustness-benchmark
docker build -t ai-mod-bench .
docker run --rm -v "${PWD}/results:/app/results" ai-mod-bench
```

The run prints a summary table and writes `results/smoke_test_<timestamp>/`.
On Windows `cmd.exe` replace `${PWD}` with `%cd%` (PowerShell works as is).

**Without Docker:**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.benchmark --config configs/config.yaml
```

Run the tests: `python -m unittest discover -s tests -v`

## Full experiment (Week 7)

```bash
docker build -t ai-mod-bench:full --build-arg WITH_TRANSFORMER=1 .
docker run --rm -v "${PWD}/data:/app/data" --entrypoint python ai-mod-bench:full scripts/download_data.py
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" \
           -v hf-cache:/cache/huggingface ai-mod-bench:full --config configs/experiment.yaml
```

Without Docker: `pip install -r requirements-transformer.txt`, `python scripts/download_data.py`,
`python -m src.benchmark --config configs/experiment.yaml`. Expected time on a 4-core
laptop CPU is 10–20 minutes. The transformer accounts for most of it.

## Outputs

Every run creates `results/<experiment>_<YYYYmmdd-HHMMSS>/`:

| File | Content |
|---|---|
| `benchmark.log` | full console log of the run |
| `metrics.csv` | one row per model × condition: all dependent variables + guardrail flags |
| `hypothesis_tests.csv` | recall clean vs. perturbed, Δ pp, bootstrap 95% CI, McNemar *p*, Holm *p*, `h0_rejected` |
| `predictions.csv` | item-level scores and predictions (for error analysis) |
| `run_metadata.json` | config snapshot, SHA-256 of data files, Python/OS/CPU/RAM, package versions, git commit |

## Configuration

All experimental parameters live in `configs/*.yaml`: data paths, models and their
hyper-parameters (IV 1), perturbation types, `word_rate` and `apply_to` (IV 2),
decision threshold, batch size, warm-up batches, timeout per condition, bootstrap
iterations, α, minimal effect size, guardrail thresholds and CPU threads.

## Reproducibility

- Exact version pins for every dependency (including transitive ones); `pip check` runs during `docker build`.
- GitHub Actions rebuilds the Docker image and runs the smoke test and unit tests on every push (badge at the top).
- Fixed seed (42) for perturbations, model initialisation and bootstrap; perturbations are seeded per item, so every model sees identical inputs.
- Models are trained on **clean** text only; obfuscation is applied at test time (adversarial setting).
- Latency excludes warm-up batches; the transformer's Hub commit hash is logged.

## Citation

```bibtex
@software{sultanbek_egemkul_2026_moderation_bench,
  author  = {Sultanbek, Azaliya and Egemkul, Zeinep},
  title   = {Robustness of AI-Based Content Moderation to Character-Level Obfuscation of Harmful Content},
  year    = {2026},
  url     = {https://github.com/zeinepka07/ai-moderation-robustness-benchmark},
  version = {0.1.0}
}
```

Dataset: Basile et al. (2019), SemEval-2019 Task 5; Barbieri et al. (2020), TweetEval.
Full bibliography: [`docs/references.bib`](docs/references.bib).

## License

Code is released under the [MIT License](LICENSE). The TweetEval data and the
pre-trained model keep their original licenses and are not redistributed here.

## Contributions

| Author | Contribution |
|---|---|
| Zeinep Egemkul | Research design, hypotheses (H0/H1), variable matrix and metrics, methodology card |
| Azaliya Sultanbek | Literature review, repository setup, Docker environment, README |
