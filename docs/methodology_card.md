# Methodology Passport Card: Task 2

## 1. Project information

| | |
|---|---|
| **Title** | The Effectiveness of AI-Based Content Moderation in Detecting Harmful Content on Social Media: Robustness to Character-Level Obfuscation |
| **Authors (pair)** | Azaliya Sultanbek, Zeinep Egemkul (Narxoz University) |
| **GitHub repository** | https://github.com/zeinepka07/ai-moderation-robustness-benchmark |
| **Builds on** | Task 1 proposal, RQ3 (linguistic and contextual challenges → detection accuracy) |

## 2. Type of research

**Quantitative: controlled computational experiment with a paired (within-items) design.**

*Justification.* The hypotheses in Task 1 concern a measurable quantity, detection
accuracy, and how it changes when one factor is varied. The factor (obfuscation of
the text) can be manipulated directly and deterministically while everything else
(dataset, model weights, threshold, hardware, library versions) stays fixed. That
allows a causal interpretation and a formal significance test. Each test item is
evaluated both in its original and its obfuscated form, so items serve as their own
control and between-item noise is removed. A small qualitative error analysis of
missed posts (Week 8) will support interpretation, but the hypothesis decision is
purely quantitative.

## 3. Hypotheses

**Week 5 refinement.** In Task 1, RQ3 named sarcasm, slang, ambiguity and
code-switching as challenges. To make RQ3 empirically testable within one semester
on a CPU, we operationalise *linguistic challenge* as **character-level
obfuscation**: the human-written evasion strategies documented by NoisyHate
(Ye et al., 2023). They can be applied in a controlled, reproducible way.
RQ1 and RQ2 remain covered by the literature review.

> **Refined RQ:** How does character-level obfuscation of harmful posts affect the
> recall of AI-based content moderation models, and does the effect differ between
> model families?

| | Formulation |
|---|---|
| **H0 (null)** | For a given moderation model, recall on harmful posts with character-level obfuscation is **not lower** than recall on the same posts in their original form (recall_perturbed ≥ recall_clean; one-sided exact McNemar test, Holm-adjusted *p* ≥ 0.05). |
| **H1 (alternative)** | Character-level obfuscation **reduces** recall on harmful posts: recall_clean − recall_perturbed ≥ **5 percentage points**, with Holm-adjusted *p* < 0.05. |

*Decision rule.* H0 is rejected for a model × obfuscation pair only if the
difference is both statistically significant (α = 0.05 after Holm correction over
all 12 comparisons) and practically meaningful (≥ 5 pp). A paired bootstrap 95%
confidence interval (1,000 resamples) is reported for every difference.

*Secondary (exploratory) question:* is the recall drop of the pre-trained
transformer smaller than that of the word-level baseline? This is compared through
the confidence intervals and is not tested as a separate formal hypothesis.

## 4. Variable matrix

| Type | Variable | Levels / unit | Where set |
|---|---|---|---|
| **Independent** | Moderation model / algorithm | `tfidf_word_lr` (baseline), `tfidf_char_lr`, `roberta_hate` (pre-trained transformer) | `models:` in `config.yaml` |
| **Independent** | Text condition (obfuscation type) | `none` (control), `leetspeak`, `char_insert`, `homoglyph`, `char_swap` | `perturbations.types` |
| *Independent (fixed in main run, sweep optional)* | Obfuscation intensity | share of eligible words perturbed: 0.5 | `perturbations.word_rate` |
| **Dependent** | Recall (detection rate) on harmful class | % | `metrics.csv` |
| **Dependent** | Δ recall vs. clean condition | percentage points + 95% CI | `hypothesis_tests.csv` |
| **Dependent** | Precision, F1, macro-F1, ROC-AUC | 0–1 | `metrics.csv` |
| **Dependent** | False-positive rate on benign posts | 0–1 | `metrics.csv` |
| **Dependent** | Latency per text: mean / p50 / p95 | ms | `metrics.csv` |
| **Dependent** | Throughput | texts/s | `metrics.csv` |
| **Dependent** | Peak resident memory (RAM) | MB | `metrics.csv` |
| **Controlled** | Dataset | TweetEval-hate (SemEval-2019 HatEval, English): train 9,000, test 2,970; SHA-256 logged | `data:` |
| **Controlled** | Training data | clean training split only, no obfuscated examples | runner |
| **Controlled** | Random seed | 42 (perturbations, model init, bootstrap) | `experiment.seed` |
| **Controlled** | Decision threshold | P(harmful) ≥ 0.5 | `evaluation.threshold` |
| **Controlled** | Text eligible for obfuscation | words ≥ 4 letters; `@user`, hashtags, URLs untouched | `perturbations.min_word_length` |
| **Controlled** | Model hyper-parameters / snapshot | fixed in config; transformer Hub commit hash logged; max 128 tokens | `models.params` |
| **Controlled** | Batch size, warm-up | 32; 1 untimed warm-up batch | `evaluation` |
| **Controlled** | Hardware | CPU only, 4 threads; CPU model, cores and RAM logged | `runtime.num_threads` |
| **Controlled** | Software environment | Docker `python:3.13.5-slim-bookworm`; exact pins of all packages (numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, torch 2.6.0 CPU, transformers 4.49.0, …) | `Dockerfile`, `requirements*.txt` |

## 5. Metrics and baseline

| Role | Metric | Why |
|---|---|---|
| **Primary metric** | **Recall on the harmful class**, and its paired change Δ recall (pp) between the clean and the obfuscated version of the same posts | Obfuscation is used to make harmful posts pass moderation. A missed harmful post (false negative) is exactly the failure H1 predicts. |
| Secondary metrics | Precision, F1, macro-F1, ROC-AUC | Show whether the overall quality changes, not only recall |
| **Guardrail 1** | False-positive rate on benign posts ≤ **0.20** | A model that “keeps” recall by flagging everything would censor benign users. This is the over-moderation risk raised in Task 1. |
| **Guardrail 2** | p95 latency ≤ **100 ms per text** on CPU | Moderation must keep up with the stream of posts |
| **Guardrail 3** | Peak RAM ≤ **4,096 MB** | Feasibility on ordinary hardware |
| **Guardrail 4** | Error / timeout rate = **0%** (timeout 1,800 s per condition) | Every item must receive a decision, otherwise metrics are biased |

**Baseline / control.**
1. *Control condition*: the **original, non-obfuscated text** (`none`) of the same
   2,970 test posts. Every obfuscated condition is compared with it.
2. *Baseline model*: **TF-IDF word 1–2-grams + logistic regression**
   (`tfidf_word_lr`), a standard, un-optimised algorithm trained on the same clean
   training data. It represents keyword-style legacy moderation, and the two
   other models are compared against it. The pre-trained transformer
   (`cardiffnlp/twitter-roberta-base-hate-latest`, Antypas & Camacho-Collados, 2023)
   represents the current state of practice.

## 6. Threats to validity (short)

- **Construct:** synthetic obfuscation approximates, but does not equal, real
  human evasion. Examples are kept human-like, and NoisyHate (Ye et al., 2023) can
  serve as an external check.
- **Internal:** the transformer may have seen HatEval *training* data. Only the
  official test split is used.
- **External:** English Twitter data only. Results may not transfer to other
  languages or platforms (cf. Mullah et al., 2026; Nejadgholi & Kiritchenko, 2020).

## 7. Measurement tool (status at the end of Week 6)

`python -m src.benchmark --config configs/config.yaml` runs the full pipeline on
`data/sample/` (8 synthetic rows) and writes `metrics.csv`, `hypothesis_tests.csv`,
`predictions.csv`, `run_metadata.json` and `benchmark.log` to `results/`.
10 automated tests pass (`python -m unittest discover -s tests`).
The Week-7 run uses `configs/experiment.yaml`.
