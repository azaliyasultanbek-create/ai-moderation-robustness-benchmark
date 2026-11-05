"""TASK 3, step 3: figures. Rules: axes with units, 95% CI always shown, colour-blind palette
plus different markers / line styles / hatches (readable in black and white). Saved as .pdf and 300-dpi .png.
Run:  python -m src.visualize
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.process_data import drop_warmup, latency_table, load_log, quality_table

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
COLORS = dict(zip(["tfidf_word_lr", "tfidf_char_lr", "roberta_hate"], sns.color_palette("colorblind", 3)))
MARKER = {"tfidf_word_lr": "o", "tfidf_char_lr": "s", "roberta_hate": "^"}
LINE = {"raw": "-", "normalized": "--"}
HATCH = {"raw": "", "normalized": "//"}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"pdf.fonttype": 42, "legend.frameon": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def fig1_latency_vs_load(lat, guardrail_ms=100):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (m, prep), g in lat.groupby(["model", "prep"], sort=False):
        ax.plot(g.batch_size, g.median_ms, color=COLORS[m], marker=MARKER[m], ls=LINE[prep], label=f"{m}, {prep}")
        ax.fill_between(g.batch_size, g.median_ci_low, g.median_ci_high, color=COLORS[m], alpha=0.25)
    ax.axhline(guardrail_ms, color="0.3", ls=":", label=f"guardrail p95 ≤ {guardrail_ms} ms")
    ax.set(xscale="log", yscale="log", xticks=sorted(lat.batch_size.unique()),
           xlabel="Load: batch size (posts per call)", ylabel="Latency per post, median (ms)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, title="band = 95% CI", title_fontsize=8)
    save(fig, "fig1_latency_vs_load")
    return fig


def fig2_distribution_boxplot(ev):
    d = ev[ev.batch_size == 1]
    groups = list(d.groupby(["model", "prep"], sort=False))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bp = ax.boxplot([g.ms_per_item for _, g in groups], notch=True, patch_artist=True, flierprops=dict(markersize=2))
    for box, ((m, prep), _) in zip(bp["boxes"], groups):
        box.set(facecolor=COLORS[m], alpha=0.7, hatch=HATCH[prep])
    ax.set_xticks(range(1, len(groups) + 1), [f"{m}\n{prep}" for (m, prep), _ in groups], fontsize=8)
    ax.set(yscale="log", ylabel="Latency per post, batch size 1 (ms)",
           title="Distribution of latency (notch = 95% CI of the median)")
    save(fig, "fig2_distribution_boxplot")
    return fig


def fig3_recall(q):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [5, 2]})
    for ax, ds, title in [(axes[0], "tweeteval", "(a) TweetEval: our obfuscations"),
                          (axes[1], "noisyhate", "(b) NoisyHate: real human obfuscations")]:
        d = q[q.dataset == ds]
        conds = list(d.condition.unique())
        pipelines = list(d.groupby(["model", "prep"], sort=False).groups)
        for i, (m, prep) in enumerate(pipelines):
            g = d[(d.model == m) & (d.prep == prep)].set_index("condition").loc[conds]
            x = np.arange(len(conds)) + (i - (len(pipelines) - 1) / 2) * 0.15
            ax.errorbar(x, g.recall * 100, yerr=[(g.recall - g.recall_ci_low) * 100, (g.recall_ci_high - g.recall) * 100],
                        color=COLORS[m], marker=MARKER[m], ls=LINE[prep], capsize=3, label=f"{m}, {prep}")
        ax.set_xticks(range(len(conds)), conds, rotation=15)
        ax.set(title=title, xlabel="Text condition", ylabel="Recall on harmful posts (%)")
    fig.tight_layout()
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.7, -0.22), ncol=3)
    save(fig, "fig3_recall_by_obfuscation")
    return fig


if __name__ == "__main__":
    meta, lat, q, mem = load_log()
    ev, _ = drop_warmup(lat)
    fig1_latency_vs_load(latency_table(ev))
    fig2_distribution_boxplot(ev)
    fig3_recall(quality_table(q))
    print("figures saved to", FIG)
