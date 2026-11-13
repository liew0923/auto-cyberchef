"""Render the report figures from results/results.json and results/synthetic.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

RESULTS = Path(__file__).parent / "results"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

FAMILY = {
    "exact": "#52514e",
    "simhash": BLUE,
    "charminhash": "#6250d6",
    "minhash": BLUE,
    "semhash": ORANGE,
    "hybrid": "#1baf7a",
}


def family_of(method: str) -> str:
    return next((color for prefix, color in FAMILY.items() if method.startswith(prefix)), MUTED)


ROWS = [
    ("exact", "Exact hash"),
    ("exact_norm", "Exact hash, normalised"),
    ("bloom", "Bloom filter, normalised"),
    ("simhash_3", "SimHash, Hamming ≤ 3"),
    ("minhash_0.7", "MinHash-LSH, Jaccard ≥ 0.7"),
    ("charminhash_0.6", "Character MinHash ≥ 0.6"),
    ("semhash_0.9", "SemHash, cosine ≥ 0.9"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def removal_chart(results: dict) -> None:
    titles = {"ag_news": "ag_news train", "wikitext": "wikitext-103 train, non-empty lines"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, titles.items()):
        data = results["datasets"][key]
        rows = [r for r in ROWS if r[0] in data["methods"]]
        values = [data["methods"][m]["removed_pct"] for m, _ in rows]
        colors = [family_of(m) for m, _ in rows]
        positions = range(len(rows))
        ax.barh(positions, values, height=0.55, color=colors)
        for y, value in zip(positions, values):
            ax.text(value + max(values) * 0.015, y, f"{value:.2f}%", va="center", color=INK, fontsize=9)
        ax.set_yticks(positions, [label for _, label in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)  # top-to-bottom; invert_yaxis() on shared axes cancels out
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel("records removed (%)")
        ax.set_title(f"{title}  (n = {data['n_records']:,})", loc="left", fontsize=10.5, color=INK)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.legend(
        handles=[
            Patch(color=BLUE, label="word-level, implemented here"),
            Patch(color="#6250d6", label="character-level, implemented here"),
            Patch(color=ORANGE, label="SemHash baseline"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("Share of records removed as duplicates", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(RESULTS / "removal_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


SHORT_EDIT = {
    "reformat (case/punct.)": "reformat",
    "typos (3% of chars)": "typos\n3% of chars",
    "drop 15% of words": "drop\n15% words",
    "replace 15% of words": "replace\n15% words",
    "truncate to 70%": "truncate\nto 70%",
    "swap halves": "swap\nhalves",
    "wrapped in boilerplate": "boilerplate\nwrap",
}


def recall_heatmap(synthetic: dict) -> None:
    rows = [r for r in ROWS if r[0] != "bloom" and r[0] in synthetic["methods"]]
    rows = rows + [("hybrid_cascade_char", "Cascade (character)")]
    rows = [r for r in rows if r[0] in synthetic["methods"]]
    types = synthetic["perturbations"]
    grid = [[synthetic["methods"][m]["recall_by_type"][t] * 100 for t in types] for m, _ in rows]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    image = ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.0f}", ha="center", va="center", color="white" if value > 55 else INK, fontsize=10)
    ax.set_xticks(range(len(types)), [SHORT_EDIT.get(t, t) for t in types], color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.set_xticks([x - 0.5 for x in range(1, len(types))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("recall (%)")
    bar.outline.set_visible(False)
    ax.set_title(
        "Recall on injected near-duplicates, by kind of edit (%)", loc="left", fontsize=12.5, fontweight="bold", color=INK
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "synthetic_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


LABELLED = {
    "exact_norm": ("exact, normalised", (8, -12)),
    "simhash_3": ("SimHash ≤ 3", (8, 4)),
    "minhash_0.7": ("MinHash 0.7", (-10, -16)),
    "minhash_0.5": ("MinHash 0.5", (-18, -16)),
    "charminhash_0.6": ("character MinHash 0.6", (-30, 10)),
    "semhash_0.9": ("SemHash 0.9", (-78, -4)),
    "semhash_0.85": ("SemHash 0.85", (8, -4)),
    "semhash_0.8": ("SemHash 0.8", (-20, -16)),
    "hybrid_two_signal_0.85": ("two-signal", (-34, -20)),
    "hybrid_cascade_char": ("cascade (character)", (-22, 13)),
}


def precision_recall(synthetic: dict) -> None:
    """Precision against recall on the injected duplicates: the quality frontier."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for method, row in synthetic["methods"].items():
        if method == "exact":
            continue
        x, y = row["recall"] * 100, row["precision_lower_bound"] * 100
        color = family_of(method)
        labelled = method in LABELLED
        ax.scatter(
            x,
            y,
            s=96 if labelled else 34,
            color=color,
            alpha=1.0 if labelled else 0.38,
            zorder=4 if labelled else 3,
            edgecolor=SURFACE,
            linewidth=2 if labelled else 1,
        )
        if labelled:
            text, offset = LABELLED[method]
            ax.annotate(text, (x, y), textcoords="offset points", xytext=offset, fontsize=9, color=INK)
    ax.set_xlabel("recall on injected near-duplicates (%)")
    ax.set_ylabel("precision, lower bound (%)")
    ax.set_xlim(10, 100)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(
        "Quality frontier: up and to the right is better",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.legend(
        handles=[
            Patch(color=BLUE, label="word-level lexical"),
            Patch(color="#6250d6", label="character-level lexical"),
            Patch(color=ORANGE, label="SemHash baseline"),
            Patch(color="#1baf7a", label="hybrid (this work)"),
        ],
        frameon=False,
        loc="lower left",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


SCALING_LABELS = {
    "exact_norm": "exact, normalised",
    "bloom": "Bloom filter",
    "minhash": "MinHash (word)",
    "charminhash": "MinHash (char)",
    "simhash": "SimHash \u2264 3",
    "semhash": "SemHash 0.9",
    "hybrid_cascade": "cascade (word)",
    "hybrid_cascade_char": "cascade (char)",
}


# Methods within a family share a colour, so each gets its own dash pattern and marker.
SCALING_STYLE = {
    "exact_norm": ("-", "o"),
    "bloom": (":", "s"),
    "minhash": ("-", "o"),
    "simhash": (":", "s"),
    "charminhash": ("-", "o"),
    "semhash": ("-", "o"),
    "hybrid_cascade": ("--", "o"),
    "hybrid_cascade_char": ("-.", "s"),
}


def scaling_chart(rows: list) -> None:
    """Runtime and memory against corpus size, log-log."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = [m for m in SCALING_LABELS if any(r["method"] == m for r in rows)]
    for method in order:
        series = sorted((r for r in rows if r["method"] == method), key=lambda r: r["n"])
        xs = [r["n"] for r in series]
        style, marker = SCALING_STYLE[method]
        label = SCALING_LABELS[method]
        for ax, key in ((axes[0], "seconds"), (axes[1], "method_rss_mb")):
            ax.plot(xs, [r[key] for r in series], style, color=family_of(method),
                    marker=marker, ms=4, mfc="none" if marker == "s" else None, label=label)
    for ax, ylabel, title in (
        (axes[0], "wall-clock seconds", "Runtime"),
        (axes[1], "MB above the loaded corpus", "Working memory"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("records")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_yscale("log")
    # Bloom uses a fixed bit array, so its figure is 0 MB: symlog keeps that visible.
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_ylim(0, 8000)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
    fig.suptitle("Scaling on wikitext-103", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS / "scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


LABELS = {
    "exact_norm": "exact, normalised",
    "simhash_3": "SimHash ≤ 3",
    "minhash_0.7": "MinHash ≥ 0.7",
    "semhash_0.9": "SemHash ≥ 0.9",
    "semhash_0.8": "SemHash ≥ 0.8",
}


def downstream_chart(down: dict, variance: dict) -> None:
    """Deduplicated training set against an equal-size random subsample, with error bars."""
    rows = [(k, v) for k, v in variance["settings"].items()]
    labels = [f"{LABELS.get(k, k)}\n{v['train_size']:,} records" for k, v in rows]
    deduped = [v["deduplicated_accuracy"] * 100 for _, v in rows]
    control = [v["control_mean"] * 100 for _, v in rows]
    errors = [2 * v["control_std"] * 100 for _, v in rows]
    baseline = down["accuracy"]["full (no dedup)"]["accuracy"] * 100

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvline(baseline, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.barh(y - 0.2, deduped, height=0.36, color=BLUE, zorder=3, label="deduplicated")
    ax.barh(
        y + 0.2,
        control,
        height=0.36,
        color="#9ec5f4",
        zorder=3,
        xerr=errors,
        error_kw={"ecolor": MUTED, "elinewidth": 1.4, "capsize": 3},
        label="random subsample of the same size (mean ± 2σ, 5 draws)",
    )
    for yi, ((_, row), value, ctrl) in enumerate(zip(rows, deduped, control)):
        mark = "  ←" if row["beyond_two_sigma"] else ""
        ax.text(value + 0.03, yi - 0.2, f"{value:.2f}{mark}", va="center", fontsize=9, color=INK)
        ax.text(ctrl + 0.03 + errors[yi], yi + 0.2, f"{ctrl:.2f}", va="center", fontsize=9, color=MUTED)

    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(control + deduped) - 0.45, max(control + deduped + [baseline]) + 0.45)
    ax.set_xlabel("ag_news test accuracy (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.annotate(
        f"full corpus, no deduplication: {baseline:.2f}%",
        xy=(baseline, len(rows) - 0.45),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=MUTED,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1)
    ax.set_title(
        "Deduplication pays only when the corpus is already being cut",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "downstream.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    written = []
    if (RESULTS / "results.json").exists():
        removal_chart(json.loads((RESULTS / "results.json").read_text()))
        written.append("removal_rate.png")
    if (RESULTS / "synthetic.json").exists():
        synthetic = json.loads((RESULTS / "synthetic.json").read_text())
        recall_heatmap(synthetic)
        precision_recall(synthetic)
        written += ["synthetic_recall.png", "precision_recall.png"]
    if (RESULTS / "scaling.json").exists():
        scaling_chart(json.loads((RESULTS / "scaling.json").read_text()))
        written.append("scaling.png")
    if (RESULTS / "downstream.json").exists() and (RESULTS / "control_variance.json").exists():
        downstream_chart(
            json.loads((RESULTS / "downstream.json").read_text()),
            json.loads((RESULTS / "control_variance.json").read_text()),
        )
        written.append("downstream.png")
    print("figures written to", RESULTS, written)
