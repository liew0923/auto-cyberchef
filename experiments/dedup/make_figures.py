"""Render the report figures from results/results.json and results/synthetic.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

RESULTS = Path(__file__).parent / "results"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

ROWS = [
    ("exact", "Exact hash"),
    ("exact_norm", "Exact hash, normalised"),
    ("bloom", "Bloom filter, normalised"),
    ("simhash_3", "SimHash, Hamming ≤ 3"),
    ("minhash_0.7", "MinHash-LSH, Jaccard ≥ 0.7"),
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
        values = [data["methods"][m]["removed_pct"] for m, _ in ROWS]
        colors = [ORANGE if m.startswith("semhash") else BLUE for m, _ in ROWS]
        positions = range(len(ROWS))
        ax.barh(positions, values, height=0.55, color=colors)
        for y, value in zip(positions, values):
            ax.text(value + max(values) * 0.015, y, f"{value:.2f}%", va="center", color=INK, fontsize=9)
        ax.set_yticks(positions, [label for _, label in ROWS])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel("records removed (%)")
        ax.set_title(f"{title}  (n = {data['n_records']:,})", loc="left", fontsize=10.5, color=INK)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.legend(
        handles=[Patch(color=BLUE, label="implemented in this experiment"), Patch(color=ORANGE, label="SemHash baseline")],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("Share of records removed as duplicates", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(RESULTS / "removal_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def recall_heatmap(synthetic: dict) -> None:
    rows = [r for r in ROWS if r[0] != "bloom"]
    types = synthetic["perturbations"]
    grid = [[synthetic["methods"][m]["recall_by_type"][t] * 100 for t in types] for m, _ in rows]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    image = ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.0f}", ha="center", va="center", color="white" if value > 55 else INK, fontsize=10)
    ax.set_xticks(range(len(types)), [t.replace(" (", "\n(") for t in types], color=INK)
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


if __name__ == "__main__":
    removal_chart(json.loads((RESULTS / "results.json").read_text()))
    recall_heatmap(json.loads((RESULTS / "synthetic.json").read_text()))
    print("figures written to", RESULTS)
