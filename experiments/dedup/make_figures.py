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


FAMILY = {"exact": "#52514e", "simhash": BLUE, "minhash": BLUE, "semhash": ORANGE, "hybrid": "#1baf7a"}


def family_of(method: str) -> str:
    return next((color for prefix, color in FAMILY.items() if method.startswith(prefix)), MUTED)


def precision_recall(synthetic: dict) -> None:
    """Precision vs recall on the injected duplicates: the quality frontier."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for method, row in synthetic["methods"].items():
        if method == "exact":
            continue
        color = family_of(method)
        ax.scatter(row["recall"] * 100, row["precision"] * 100, s=70, color=color, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            method.replace("hybrid_", "hybrid "),
            (row["recall"] * 100, row["precision"] * 100),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8.5,
            color=INK,
        )
    ax.set_xlabel("recall on injected near-duplicates (%)")
    ax.set_ylabel("precision (%)")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Quality frontier: precision vs recall", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    handles = [
        Patch(color=BLUE, label="lexical (MinHash / SimHash)"),
        Patch(color=ORANGE, label="semantic (SemHash)"),
        Patch(color="#1baf7a", label="hybrid (this work)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def scaling_chart(rows: list) -> None:
    """Throughput against corpus size, log-log."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    methods = sorted({row["method"] for row in rows})
    styles = {m: ("-" if not m.startswith("hybrid") else "--") for m in methods}
    for method in methods:
        series = sorted((r for r in rows if r["method"] == method), key=lambda r: r["n"])
        xs = [r["n"] for r in series]
        axes[0].plot(xs, [r["seconds"] for r in series], styles[method], color=family_of(method), marker="o", ms=4, label=method)
        axes[1].plot(xs, [r["peak_rss_mb"] for r in series], styles[method], color=family_of(method), marker="o", ms=4, label=method)
    for ax, ylabel, title in (
        (axes[0], "wall-clock seconds", "Runtime"),
        (axes[1], "peak RSS (MB)", "Memory"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("records")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Scaling on wikitext-103", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def downstream_chart(down: dict) -> None:
    """Deduplicated training set vs an equal-size random subsample."""
    rows = [(k, v) for k, v in down["accuracy"].items() if k != "full (no dedup)"]
    labels = [k for k, _ in rows]
    deduped = [v["accuracy"] * 100 for _, v in rows]
    control = [v["random_subsample_accuracy"] * 100 for _, v in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.barh(y - 0.19, deduped, height=0.34, color=BLUE, label="deduplicated")
    ax.barh(y + 0.19, control, height=0.34, color="#9ec5f4", label="random subsample, same size")
    baseline = down["accuracy"]["full (no dedup)"]["accuracy"] * 100
    ax.axvline(baseline, color=ORANGE, linewidth=2, zorder=4)
    ax.text(baseline, -0.85, f" full corpus {baseline:.2f}%", color=ORANGE, fontsize=9, va="center")
    for yi, (value, ctrl) in enumerate(zip(deduped, control)):
        ax.text(value + 0.05, yi - 0.19, f"{value:.2f}", va="center", fontsize=8.5, color=INK)
        ax.text(ctrl + 0.05, yi + 0.19, f"{ctrl:.2f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    low = min(deduped + control + [baseline])
    ax.set_xlim(low - 1.2, max(deduped + control + [baseline]) + 0.8)
    ax.set_xlabel("ag_news test accuracy (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(
        "Downstream accuracy after deduplication, size-controlled", loc="left", fontsize=12.5, fontweight="bold", color=INK
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
    if (RESULTS / "downstream.json").exists():
        downstream_chart(json.loads((RESULTS / "downstream.json").read_text()))
        written.append("downstream.png")
    print("figures written to", RESULTS, written)
