"""Error bars for the size-controlled comparison.

downstream.py pairs each deduplicated training set with one random subsample of the same size.
One draw cannot say whether a 0.3-point gap is a real effect or sampling noise, so this script
redraws the control five times per size and reports the spread. The deduplicated run itself is
deterministic (the selection is fixed and lbfgs converges to the same solution), so only the
control needs repeating.

Usage:  python control_variance.py   (writes results/control_variance.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from downstream import SETTINGS, classifier

RESULTS = Path(__file__).parent / "results"
SEEDS = (101, 202, 303, 404, 505)


def main() -> None:
    ag = load_dataset("fancyzhx/ag_news")
    texts, labels = list(ag["train"]["text"]), np.array(ag["train"]["label"])
    test_texts, test_labels = list(ag["test"]["text"]), np.array(ag["test"]["label"])
    down = json.loads((RESULTS / "downstream.json").read_text())

    out = {"seeds": list(SEEDS), "settings": {}}
    for name in SETTINGS:
        size = down["accuracy"][name]["train_size"]
        scores = []
        for seed in SEEDS:
            keep = np.sort(np.random.default_rng(seed).choice(len(texts), size=size, replace=False))
            model = classifier()
            model.fit([texts[i] for i in keep], labels[keep])
            scores.append(float((model.predict(test_texts) == test_labels).mean()))
        scores = np.array(scores)
        deduped = down["accuracy"][name]["accuracy"]
        # A gap larger than this is unlikely to be a lucky draw of the control.
        margin = 2 * float(scores.std(ddof=1))
        out["settings"][name] = {
            "train_size": size,
            "deduplicated_accuracy": deduped,
            "control_mean": round(float(scores.mean()), 4),
            "control_std": round(float(scores.std(ddof=1)), 5),
            "control_scores": [round(s, 4) for s in scores.tolist()],
            "delta_vs_control_mean": round(deduped - float(scores.mean()), 4),
            "beyond_two_sigma": bool(abs(deduped - float(scores.mean())) > margin),
        }
        row = out["settings"][name]
        print(
            f"{name:13s} n={size:>7,} dedup={deduped:.4f} control={row['control_mean']:.4f}"
            f" +-{row['control_std']:.4f}  delta={row['delta_vs_control_mean']:+.4f}"
            f" {'(beyond 2 sigma)' if row['beyond_two_sigma'] else '(within noise)'}",
            flush=True,
        )
    (RESULTS / "control_variance.json").write_text(json.dumps(out, indent=2))
    print("wrote", RESULTS / "control_variance.json")


if __name__ == "__main__":
    main()
