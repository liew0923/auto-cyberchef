"""Controlled benchmark with known ground truth.

Real datasets have no duplicate labels, so removal counts alone cannot say which method is
*right*. Here we take ag_news records that no method flagged in the main run (a clean base),
inject perturbed copies of some of them, and measure for every method:

* recall    - injected copies that were removed, per perturbation type
* precision - removed records that really were injected copies

Usage:  python synthetic_eval.py   (needs results/masks_ag_news.npz from run_experiment.py)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset

import methods as M
from run_experiment import MINHASH_THRESHOLDS, SEMHASH_THRESHOLDS, SIMHASH_DISTANCES, course_methods, semhash_self

RESULTS = Path(__file__).parent / "results"
N_BASE, N_PER_TYPE, SEED = 20_000, 1_500, 13


def reformat(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Same words, different surface form: case, punctuation and whitespace changes."""
    words = M.TOKEN_RE.findall(text.lower())
    return "  ".join(word.upper() if rng.random() < 0.5 else word for word in words)


def typos(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Corrupt about 3% of the characters (substitute, delete or transpose)."""
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < 0.03:
            op = rng.choice("sdt")
            if op == "s":
                chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            elif op == "d":
                chars[i] = ""
            else:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def drop_words(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Delete about 15% of the words."""
    words = text.split()
    return " ".join(word for word in words if rng.random() >= 0.15) or words[0]


def replace_words(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Replace about 15% of the words with random other words."""
    return " ".join(rng.choice(vocab) if rng.random() < 0.15 else word for word in text.split())


def truncate(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Keep only the first 70% of the words."""
    words = text.split()
    return " ".join(words[: max(1, int(len(words) * 0.7))])


def swap_halves(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Move the second half of the text in front of the first half."""
    words = text.split()
    return " ".join(words[len(words) // 2 :] + words[: len(words) // 2])


PERTURBATIONS = {
    "reformat (case/punct.)": reformat,
    "typos (3% of chars)": typos,
    "drop 15% of words": drop_words,
    "replace 15% of words": replace_words,
    "truncate to 70%": truncate,
    "swap halves": swap_halves,
}


def main() -> None:
    texts = load_dataset("fancyzhx/ag_news")["train"]["text"]
    flagged = np.zeros(len(texts), dtype=bool)
    with np.load(RESULTS / "masks_ag_news.npz") as masks:
        for name in masks.files:
            flagged |= masks[name]
    clean = np.flatnonzero(~flagged)
    rng_np, rng = np.random.default_rng(SEED), random.Random(SEED)
    base = [texts[i] for i in rng_np.choice(clean, size=N_BASE, replace=False)]
    vocab = sorted({word for text in base[:2000] for word in text.split()})

    corpus, kind = list(base), ["base"] * N_BASE
    sources = rng_np.choice(N_BASE, size=N_PER_TYPE * len(PERTURBATIONS), replace=False)
    for t, (name, fn) in enumerate(PERTURBATIONS.items()):
        for src in sources[t * N_PER_TYPE : (t + 1) * N_PER_TYPE]:
            corpus.append(fn(base[src], rng, vocab))
            kind.append(name)
    kind_arr = np.array(kind)
    is_dup = kind_arr != "base"

    masks, _, _ = course_methods(corpus)
    masks.update(semhash_self(corpus)[0])
    wanted = (
        ["exact", "exact_norm"]
        + [f"minhash_{t}" for t in MINHASH_THRESHOLDS]
        + [f"simhash_{k}" for k in SIMHASH_DISTANCES]
        + [f"semhash_{t}" for t in SEMHASH_THRESHOLDS]
    )
    out = {"n_base": N_BASE, "n_injected": int(is_dup.sum()), "perturbations": list(PERTURBATIONS), "methods": {}}
    for method in wanted:
        removed = masks[method]
        tp, fp = int((removed & is_dup).sum()), int((removed & ~is_dup).sum())
        recall = tp / is_dup.sum()
        precision = tp / (tp + fp) if tp + fp else 1.0
        out["methods"][method] = {
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(2 * precision * recall / (precision + recall)) if tp else 0.0, 4),
            "false_positives": fp,
            "recall_by_type": {
                name: round(float(removed[kind_arr == name].mean()), 4) for name in PERTURBATIONS
            },
        }
        row = out["methods"][method]
        print(f"{method:14s} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} FP={fp}")
    (RESULTS / "synthetic.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
