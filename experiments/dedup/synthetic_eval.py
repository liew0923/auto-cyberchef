"""Controlled benchmark with known ground truth.

Real datasets carry no duplicate labels, so removal counts alone cannot say which method is
*right*. Here a base sample of ag_news is taken, perturbed copies of some of its records are
injected, and every method is scored against that known answer key.

Three numbers per method:

* **recall** - share of injected copies removed, broken down by kind of edit.
* **precision (lower bound)** - of everything removed, the share that was an injected copy.
  The base is a natural sample, so some of the remaining removals may be genuine duplicates
  that ag_news already contained; precision is therefore pessimistic by construction and is
  reported as a lower bound rather than as a verdict.
* **label disagreement among base removals** - independent evidence of real mistakes. ag_news
  is labelled and a true duplicate of a World-news item is also World news, so a removed pair
  whose two labels differ is a near-certain error rather than a missed natural duplicate.

The base deliberately keeps natural near-duplicates (only exact duplicates are dropped). An
earlier version of this script built the base from records that no method had flagged, which
removed every hard case and awarded all methods a precision of exactly 1.0.

Usage:  python synthetic_eval.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset

import hybrid
import methods as M
from run_experiment import (
    CHAR_THRESHOLDS,
    MINHASH_THRESHOLDS,
    SEMHASH_THRESHOLDS,
    SIMHASH_DISTANCES,
    course_methods,
    semhash_self,
)

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


BOILERPLATE = (
    "Advertisement. Subscribe to our newsletter for daily updates.",
    "Reuters - All rights reserved. Click here for the full story.",
    "Share this article on social media. Comments are now closed.",
)


def boilerplate(text: str, rng: random.Random, vocab: list[str]) -> str:
    """Wrap the text in web boilerplate, as a scraped copy of the same article would be."""
    return f"{rng.choice(BOILERPLATE)} {text} {rng.choice(BOILERPLATE)}"


PERTURBATIONS = {
    "reformat (case/punct.)": reformat,
    "typos (3% of chars)": typos,
    "drop 15% of words": drop_words,
    "replace 15% of words": replace_words,
    "truncate to 70%": truncate,
    "swap halves": swap_halves,
    "wrapped in boilerplate": boilerplate,
}


def main() -> None:
    ag = load_dataset("fancyzhx/ag_news")["train"]
    texts, labels = list(ag["text"]), np.array(ag["label"])
    # Natural base: a random sample with only exact duplicates dropped, so hard cases survive.
    eligible = np.flatnonzero(M.exact_dedup(texts, normalized=True) < 0)
    rng_np, rng = np.random.default_rng(SEED), random.Random(SEED)
    chosen = rng_np.choice(eligible, size=N_BASE, replace=False)
    base, base_labels = [texts[i] for i in chosen], labels[chosen]
    vocab = sorted({word for text in base[:2000] for word in text.split()})

    corpus, kind, source_of = list(base), ["base"] * N_BASE, list(range(N_BASE))
    sources = rng_np.choice(N_BASE, size=N_PER_TYPE * len(PERTURBATIONS), replace=False)
    for t, (name, fn) in enumerate(PERTURBATIONS.items()):
        for src in sources[t * N_PER_TYPE : (t + 1) * N_PER_TYPE]:
            corpus.append(fn(base[src], rng, vocab))
            kind.append(name)
            source_of.append(int(src))
    kind_arr, source_of = np.array(kind), np.array(source_of)
    is_dup = kind_arr != "base"
    corpus_labels = base_labels[source_of]  # an injected copy inherits its source's label

    masks, reps, secs = course_methods(corpus)
    sem_masks, sem_reps, _ = semhash_self(corpus)
    masks.update(sem_masks)
    reps.update(sem_reps)
    for name, result in (
        ("hybrid_cascade", hybrid.cascade(corpus)),
        ("hybrid_two_signal", hybrid.two_signal(corpus)),
        ("hybrid_two_signal_0.85", hybrid.two_signal(corpus, cosine=0.85)),
    ):
        masks[name], reps[name] = result.removed, result.rep
        secs[name] = result.seconds
    wanted = (
        ["exact", "exact_norm"]
        + [f"minhash_{t}" for t in MINHASH_THRESHOLDS]
        + [f"charminhash_{t}" for t in CHAR_THRESHOLDS]
        + [f"simhash_{k}" for k in SIMHASH_DISTANCES]
        + [f"semhash_{t}" for t in SEMHASH_THRESHOLDS]
        + ["hybrid_cascade", "hybrid_two_signal", "hybrid_two_signal_0.85"]
    )
    out = {
        "n_base": N_BASE,
        "n_injected": int(is_dup.sum()),
        "perturbations": list(PERTURBATIONS),
        "note": "precision is a lower bound: the base is natural ag_news text and may contain duplicates of its own",
        "methods": {},
    }
    for method in wanted:
        removed, rep = masks[method], reps[method]
        tp, fp = int((removed & is_dup).sum()), int((removed & ~is_dup).sum())
        recall = tp / is_dup.sum()
        precision = tp / (tp + fp) if tp + fp else 1.0
        # Independent error evidence: a removed pair whose ag_news labels differ is a near-certain mistake.
        dropped = np.flatnonzero(rep >= 0)
        cross_label = int((corpus_labels[dropped] != corpus_labels[rep[dropped]]).sum())
        out["methods"][method] = {
            "recall": round(float(recall), 4),
            "precision_lower_bound": round(float(precision), 4),
            "f1_lower_bound": round(float(2 * precision * recall / (precision + recall)) if tp else 0.0, 4),
            "removals_outside_the_answer_key": fp,
            "cross_label_removals": cross_label,
            "cross_label_pct_of_removals": round(100 * cross_label / max(len(dropped), 1), 2),
            "recall_by_type": {
                name: round(float(removed[kind_arr == name].mean()), 4) for name in PERTURBATIONS
            },
            "seconds": secs.get(method),
        }
        row = out["methods"][method]
        print(
            f"{method:22s} R={row['recall']:.3f} P>={row['precision_lower_bound']:.3f}"
            f" F1>={row['f1_lower_bound']:.3f} extra={fp:>5} cross-label={cross_label:>4}"
            f" ({row['cross_label_pct_of_removals']:.1f}%)",
            flush=True,
        )
    (RESULTS / "synthetic.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
