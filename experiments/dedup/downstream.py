"""Does deduplication actually help? Downstream impact on ag_news topic classification.

Removal counts say nothing about whether a method is *right*. ag_news is labelled, so this
script measures three things that do:

1. **Label agreement of removed pairs.** A true duplicate of a World-news story is also World
   news. The share of removed pairs whose labels disagree is a label-free precision proxy that
   works on the real corpus, not only on injected duplicates.
2. **Train/test leakage.** Test records that duplicate a training record are scored on material
   the model memorised. We report accuracy on the leaked slice vs. the rest.
3. **Accuracy after deduplication, size-controlled.** Deduplication shrinks the training set, so
   comparing against the full corpus confounds data quality with data quantity. Every
   deduplicated run is therefore paired with a random subsample of exactly the same size.

Usage:  python downstream.py   (writes results/downstream.json)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from semhash import SemHash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import methods as M
from run_experiment import course_methods, semhash_self

RESULTS = Path(__file__).parent / "results"
SEED = 7
SETTINGS = ["exact_norm", "simhash_3", "minhash_0.7", "semhash_0.9", "semhash_0.8"]


def classifier() -> object:
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, strip_accents="unicode"),
        LogisticRegression(max_iter=1000, C=4.0, n_jobs=-1),
    )


def fit_score(
    texts: list[str], labels: np.ndarray, keep: np.ndarray, test_texts: list[str], test_labels: np.ndarray
) -> dict:
    """Train on the kept rows and score on the full test set."""
    model = classifier()
    start = time.perf_counter()
    model.fit([texts[i] for i in keep], labels[keep])
    predicted = model.predict(test_texts)
    return {
        "train_size": int(len(keep)),
        "accuracy": round(float((predicted == test_labels).mean()), 4),
        "fit_seconds": round(time.perf_counter() - start, 1),
        "_predicted": predicted,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ag = load_dataset("fancyzhx/ag_news")
    texts, labels = ag["train"]["text"], np.array(ag["train"]["label"])
    test_texts, test_labels = ag["test"]["text"], np.array(ag["test"]["label"])
    n = len(texts)
    rng = np.random.default_rng(SEED)

    print("recomputing duplicate assignments on ag_news ...", flush=True)
    _, reps, _ = course_methods(texts)
    reps.update(semhash_self(texts)[1])

    out: dict = {"n_train": n, "n_test": len(test_texts), "label_agreement": {}, "accuracy": {}, "leakage": {}}

    # --- 1. label agreement among removed pairs -----------------------------------------
    for name in SETTINGS:
        rep = reps[name]
        removed = np.flatnonzero(rep >= 0)
        same = labels[removed] == labels[rep[removed]]
        out["label_agreement"][name] = {
            "pairs": int(len(removed)),
            "same_label": int(same.sum()),
            "same_label_pct": round(100 * float(same.mean()), 2) if len(removed) else None,
        }
        print(f"   {name:13s} label agreement {out['label_agreement'][name]['same_label_pct']}%", flush=True)

    # Random pairs give the floor: ag_news has four roughly balanced classes, so ~25%.
    pairs = rng.integers(0, n, size=(20000, 2))
    out["label_agreement"]["random_pairs_baseline"] = {
        "same_label_pct": round(100 * float((labels[pairs[:, 0]] == labels[pairs[:, 1]]).mean()), 2)
    }

    # --- 2. baseline model and leakage --------------------------------------------------
    print("training baseline on the full training set ...", flush=True)
    full = fit_score(texts, labels, np.arange(n), test_texts, test_labels)
    baseline_pred = full.pop("_predicted")
    out["accuracy"]["full (no dedup)"] = full
    print(f"   full corpus accuracy {full['accuracy']}", flush=True)

    # Which test records duplicate a training record? Use the strictest and the loosest method.
    combined = texts + test_texts
    signatures, _ = M.featurize(combined, shingle_size=3, num_perm=128)
    pi, pj, sim = M.minhash_lsh_pairs(signatures, min_similarity=0.7)
    del signatures
    cross = (pi < n) & (pj >= n)
    leaked_minhash = np.zeros(len(test_texts), dtype=bool)
    leaked_minhash[pj[cross] - n] = True

    index = SemHash.from_records(texts)
    leaked_semhash = np.zeros(len(test_texts), dtype=bool)
    semhash_result = index.deduplicate(test_texts, threshold=0.9)
    leaked_lookup = {dup.record if isinstance(dup.record, str) else dup.record["text"] for dup in semhash_result.filtered}
    for i, text in enumerate(test_texts):
        if text in leaked_lookup:
            leaked_semhash[i] = True

    for name, leaked in (("minhash_0.7", leaked_minhash), ("semhash_0.9", leaked_semhash)):
        correct = baseline_pred == test_labels
        out["leakage"][name] = {
            "leaked_test_records": int(leaked.sum()),
            "leaked_pct": round(100 * float(leaked.mean()), 2),
            "accuracy_on_leaked": round(float(correct[leaked].mean()), 4) if leaked.any() else None,
            "accuracy_on_clean": round(float(correct[~leaked].mean()), 4),
        }
        print(f"   leakage[{name}] {out['leakage'][name]}", flush=True)

    # --- 3. accuracy after dedup, against an equal-size random subsample ----------------
    for name in SETTINGS:
        keep = np.flatnonzero(reps[name] < 0)
        scored = fit_score(texts, labels, keep, test_texts, test_labels)
        scored.pop("_predicted")
        control = fit_score(texts, labels, np.sort(rng.choice(n, size=len(keep), replace=False)), test_texts, test_labels)
        control.pop("_predicted")
        scored["random_subsample_accuracy"] = control["accuracy"]
        scored["delta_vs_random_subsample"] = round(scored["accuracy"] - control["accuracy"], 4)
        scored["delta_vs_full"] = round(scored["accuracy"] - full["accuracy"], 4)
        out["accuracy"][name] = scored
        print(f"   {name:13s} acc {scored['accuracy']} (random same-size {control['accuracy']})", flush=True)

    (RESULTS / "downstream.json").write_text(json.dumps(out, indent=2))
    print("wrote", RESULTS / "downstream.json")


if __name__ == "__main__":
    main()
