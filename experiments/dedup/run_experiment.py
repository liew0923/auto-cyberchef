"""Experiment 1: deduplicate ag_news and wikitext, and compare against SemHash.

Usage:  python run_experiment.py            (writes results/results.json and results/masks_*.npz)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from semhash import SemHash

import methods as M

RESULTS = Path(__file__).parent / "results"
MINHASH_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
SIMHASH_DISTANCES = (0, 1, 2, 3)
SEMHASH_THRESHOLDS = (0.8, 0.85, 0.9, 0.95)
MAIN = {"minhash": 0.7, "simhash": 3, "semhash": 0.9}


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = round(time.perf_counter() - self.start, 2)


def expand(rep_unique: np.ndarray, unique_idx: np.ndarray, exact_rep: np.ndarray) -> np.ndarray:
    """Map a result computed on exact-unique records back to all records (exact copies stay removed)."""
    rep = exact_rep.copy()
    hit = rep_unique >= 0
    rep[unique_idx[hit]] = unique_idx[rep_unique[hit]]
    return rep


def course_methods(texts: list[str]) -> tuple[dict, dict, dict]:
    """Run every from-scratch technique. Returns (removed masks, rep arrays, timings in seconds)."""
    masks, reps, secs = {}, {}, {}

    with Timer() as t:
        exact_rep = M.exact_dedup(texts)
    reps["exact"], secs["exact"] = exact_rep, t.seconds
    with Timer() as t:
        reps["exact_norm"] = M.exact_dedup(texts, normalized=True)
    secs["exact_norm"] = t.seconds
    with Timer() as t:
        masks["bloom"], bloom = M.bloom_dedup(texts)
    secs["bloom"] = t.seconds
    secs["bloom_bytes"] = len(bloom.bits)

    # Near-duplicate search runs on the exact-unique records, like SemHash does.
    unique_idx = np.flatnonzero(exact_rep < 0)
    unique = [texts[i] for i in unique_idx]

    with Timer() as t:
        signatures, _ = M.featurize(unique, shingle_size=3, num_perm=128)
        pi, pj, sim = M.minhash_lsh_pairs(signatures, min_similarity=min(MINHASH_THRESHOLDS))
    secs["minhash_index"] = t.seconds
    del signatures
    for th in MINHASH_THRESHOLDS:
        with Timer() as t:
            ok = sim >= th
            reps[f"minhash_{th}"] = expand(M.greedy_select(len(unique), pi[ok], pj[ok]), unique_idx, exact_rep)
        secs[f"minhash_{th}"] = round(secs["exact"] + secs["minhash_index"] + t.seconds, 2)

    with Timer() as t:
        _, fingerprints = M.featurize(unique, shingle_size=1, num_perm=0)
        qi, qj, dist = M.simhash_pairs(fingerprints, max_distance=max(SIMHASH_DISTANCES))
    secs["simhash_index"] = t.seconds
    for k in SIMHASH_DISTANCES:
        with Timer() as t:
            ok = dist <= k
            reps[f"simhash_{k}"] = expand(M.greedy_select(len(unique), qi[ok], qj[ok]), unique_idx, exact_rep)
        secs[f"simhash_{k}"] = round(secs["exact"] + secs["simhash_index"] + t.seconds, 2)

    masks.update({name: rep >= 0 for name, rep in reps.items()})
    return masks, reps, secs


def semhash_self(texts: list[str]) -> tuple[dict, dict, dict]:
    """SemHash self-deduplication at every threshold. Returns (removed masks, rep arrays, timings)."""
    records = [{"id": i, "text": text} for i, text in enumerate(texts)]
    masks, reps, secs = {}, {}, {}
    with Timer() as t:
        index = SemHash.from_records(records, columns=["text"])
    secs["semhash_index"] = t.seconds

    def collect(result: object, th: float, seconds: float) -> None:
        rep = np.full(len(texts), -1, dtype=np.int64)
        for dup in result.filtered:
            rep[dup.record["id"]] = dup.duplicate_of["id"]
        reps[f"semhash_{th}"], masks[f"semhash_{th}"] = rep, rep >= 0
        secs[f"semhash_{th}"] = round(secs["semhash_index"] + seconds, 2)

    # One index query per pair of thresholds; the higher one reuses the neighbours (SemHash's rethreshold).
    for low, high in ((0.9, 0.95), (0.8, 0.85)):
        with Timer() as t:
            result = index.self_deduplicate(threshold=low)
        collect(result, low, t.seconds)
        with Timer() as t2:
            result.rethreshold(high)
        collect(result, high, t.seconds + t2.seconds)
    return masks, reps, secs


def summarise(name: str, texts: list[str], masks: dict, secs: dict) -> dict:
    n = len(texts)
    out = {"dataset": name, "n_records": n, "methods": {}, "timing_detail": secs}
    for method, mask in masks.items():
        out["methods"][method] = {
            "removed": int(mask.sum()),
            "kept": int(n - mask.sum()),
            "removed_pct": round(100 * float(mask.mean()), 3),
            "seconds": secs.get(method),
        }
    truth = masks["exact_norm"]
    out["bloom_false_positives"] = int((masks["bloom"] & ~truth).sum())
    return out


def examples(texts: list[str], reps: dict, a: str, b: str, k: int = 4, seed: int = 0) -> list[dict]:
    """Sample pairs that method ``a`` removes but method ``b`` keeps."""
    only = np.flatnonzero((reps[a] >= 0) & (reps[b] < 0) & (reps["exact_norm"] < 0))
    picks = np.random.default_rng(seed).choice(only, size=min(k, len(only)), replace=False)
    return [{"removed": texts[j][:300], "duplicate_of": texts[reps[a][j]][:300]} for j in picks]


def run_dataset(name: str, texts: list[str]) -> dict:
    print(f"== {name}: {len(texts):,} records", flush=True)
    masks, reps, secs = course_methods(texts)
    print("   course methods done", flush=True)
    sem_masks, sem_reps, sem_secs = semhash_self(texts)
    masks.update(sem_masks), reps.update(sem_reps), secs.update(sem_secs)
    summary = summarise(name, texts, masks, secs)

    mh, sh, sem = (f"minhash_{MAIN['minhash']}", f"simhash_{MAIN['simhash']}", f"semhash_{MAIN['semhash']}")
    summary["overlap"] = {
        f"{a}|{b}": {
            "both": int((masks[a] & masks[b]).sum()),
            "only_first": int((masks[a] & ~masks[b]).sum()),
            "only_second": int((~masks[a] & masks[b]).sum()),
        }
        for a, b in ((mh, sem), (sh, sem), (mh, sh), ("exact_norm", sem))
    }
    summary["examples"] = {
        "semhash_only": examples(texts, reps, sem, mh),
        "minhash_only": examples(texts, reps, mh, sem),
    }
    np.savez_compressed(RESULTS / f"masks_{name}.npz", **masks)
    for method, row in summary["methods"].items():
        print(f"   {method:14s} removed {row['removed']:>9,} ({row['removed_pct']:6.2f}%)  {row['seconds']}s", flush=True)
    return summary


def cross_split(train: list[str], test: list[str]) -> dict:
    """Train/test leakage on ag_news: which test records duplicate a training record?"""
    n_train, texts = len(train), train + test
    out = {"n_train": n_train, "n_test": len(test), "methods": {}}

    def report(method: str, flagged: np.ndarray) -> None:
        out["methods"][method] = {"flagged": int(flagged.sum()), "flagged_pct": round(100 * float(flagged.mean()), 3)}

    def flags(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
        cross = (pi < n_train) & (pj >= n_train)
        flagged = np.zeros(len(test), dtype=bool)
        flagged[pj[cross] - n_train] = True
        return flagged

    for key, normalized in (("exact", False), ("exact_norm", True)):
        seen = {M.normalize(t) if normalized else t for t in train}
        report(key, np.array([(M.normalize(t) if normalized else t) in seen for t in test]))

    signatures, _ = M.featurize(texts, shingle_size=3, num_perm=128)
    pi, pj, sim = M.minhash_lsh_pairs(signatures, min_similarity=min(MINHASH_THRESHOLDS))
    for th in MINHASH_THRESHOLDS:
        report(f"minhash_{th}", flags(pi[sim >= th], pj[sim >= th]))
    _, fingerprints = M.featurize(texts, shingle_size=1, num_perm=0)
    qi, qj, dist = M.simhash_pairs(fingerprints)
    for k in SIMHASH_DISTANCES:
        report(f"simhash_{k}", flags(qi[dist <= k], qj[dist <= k]))

    index = SemHash.from_records(train)
    for th in SEMHASH_THRESHOLDS:
        result = index.deduplicate(test, threshold=th)
        report(f"semhash_{th}", np.arange(len(test)) < len(result.filtered))
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ag = load_dataset("fancyzhx/ag_news")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"]
    wiki_lines = [line for line in wiki if line.strip()]
    results = {
        "main_settings": MAIN,
        "wikitext_raw_lines": len(wiki),
        "wikitext_empty_lines": len(wiki) - len(wiki_lines),
        "datasets": {
            "ag_news": run_dataset("ag_news", ag["train"]["text"]),
            "wikitext": run_dataset("wikitext", wiki_lines),
        },
    }
    print("== ag_news train/test leakage", flush=True)
    results["ag_news_cross_split"] = cross_split(ag["train"]["text"], ag["test"]["text"])
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", RESULTS / "results.json")


if __name__ == "__main__":
    main()
