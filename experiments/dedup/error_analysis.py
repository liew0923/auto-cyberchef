"""Where the methods disagree, and who is wrong.

The main run shows SemHash removing far more than any lexical method: 57,454 wikitext lines
that MinHash does not touch, and 18.5% of the ag_news test set as train duplicates. This script
characterises those disagreements instead of assuming the extra removals are all correct.

Measured per dataset, for the records SemHash removes that MinHash does not:

* **length profile** - short lines carry little information, so an embedding of a five-word
  section heading is dominated by its topic. If the disagreement concentrates on short records,
  the extra removals are mostly heading collisions rather than duplicate prose.
* **lexical overlap** - the estimated Jaccard of each removed pair. A pair that shares no
  wording at all is a paraphrase at best and a topic twin at worst.

Usage:  python error_analysis.py   (needs results/masks_*.npz)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

import methods as M

RESULTS = Path(__file__).parent / "results"
BUCKETS = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 10**9)]


def load_texts(name: str) -> list[str]:
    if name == "ag_news":
        return list(load_dataset("fancyzhx/ag_news")["train"]["text"])
    column = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"]
    return [line for line in column if line.strip()]


def profile(texts: list[str], subset: np.ndarray) -> dict:
    lengths = np.array([len(M.TOKEN_RE.findall(texts[i])) for i in subset]) if len(subset) else np.zeros(0)
    return {
        "count": int(len(subset)),
        "median_tokens": float(np.median(lengths)) if len(lengths) else None,
        "share_under_10_tokens": round(float((lengths < 10).mean()), 4) if len(lengths) else None,
        "length_buckets": {
            f"{lo}-{hi if hi < 10**9 else '+'}": int(((lengths >= lo) & (lengths < hi)).sum()) for lo, hi in BUCKETS
        },
    }


def main() -> None:
    out: dict = {}
    for name in ("ag_news", "wikitext"):
        texts = load_texts(name)
        with np.load(RESULTS / f"masks_{name}.npz") as masks:
            semhash, minhash = masks["semhash_0.9"], masks["minhash_0.7"]
        disputed = np.flatnonzero(semhash & ~minhash)
        agreed = np.flatnonzero(semhash & minhash)
        kept = np.flatnonzero(~semhash & ~minhash)

        # Lexical overlap of the disputed removals, estimated from MinHash signatures.
        # Candidate generation is bounded at Jaccard 0.2: a disputed record with no candidate
        # at all shares almost no wording with anything, which is the finding of interest.
        signatures, _ = M.featurize(texts, shingle_size=3, num_perm=128)
        pi, pj, sim = M.minhash_lsh_pairs(signatures, min_similarity=0.2)
        best = np.zeros(len(texts), dtype=np.float32)
        np.maximum.at(best, pi, sim)
        np.maximum.at(best, pj, sim)
        overlaps = best[disputed]
        del signatures, pi, pj, sim

        out[name] = {
            "semhash_0.9_removed": int(semhash.sum()),
            "minhash_0.7_removed": int(minhash.sum()),
            "disputed_semhash_only": profile(texts, disputed),
            "agreed_by_both": profile(texts, agreed),
            "kept_by_both": profile(texts, kept),
            "disputed_lexical_overlap": {
                "records": int(len(disputed)),
                "median_best_jaccard": round(float(np.median(overlaps)), 4),
                "share_below_0.2_jaccard": round(float((overlaps < 0.2).mean()), 4),
                "share_above_0.5": round(float((overlaps >= 0.5).mean()), 4),
            },
        }
        print(name, json.dumps(out[name], indent=2), flush=True)
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=2))
    print("wrote", RESULTS / "error_analysis.json")


if __name__ == "__main__":
    main()
