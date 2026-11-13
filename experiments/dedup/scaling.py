"""How each method scales: wall-clock time and peak memory against corpus size.

Quality without cost is half an answer: a method that is 1% better and 20x slower is not
better for a pretraining pipeline. Sizes are drawn from wikitext-103 (long, varied lines) and
each method runs in a separate process so peak RSS is attributed correctly.

Usage:  python scaling.py            (writes results/scaling.json)
        python scaling.py <method> <n>   (internal: one measurement)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
SIZES = (10_000, 50_000, 200_000, 600_000, 1_165_029)
METHODS = (
    "exact_norm",
    "bloom",
    "minhash",
    "charminhash",
    "simhash",
    "semhash",
    "hybrid_cascade",
    "hybrid_cascade_char",
)


PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 1024 / 1024


def current_rss_mb() -> float:
    """Resident memory right now, in MB (field 2 of /proc/self/statm is resident pages)."""
    with open("/proc/self/statm", "rb") as handle:
        return int(handle.read().split()[1]) * PAGE_MB


class RssSampler:
    """Track peak resident memory while a block runs.

    ``resource.getrusage`` reports a high-water mark that the corpus load has already set, so
    it cannot see what a method allocates on top. Sampling the live value does.
    """

    def __init__(self, interval: float = 0.02) -> None:
        self.interval, self.peak, self._stop = interval, 0.0, threading.Event()

    def __enter__(self) -> "RssSampler":
        self.baseline = self.peak = current_rss_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, current_rss_mb())

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak = max(self.peak, current_rss_mb())

    @property
    def used_mb(self) -> float:
        return round(max(self.peak - self.baseline, 0.0), 1)


def corpus(n: int) -> list[str]:
    from datasets import load_dataset

    column = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"]
    lines = [line for line in column if line.strip()]
    return lines if n >= len(lines) else lines[:n]


def measure(method: str, n: int) -> dict:
    """Run one method over n records and report seconds, peak RSS and removal count."""
    import hybrid
    import methods as M

    texts = corpus(n)
    sampler = RssSampler()
    sampler.__enter__()
    start = time.perf_counter()

    if method == "exact_norm":
        removed = int((M.exact_dedup(texts, normalized=True) >= 0).sum())
    elif method == "bloom":
        removed = int(M.bloom_dedup(texts)[0].sum())
    elif method == "minhash":
        signatures, _ = M.featurize(texts, shingle_size=3, num_perm=128)
        pi, pj, sim = M.minhash_lsh_pairs(signatures, min_similarity=0.7)
        removed = int((M.greedy_select(len(texts), pi, pj) >= 0).sum())
    elif method == "charminhash":
        signatures, _ = M.featurize(texts, shingle_size=5, num_perm=128, unit="char")
        pi, pj, sim = M.minhash_lsh_pairs(signatures, min_similarity=0.6)
        removed = int((M.greedy_select(len(texts), pi, pj) >= 0).sum())
    elif method == "simhash":
        _, fingerprints = M.featurize(texts, shingle_size=1, num_perm=0)
        qi, qj, _ = M.simhash_pairs(fingerprints, max_distance=3)
        removed = int((M.greedy_select(len(texts), qi, qj) >= 0).sum())
    elif method == "semhash":
        from semhash import SemHash

        removed = len(SemHash.from_records(texts).self_deduplicate(threshold=0.9).filtered)
    elif method == "hybrid_cascade":
        removed = int(hybrid.cascade(texts).removed.sum())
    elif method == "hybrid_cascade_char":
        removed = int(hybrid.cascade(texts, jaccard=0.6, unit="char").removed.sum())
    else:
        raise SystemExit(f"unknown method {method}")

    elapsed = time.perf_counter() - start
    sampler.__exit__()
    return {
        "method": method,
        "n": len(texts),
        "seconds": round(elapsed, 2),
        "corpus_rss_mb": round(sampler.baseline, 1),
        "method_rss_mb": sampler.used_mb,
        "removed": removed,
        "records_per_second": int(len(texts) / max(elapsed, 1e-9)),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for n in SIZES:
        for method in METHODS:
            proc = subprocess.run(
                [sys.executable, __file__, method, str(n)],
                capture_output=True,
                text=True,
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if proc.returncode != 0 or not line.startswith("{"):
                print(f"   {method} @ {n}: FAILED\n{proc.stderr[-800:]}", flush=True)
                continue
            row = json.loads(line)
            rows.append(row)
            print(
                f"   {row['method']:18s} n={row['n']:>9,}  {row['seconds']:>8.2f}s"
                f"  {row['records_per_second']:>8,} rec/s  {row['method_rss_mb']:>7.0f} MB over corpus",
                flush=True,
            )
            (RESULTS / "scaling.json").write_text(json.dumps(rows, indent=2))
    print("wrote", RESULTS / "scaling.json")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(measure(sys.argv[1], int(sys.argv[2]))))
    else:
        main()
