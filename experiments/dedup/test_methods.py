"""Correctness tests for the from-scratch deduplication primitives.

Each probabilistic structure is checked against the exact quantity it approximates:
MinHash against the true Jaccard similarity, the LSH index and the SimHash block index
against brute-force search, and the Bloom filter against a Python set.

Run:  python -m pytest test_methods.py -q
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import methods as M


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    """Documents with controlled overlap: each base text plus edited copies of it."""
    rng = random.Random(0)
    vocab = [f"word{i}" for i in range(400)]
    texts = []
    for _ in range(300):
        base = " ".join(rng.choice(vocab) for _ in range(rng.randint(5, 60)))
        texts.append(base)
        words = base.split()
        for fraction in (0.05, 0.3):
            edited = [rng.choice(vocab) if rng.random() < fraction else word for word in words]
            texts.append(" ".join(edited))
    texts += ["", "   ", "a", "a b"]  # degenerate inputs
    return texts


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = M.TOKEN_RE.findall(text.lower())
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / len(sa | sb)


def test_exact_dedup_matches_a_dict(corpus: list[str]) -> None:
    rep = M.exact_dedup(corpus)
    first: dict[str, int] = {}
    for j, text in enumerate(corpus):
        if text in first:
            assert rep[j] == first[text], "a repeated text must point at its first occurrence"
        else:
            first[text] = j
            assert rep[j] == -1, "a first occurrence is always kept"


def test_normalised_dedup_ignores_case_and_punctuation() -> None:
    texts = ["Hello, World!", "hello world", "  HELLO   world  ", "goodbye world"]
    rep = M.exact_dedup(texts, normalized=True)
    assert rep.tolist() == [-1, 0, 0, -1]


def test_bloom_filter_has_no_false_negatives(corpus: list[str]) -> None:
    """A Bloom filter may over-report, never under-report."""
    removed, _ = M.bloom_dedup(corpus, fp_rate=1e-3)
    truth = M.exact_dedup(corpus, normalized=True) >= 0
    assert np.all(removed[truth]), "every true duplicate must be caught"
    false_positive_rate = (removed & ~truth).sum() / max((~truth).sum(), 1)
    assert false_positive_rate < 0.01


def test_minhash_estimates_true_jaccard(corpus: list[str]) -> None:
    signatures, _ = M.featurize(corpus, shingle_size=3, num_perm=256, seed=1)
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(corpus), size=(400, 2))
    errors = []
    for i, j in pairs:
        if i == j:
            continue
        estimate = float((signatures[i] == signatures[j]).mean())
        errors.append(abs(estimate - jaccard(corpus[i], corpus[j])))
    # Standard error of a 256-permutation estimate is at most 1/(2*sqrt(256)) = 0.031.
    assert np.mean(errors) < 0.04, f"mean absolute error too high: {np.mean(errors):.3f}"
    assert np.max(errors) < 0.25


def test_shingling_handles_short_and_empty_texts() -> None:
    signatures, fingerprints = M.featurize(["", "one", "one two", "one two three"], shingle_size=3)
    assert signatures.shape[0] == 4 and fingerprints.shape[0] == 4
    assert (signatures[1] != signatures[2]).any(), "different short texts get different signatures"


def test_lsh_recall_against_brute_force(corpus: list[str]) -> None:
    """LSH is allowed to miss low-similarity pairs, but must find the high-similarity ones."""
    signatures, _ = M.featurize(corpus, shingle_size=3, num_perm=126, seed=2)
    pi, pj, _ = M.minhash_lsh_pairs(signatures, bands=21, rows=6, min_similarity=0.5)
    found = {(int(a), int(b)) for a, b in zip(pi, pj)}
    missed = 0
    truth = 0
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if jaccard(corpus[i], corpus[j]) >= 0.8:
                truth += 1
                missed += (i, j) not in found and (j, i) not in found
    assert truth > 20, "the fixture should contain high-similarity pairs"
    assert missed / truth < 0.05, f"LSH missed {missed}/{truth} pairs above Jaccard 0.8"


def test_simhash_index_is_exhaustive_up_to_three_bits() -> None:
    """The six-block index must find every pair within Hamming distance 3."""
    rng = np.random.default_rng(3)
    fingerprints = rng.integers(0, 2**64, size=3000, dtype=np.uint64)
    # Plant near-duplicates by flipping a few bits of existing fingerprints.
    for source in range(0, 60, 2):
        flips = rng.choice(64, size=rng.integers(1, 4), replace=False)
        mutated = fingerprints[source]
        for bit in flips:
            mutated ^= np.uint64(1) << np.uint64(int(bit))
        fingerprints[source + 1] = mutated

    qi, qj, distances = M.simhash_pairs(fingerprints, max_distance=3)
    found = {(int(a), int(b)) for a, b in zip(qi, qj)}
    expected = set()
    block = np.bitwise_count(fingerprints[:, None] ^ fingerprints[None, :])
    for i, j in zip(*np.nonzero(block <= 3)):
        if i < j:
            expected.add((int(i), int(j)))
    assert found == expected
    assert np.all(distances <= 3)


def test_simhash_rejects_unsupported_distance() -> None:
    with pytest.raises(ValueError):
        M.simhash_pairs(np.zeros(4, dtype=np.uint64), max_distance=4)


def test_greedy_select_does_not_follow_transitive_chains() -> None:
    """Pairs (0,1) and (1,2): record 2 matches only the removed record 1, so it is kept.

    Similarity is not transitive - 0 and 2 can be far apart while both are close to 1 - so a
    record is removed only when it matches a record that is itself kept. SemHash's
    ``select_canonicals`` makes the same choice, which keeps the two comparable.
    """
    rep = M.greedy_select(3, np.array([0, 1]), np.array([1, 2]))
    assert rep.tolist() == [-1, 0, -1]

    # When record 2 also matches the kept record 0 directly, it is removed and points at 0.
    rep = M.greedy_select(3, np.array([0, 1, 0]), np.array([1, 2, 2]))
    assert rep.tolist() == [-1, 0, 0]


def test_greedy_select_is_order_independent_in_its_output_shape() -> None:
    pairs = [(0, 3), (1, 3), (0, 2)]
    rep = M.greedy_select(4, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    assert rep[0] == -1 and rep[1] == -1
    assert rep[2] == 0 and rep[3] == 0
