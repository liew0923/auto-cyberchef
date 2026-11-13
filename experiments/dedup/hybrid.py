"""Two hybrid deduplicators that combine the lexical and the semantic signal.

Motivation from the single-method results: MinHash-LSH is cheap and precise but blind to
paraphrase, while SemHash removes 10-25% of ag_news, far more than the corpus plausibly
contains, because an embedding at cosine 0.8-0.9 cannot separate "the same story" from "another
story about the same topic". Each failure mode is the other's strength.

``cascade``      - recall/cost oriented: exact -> MinHash-LSH -> embed only the survivors.
                   Same removals as running the semantic pass on everything, minus the records
                   the lexical pass already caught, so fewer texts have to be embedded.
``two_signal``   - precision oriented: remove a record only when the semantic score clears a
                   high threshold AND the pair shares some literal wording (estimated Jaccard
                   over 3-word shingles). Topic twins share embeddings space but not wording.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from model2vec import StaticModel
from semhash.index import Index
from vicinity import Backend

import methods as M

DEFAULT_MODEL = "minishlab/potion-base-8M"


@dataclass
class HybridResult:
    rep: np.ndarray
    seconds: float
    stages: dict = field(default_factory=dict)

    @property
    def removed(self) -> np.ndarray:
        return self.rep >= 0


def _embed(texts: list[str], model: StaticModel) -> np.ndarray:
    vectors = model.encode(texts, show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _semantic_pairs(vectors: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs (i < j) of embeddings with cosine >= threshold.

    Uses SemHash's own index wrapper, so the approximate-NN behaviour is identical to the
    baseline and the comparison measures the deduplication rule, not the index.
    """
    index = Index.from_vectors_and_items(
        vectors=vectors, items=[[{"i": str(i)}] for i in range(len(vectors))], backend_type=Backend.USEARCH
    )
    lefts, rights, scores = [], [], []
    for query, (neighbours, similarities) in enumerate(index.query_threshold(vectors, threshold=threshold)):
        neighbours = np.asarray(neighbours, dtype=np.int64)
        keep = neighbours < query
        if keep.any():
            lefts.append(neighbours[keep])
            rights.append(np.full(int(keep.sum()), query, dtype=np.int64))
            scores.append(np.asarray(similarities, dtype=np.float32)[keep])
    if not lefts:
        empty_i = np.empty(0, np.int64)
        return empty_i, empty_i.copy(), np.empty(0, np.float32)
    return np.concatenate(lefts), np.concatenate(rights), np.concatenate(scores)


def cascade(
    texts: list[str], jaccard: float = 0.7, cosine: float = 0.9, model: StaticModel | None = None
) -> HybridResult:
    """Exact -> MinHash-LSH -> semantic pass on the survivors only."""
    model = model or StaticModel.from_pretrained(DEFAULT_MODEL)
    start, stages = time.perf_counter(), {}

    rep = M.exact_dedup(texts, normalized=True)
    stages["after_exact"] = int((rep < 0).sum())

    alive = np.flatnonzero(rep < 0)
    signatures, _ = M.featurize([texts[i] for i in alive], shingle_size=3, num_perm=128)
    pi, pj, _ = M.minhash_lsh_pairs(signatures, min_similarity=jaccard)
    lexical = M.greedy_select(len(alive), pi, pj)
    hit = lexical >= 0
    rep[alive[hit]] = alive[lexical[hit]]
    stages["after_minhash"] = int((rep < 0).sum())
    stages["seconds_lexical"] = round(time.perf_counter() - start, 2)

    survivors = np.flatnonzero(rep < 0)
    vectors = _embed([texts[i] for i in survivors], model)
    si, sj, _ = _semantic_pairs(vectors, cosine)
    semantic = M.greedy_select(len(survivors), si, sj)
    hit = semantic >= 0
    rep[survivors[hit]] = survivors[semantic[hit]]
    stages["after_semantic"] = int((rep < 0).sum())
    stages["embedded_records"] = int(len(survivors))
    stages["embedding_saved_pct"] = round(100 * (1 - len(survivors) / len(texts)), 2)
    return HybridResult(rep=rep, seconds=round(time.perf_counter() - start, 2), stages=stages)


def two_signal(
    texts: list[str], cosine: float = 0.9, min_jaccard: float = 0.3, model: StaticModel | None = None
) -> HybridResult:
    """Remove a record only if a semantic match ALSO shares literal wording."""
    model = model or StaticModel.from_pretrained(DEFAULT_MODEL)
    start, stages = time.perf_counter(), {}

    rep = M.exact_dedup(texts, normalized=True)
    alive = np.flatnonzero(rep < 0)
    alive_texts = [texts[i] for i in alive]

    vectors = _embed(alive_texts, model)
    si, sj, score = _semantic_pairs(vectors, cosine)
    stages["semantic_candidate_pairs"] = int(len(si))

    signatures, _ = M.featurize(alive_texts, shingle_size=3, num_perm=128)
    estimated = np.empty(len(si), dtype=np.float32)
    for lo in range(0, len(si), 200_000):
        hi = lo + 200_000
        estimated[lo:hi] = (signatures[si[lo:hi]] == signatures[sj[lo:hi]]).mean(axis=1)
    keep = estimated >= min_jaccard
    stages["pairs_rejected_as_topic_twins"] = int((~keep).sum())

    selected = M.greedy_select(len(alive), si[keep], sj[keep])
    hit = selected >= 0
    rep[alive[hit]] = alive[selected[hit]]
    return HybridResult(rep=rep, seconds=round(time.perf_counter() - start, 2), stages=stages)
