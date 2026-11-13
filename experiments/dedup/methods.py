"""Deduplication techniques implemented from scratch (NumPy only).

Every method reports its result as a ``rep`` array of length ``n``:
``rep[j] == -1`` means record ``j`` is kept, otherwise ``rep[j]`` is the index
of the earlier record that ``j`` duplicates. All methods keep the first
occurrence, which is also what SemHash does, so the outputs are comparable.

Techniques
----------
* exact hashing (raw text / normalised text)
* Bloom filter (streaming, fixed-memory approximation of exact dedup)
* MinHash signatures + LSH banding (Jaccard similarity of word shingles)
* SimHash fingerprints + block-permutation index (Hamming distance)
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from array import array

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
U64 = np.uint64
_C1, _C2 = U64(0xBF58476D1CE4E5B9), U64(0x94D049BB133111EB)
_GOLD = U64(0x9E3779B97F4A7C15)


def normalize(text: str) -> str:
    """Lower-case, Unicode-fold and reduce a text to space-separated alphanumeric tokens."""
    return " ".join(TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower()))


def mix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser: a cheap, well-distributed 64-bit integer hash."""
    x = (x ^ (x >> U64(30))) * _C1
    x = (x ^ (x >> U64(27))) * _C2
    return x ^ (x >> U64(31))


# --------------------------------------------------------------------------- exact


def exact_dedup(texts: list[str], normalized: bool = False) -> np.ndarray:
    """Hash every record and drop records whose digest was seen before."""
    rep = np.full(len(texts), -1, dtype=np.int64)
    seen: dict[bytes, int] = {}
    for j, text in enumerate(texts):
        key = hashlib.sha1((normalize(text) if normalized else text).encode()).digest()
        first = seen.setdefault(key, j)
        if first != j:
            rep[j] = first
    return rep


class BloomFilter:
    """Classic Bloom filter with k hash functions derived by double hashing."""

    def __init__(self, capacity: int, fp_rate: float = 1e-3) -> None:
        self.m = max(8, math.ceil(-capacity * math.log(fp_rate) / math.log(2) ** 2))
        self.k = max(1, round(self.m / capacity * math.log(2)))
        self.bits = bytearray(self.m // 8 + 1)

    def add_if_new(self, item: bytes) -> bool:
        """Insert ``item``; return True if it was (probably) already present."""
        digest = hashlib.blake2b(item, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1
        bits, m, present = self.bits, self.m, True
        for i in range(self.k):
            pos = (h1 + i * h2) % m
            byte, bit = pos >> 3, 1 << (pos & 7)
            if not bits[byte] & bit:
                present = False
                bits[byte] |= bit
        return present


def bloom_dedup(texts: list[str], fp_rate: float = 1e-3, normalized: bool = True) -> tuple[np.ndarray, BloomFilter]:
    """Streaming dedup with a Bloom filter. Returns a boolean 'removed' mask (no representative is known)."""
    bloom = BloomFilter(len(texts), fp_rate)
    removed = np.zeros(len(texts), dtype=bool)
    for j, text in enumerate(texts):
        removed[j] = bloom.add_if_new((normalize(text) if normalized else text).encode())
    return removed, bloom


# ----------------------------------------------------------------- featurisation


class _Vocab(dict):
    def __missing__(self, key: str) -> int:
        value = self[key] = len(self)
        return value


def _rolling_hashes(
    ids: np.ndarray, lengths: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hash every length-n window of each document's symbol sequence.

    ``ids`` is the concatenation of all documents' symbols and ``lengths`` their sizes. Windows
    are taken over the whole flat array and then masked so none spans a document boundary,
    which keeps the work in one vectorised pass instead of a Python loop per document. A
    document shorter than ``n`` contributes a single window covering all of it, so every
    document always has at least one shingle.
    """
    n_docs, total = len(lengths), int(lengths.sum())
    symbol_hash = mix64(ids.astype(U64) + _GOLD)
    doc_of_pos = np.repeat(np.arange(n_docs, dtype=np.int64), lengths)
    mult = mix64(np.arange(1, n + 1, dtype=U64))

    values, owners = [], []
    if total >= n:
        acc = np.zeros(total - n + 1, dtype=U64)
        for q in range(n):
            acc += symbol_hash[q : total - n + 1 + q] * mult[q]
        valid = doc_of_pos[: total - n + 1] == doc_of_pos[n - 1 :]
        values.append(mix64(acc[valid]))
        owners.append(doc_of_pos[: total - n + 1][valid])

    short = lengths < n
    short_acc = np.zeros(n_docs, dtype=U64)
    if total:
        starts = np.cumsum(lengths) - lengths
        in_short = short[doc_of_pos]
        pos_in_doc = (np.arange(total) - starts[doc_of_pos])[in_short]
        np.add.at(short_acc, doc_of_pos[in_short], symbol_hash[in_short] * mult[pos_in_doc])
    values.append(mix64(short_acc[short] + _GOLD))
    owners.append(np.flatnonzero(short))

    values, owners = np.concatenate(values), np.concatenate(owners)
    order = np.argsort(owners, kind="stable")
    counts = np.where(short, 1, np.maximum(lengths - n + 1, 1))
    return values[order], np.cumsum(counts) - counts, counts


def _word_shingles(texts: list[str], vocab: _Vocab, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Hashes of the word n-gram shingles of each text, plus each document's start offset."""
    ids, lens = array("q"), []
    for text in texts:
        tokens = TOKEN_RE.findall(text.lower())
        lens.append(len(tokens))
        ids.extend(map(vocab.__getitem__, tokens))
    flat = np.frombuffer(ids, dtype=np.int64) if len(ids) else np.zeros(0, dtype=np.int64)
    values, starts, _ = _rolling_hashes(flat, np.asarray(lens, dtype=np.int64), n)
    return values, starts


def _char_shingles(texts: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Hashes of the character n-gram shingles of the normalised text.

    Character shingles survive the typos and inflections that break word shingles: one wrong
    letter damages n windows instead of destroying a whole token.
    """
    buffers, lens = [], []
    for text in texts:
        codes = np.frombuffer(normalize(text).encode("utf-8", "ignore"), dtype=np.uint8)
        buffers.append(codes)
        lens.append(len(codes))
    flat = np.concatenate(buffers).astype(np.int64) if buffers else np.zeros(0, dtype=np.int64)
    values, starts, _ = _rolling_hashes(flat, np.asarray(lens, dtype=np.int64), n)
    return values, starts


def featurize(
    texts: list[str],
    shingle_size: int = 3,
    num_perm: int = 128,
    seed: int = 0,
    chunk: int = 100_000,
    unit: str = "word",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MinHash signatures (n, num_perm) uint32 and 64-bit SimHash fingerprints for all texts.

    :param unit: ``"word"`` shingles n consecutive words, ``"char"`` n consecutive characters of
        the normalised text. Character shingles cost more to build but survive typos, which
        destroy whole tokens and therefore whole word shingles.
    """
    if unit not in ("word", "char"):
        raise ValueError("unit must be 'word' or 'char'")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2**64, size=num_perm, dtype=U64) | U64(1)
    b = rng.integers(0, 2**64, size=num_perm, dtype=U64)
    signatures = np.empty((len(texts), num_perm), dtype=np.uint32)
    fingerprints = np.zeros(len(texts), dtype=U64)
    vocab = _Vocab()
    for lo in range(0, len(texts), chunk):
        batch = texts[lo : lo + chunk]
        if unit == "word":
            hashes, starts = _word_shingles(batch, vocab, shingle_size)
        else:
            hashes, starts = _char_shingles(batch, shingle_size)
        hi = lo + len(starts)
        # MinHash: minimum of each random hash function over the document's shingle set.
        for k in range(num_perm):
            permuted = ((hashes * a[k] + b[k]) >> U64(32)).astype(np.uint32)
            signatures[lo:hi, k] = np.minimum.reduceat(permuted, starts)
        # SimHash: every shingle votes +1/-1 on each bit; the sign of the sum is the fingerprint bit.
        fp = np.zeros(len(starts), dtype=U64)
        for bit in range(64):
            votes = ((hashes >> U64(bit)) & U64(1)).astype(np.int32) * 2 - 1
            fp |= (np.add.reduceat(votes, starts) > 0).astype(U64) << U64(bit)
        fingerprints[lo:hi] = fp
    return signatures, fingerprints


# ------------------------------------------------------------ candidate generation


def _bucket_pairs(keys: np.ndarray, window: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """All pairs (i < j) of rows sharing a key, limited to ``window`` neighbours inside very large buckets."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    left, right = [], []
    for d in range(1, window):
        same = sorted_keys[:-d] == sorted_keys[d:]
        if not same.any():
            break
        left.append(order[:-d][same])
        right.append(order[d:][same])
    if not left:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(left), np.concatenate(right)


def _unique_pairs(lefts: list[np.ndarray], rights: list[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    codes = np.unique(np.concatenate(lefts) * n + np.concatenate(rights))
    return codes // n, codes % n


def minhash_lsh_pairs(
    signatures: np.ndarray, bands: int = 21, rows: int = 6, min_similarity: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSH banding: records colliding in any band are candidates; verify with the full signature.

    Returns (i, j, estimated Jaccard) for all candidate pairs with estimate >= ``min_similarity``.
    """
    n = len(signatures)
    mult = mix64(np.arange(1, rows + 1, dtype=U64))
    lefts, rights = [], []
    for band in range(bands):
        cols = signatures[:, band * rows : (band + 1) * rows].astype(U64)
        left, right = _bucket_pairs(mix64((cols * mult).sum(axis=1, dtype=U64)))
        lefts.append(left)
        rights.append(right)
    pi, pj = _unique_pairs(lefts, rights, n)
    sim = np.empty(len(pi), dtype=np.float32)
    for lo in range(0, len(pi), 200_000):
        hi = lo + 200_000
        sim[lo:hi] = (signatures[pi[lo:hi]] == signatures[pj[lo:hi]]).mean(axis=1)
    keep = sim >= min_similarity
    return pi[keep], pj[keep], sim[keep]


SIMHASH_BLOCKS = (0, 11, 22, 33, 44, 54, 64)


def simhash_pairs(fingerprints: np.ndarray, max_distance: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all fingerprint pairs within ``max_distance`` bits (Manku et al. block-permutation index).

    The 64 bits are split into 6 blocks. Two fingerprints that differ in at most 3 bits
    agree exactly on at least 3 blocks, so one table per 3-block combination finds them all.
    """
    if max_distance > 3:
        raise ValueError("the 6-block index only guarantees recall up to Hamming distance 3")
    n = len(fingerprints)
    masks = [
        U64(((1 << (hi - lo)) - 1) << lo) for lo, hi in zip(SIMHASH_BLOCKS[:-1], SIMHASH_BLOCKS[1:])
    ]
    lefts, rights = [], []
    for combo in itertools.combinations(masks, 3):
        left, right = _bucket_pairs(fingerprints & (combo[0] | combo[1] | combo[2]))
        lefts.append(left)
        rights.append(right)
    pi, pj = _unique_pairs(lefts, rights, n)
    dist = np.bitwise_count(fingerprints[pi] ^ fingerprints[pj]).astype(np.int64)
    keep = dist <= max_distance
    return pi[keep], pj[keep], dist[keep]


# --------------------------------------------------------------------- selection


def greedy_select(n: int, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Keep-first selection: record j is removed if it matches an earlier record that is itself kept."""
    rep = np.full(n, -1, dtype=np.int64)
    order = np.lexsort((pi, pj))
    removed = bytearray(n)
    for i, j in zip(pi[order].tolist(), pj[order].tolist()):
        if not removed[j] and not removed[i]:
            removed[j] = 1
            rep[j] = i
    return rep
