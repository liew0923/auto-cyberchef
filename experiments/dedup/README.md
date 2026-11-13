# Experiment 1 — Data Deduplication

Deduplicating **ag_news** (120,000 news snippets) and **wikitext-103** (1,165,029 non-empty
lines), with every technique implemented from scratch in NumPy, benchmarked against the
state-of-the-art open-source project **[SemHash](https://github.com/MinishLab/semhash)**.

## What this adds to a plain "how many rows were removed" comparison

Removal counts cannot say which method is *correct*: a method that deletes a quarter of the
corpus is not four times better than one that deletes a sixteenth, it may simply be wrong. The
experiment therefore measures deduplication from four independent angles.

| Study | File | Question it answers |
|---|---|---|
| Removal rates | `run_experiment.py` | How much does each method remove, at what cost, and how much do the methods agree? |
| Ground-truth benchmark | `synthetic_eval.py` | Precision / recall / F1 against *known* near-duplicates injected with six kinds of edit. |
| Downstream impact | `downstream.py` | Does deduplication help a classifier, once the loss of training data is controlled for? How much train→test leakage exists? |
| Cost | `scaling.py` | Wall-clock and peak memory from 10k to 1.16M records. |
| Error bars | `control_variance.py` | Is the downstream difference real, or one lucky random draw? |
| Disagreement | `error_analysis.py` | Which records do the methods argue about, and who is right? |

Plus `hybrid.py`, which combines the lexical and semantic signals into two deduplicators of my
own, and `test_methods.py`, which checks each probabilistic structure against the exact
quantity it approximates.

## Techniques implemented from scratch (`methods.py`)

| Technique | Detects | Key idea |
|---|---|---|
| Exact hashing | byte-identical, or identical after Unicode/case/punctuation normalisation | SHA-1 of the (normalised) text |
| Bloom filter | the same, in fixed memory | `m = -n·ln(p)/ln²2` bits, `k` hash functions by double hashing |
| MinHash + LSH | near-duplicates by **Jaccard** of 3-word shingles | 128 permutations, banded into 21 bands × 6 rows; a shared band is a candidate |
| Character MinHash | near-duplicates **through typos** | the same index over 5-character shingles, which survive letter-level corruption |
| SimHash | near-duplicates by **Hamming** distance of a 64-bit fingerprint | bit-wise voting over token hashes; a 6-block permutation index finds every pair within 3 bits |
| SemHash (baseline) | **semantic** duplicates | static `potion-base-8M` embeddings + approximate-NN search over cosine similarity |

All shingling, MinHash and SimHash work is vectorised: documents are concatenated into one
token stream and the per-document boundaries are handled with `reduceat`, so 1.16M documents
are featurised in a single pass rather than in a Python loop.

### Hybrid deduplicators (`hybrid.py`, this work)

* **Cascade** — exact → MinHash-LSH → embed only the survivors. The lexical pass is orders of
  magnitude cheaper than embedding, so removing its hits first shrinks the semantic workload.
* **Two-signal** — remove a record only when the semantic score clears a high threshold **and**
  the pair shares literal wording (estimated Jaccard ≥ 0.3). This targets the dominant SemHash
  failure mode: two different news stories about the same topic sit close in embedding space
  but share almost no wording.

## Headline results

* Character-shingle MinHash alone reaches **F1 ≥ 0.891** on the answer key, against SemHash's
  best of 0.867 — with no embedding model at all.
* The character cascade reaches **F1 ≥ 0.916**, matching SemHash ≥ 0.8 on recall while making a
  third as many label-contradicting removals.
* Deduplication beats an equal-size random subsample **only** when removal is aggressive:
  +0.31 points at 24.6% removed (beyond 2σ), nothing measurable below 10%.
* 18.5% of the ag_news test set duplicates a training record, and the classifier scores **no
  better** on those records than on clean ones.
* Cost does not follow the lexical/semantic divide: character MinHash (995s on 1.16M wikitext
  lines) is **three times dearer** than the neural SemHash baseline (333s), because runtime
  tracks features per document rather than the kind of signature.

## Reproducing

```bash
pip install -r requirements.txt
python run_experiment.py     # removal rates on both datasets + SemHash baseline
python synthetic_eval.py     # precision / recall against injected duplicates
python downstream.py         # classifier accuracy, label agreement, train/test leakage
python scaling.py            # runtime and memory vs corpus size
python make_figures.py       # figures for the report
python -m pytest test_methods.py -q
```

Results are written to `results/` as JSON plus PNG figures. `run_experiment.py` must be run
first: the other scripts reuse its saved masks.

## Methodological choices worth noting

* **Keep-first selection.** Every method keeps the first occurrence and attributes each removal
  to a record that is itself kept — never to one that was already removed. Similarity is not
  transitive, so following chains would let a document be deleted because of a document it does
  not resemble. SemHash's `select_canonicals` makes the same choice, which keeps the comparison
  like-for-like.
* **Near-duplicate search runs on exact-unique records**, as SemHash does, so the thresholds are
  comparable across methods.
* **Size-controlled downstream evaluation.** Deduplication removes training data, so every
  deduplicated run is paired with a random subsample of exactly the same size. Without that
  control, a drop in accuracy caused by having less data looks like evidence that deduplication
  hurts.
* **Label agreement as a precision proxy.** ag_news is labelled, and a true duplicate of a
  World-news item is also World news. The share of removed pairs whose labels disagree is
  therefore a precision signal available on the real corpus, not only on injected duplicates.
