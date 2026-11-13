# Deduplicating ag_news and wikitext-103

**Experiment 1 — data deduplication.** Four deduplication techniques built from scratch,
benchmarked against [SemHash](https://github.com/MinishLab/semhash), the state-of-the-art
open-source semantic deduplicator.

The interesting question is not how much each method removes. It is which method is *right*,
and the datasets do not come with an answer key. Most of the work here is building ways to
find out.

---

## 1. Setup

| | ag_news (train) | wikitext-103 (train) |
|---|---|---|
| records | 120,000 news snippets | 1,801,350 lines, of which 636,321 are blank → **1,165,029** used |
| median length | 37 tokens | 86 tokens for substantive lines; headings are 1–3 tokens |
| exact duplicates | 0 | 191,616 (16.45%) |

wikitext is distributed as raw lines, so it is full of structural fragments: blank lines,
section headings such as `= = History = =`, discography rows, and sports tables. 16% of its
lines are byte-identical to an earlier line before any clever method is applied. ag_news has
no byte-identical pair at all, which already says the two corpora test different things.

## 2. Techniques implemented

All four are in `methods.py`, NumPy only, with no deduplication library involved.

| Technique | Detects | Mechanism |
|---|---|---|
| **Exact hashing** | identical text, raw or normalised | SHA-1 of the text, optionally after Unicode/case/punctuation folding |
| **Bloom filter** | the same, in bounded memory | `m = −n·ln p / ln²2` bits, `k = (m/n)·ln2` hash functions by double hashing |
| **MinHash + LSH** | near-duplicates by **Jaccard** of shingles | 128 permutations, banded 21 × 6; a shared band makes a candidate pair |
| **SimHash** | near-duplicates by **Hamming** distance | per-bit voting over token hashes; 6-block permutation index, exhaustive to 3 bits |
| *SemHash (baseline)* | **semantic** duplicates | `potion-base-8M` static embeddings, approximate-NN search over cosine |

Shingling, MinHash and SimHash are fully vectorised: all documents are concatenated into one
symbol stream, windows are taken over the whole array at once and then masked so none crosses a
document boundary. 1.16M documents are featurised in a single pass rather than a Python loop.

Every method keeps the first occurrence and attributes each removal to a record that is itself
kept, never to one already removed. Similarity is not transitive, so following chains would let
a document be deleted because of a document it does not resemble. SemHash's `select_canonicals`
makes the same choice, so the comparison is like-for-like.

## 3. How much each method removes

| method | ag_news removed | wikitext removed | wikitext time |
|---|---|---|---|
| Exact hash | 0 (0.00%) | 191,616 (16.45%) | 1.8 s |
| Exact, normalised | 218 (0.18%) | 205,298 (17.62%) | 18.5 s |
| Bloom filter | 231 (0.19%) | 205,330 (17.62%) | 23.1 s |
| SimHash ≤ 3 bits | 1,857 (1.55%) | 234,578 (20.14%) | 97.9 s |
| MinHash ≥ 0.7 | 5,190 (4.33%) | 210,976 (18.11%) | 122.6 s |
| **SemHash ≥ 0.9** | **12,128 (10.11%)** | **264,450 (22.70%)** | 464.4 s |
| SemHash ≥ 0.8 | 29,542 (24.62%) | 395,602 (33.96%) | 566.8 s |

The Bloom filter reproduces normalised exact dedup with 13 false positives on ag_news and 32 on
wikitext — 0.016% of its decisions — using a 2.1 MB bit array instead of storing 1.16M digests.
The approximation behaves exactly as the theory predicts, which is the point of including it.

SemHash removes **2.3× more than MinHash on ag_news** and flags **18.5% of the ag_news test set**
as duplicating a training record, against MinHash's 7.8%. Those are very different claims about
the same corpus, and nothing above can say which is correct.

## 4. Does the extra removal find real duplicates? A ground-truth benchmark

Since the corpora carry no duplicate labels, I built an answer key: 20,000 ag_news records
(exact duplicates already removed, natural near-duplicates deliberately left in), plus 1,500
perturbed copies for each of seven kinds of edit — 10,500 known duplicates in total.

Precision is reported as a **lower bound**: the base is natural text, so some removals outside
the answer key may be genuine duplicates ag_news already contained. A second, independent error
signal avoids that ambiguity — ag_news is labelled, and a true duplicate of a World-news item
is also World news, so a removed pair whose two labels disagree is a near-certain mistake.

### Recall by kind of edit (%)

| method | reformat | typos | drop 15% | replace 15% | truncate 70% | swap halves | boilerplate |
|---|---|---|---|---|---|---|---|
| Exact, normalised | 100 | 0 | 1 | 0 | 0 | 0 | 0 |
| SimHash ≤ 3 | 100 | 2 | 14 | 4 | 1 | 100 | 2 |
| MinHash ≥ 0.7 | 100 | 3 | 13 | 8 | 26 | 100 | 41 |
| MinHash ≥ 0.5 | 100 | 15 | 36 | 26 | 84 | 100 | 80 |
| SemHash ≥ 0.9 | 100 | 24 | 95 | 54 | 80 | 100 | 31 |
| SemHash ≥ 0.8 | 99 | 66 | 97 | 87 | 95 | 100 | 80 |

Four things fall out of this table that the removal counts could never show.

**Word order is free.** Every method scores 100 on *swap halves*. Shingle sets, bit-vote
fingerprints and bag-of-words embeddings are all order-insensitive, so a reordered document is
invisible to all of them. A deduplicator that claims to catch "restructured" copies is claiming
something none of these techniques actually does.

**Deletion is where semantics wins.** Dropping 15% of the words costs MinHash almost everything
(13%) and costs SemHash almost nothing (95%). Removing words destroys the specific 3-word
shingles that MinHash depends on, while the meaning — and therefore the embedding — barely
moves. This is the single clearest justification for an embedding-based deduplicator.

**Boilerplate is where semantics loses.** A copy wrapped in scraped navigation text is found by
MinHash ≥ 0.5 (80%) but missed by SemHash ≥ 0.9 (31%). The added text leaves shingle overlap
largely intact but pulls the embedding away. For web-scale corpora, where boilerplate-wrapped
copies are the norm, this is a serious gap in the semantic method.

**Typos broke my own implementation.** Character-level corruption of 3% of characters left
MinHash at 3% and SimHash at 2%: one wrong letter destroys a whole word, and therefore every
word shingle containing it. That is a flaw in my design choice, not in the technique, so I
fixed it by shingling characters instead of words — see §5.

## 5. Two hybrid deduplicators

The benchmark shows the two families failing in opposite directions: lexical methods miss
rewrites, the semantic method misses boilerplate and over-removes topic twins. Three
constructions of my own exploit that.

**Character-shingle MinHash.** Same LSH machinery, 5-character shingles instead of 3-word ones.
A typo damages a handful of character windows rather than destroying whole tokens. On a typo
pair the estimated Jaccard rises from 0.23 to 0.58, with unrelated text still at 0.00.

**Cascade** (`hybrid.py`). Exact → MinHash-LSH → embed only the survivors. The lexical pass is
roughly 20× cheaper per record than embedding, so paying it first shrinks the expensive stage.

**Two-signal** (`hybrid.py`). Remove a record only when the semantic score clears a high
threshold **and** the pair shares literal wording (estimated Jaccard ≥ 0.3). This targets
SemHash's dominant failure: two different stories about one topic sit close in embedding space
but share little wording.

### Quality frontier on the answer key

| method | recall | precision ≥ | F1 ≥ | cross-label removals |
|---|---|---|---|---|
| SimHash ≤ 3 | 0.318 | 0.984 | 0.481 | 3 |
| MinHash ≥ 0.7 | 0.414 | 0.963 | 0.579 | 13 |
| MinHash ≥ 0.5 | 0.628 | 0.960 | 0.759 | 26 |
| SemHash ≥ 0.9 | 0.691 | 0.939 | 0.796 | 47 |
| SemHash ≥ 0.85 | 0.830 | 0.907 | 0.866 | 88 |
| SemHash ≥ 0.8 | **0.891** | 0.843 | 0.866 | 204 |
| Cascade | 0.736 | 0.941 | 0.826 | 44 |
| **Two-signal (cos 0.85)** | 0.816 | 0.946 | **0.876** | 54 |

The two-signal rule gives the best F1 of anything tested. Against SemHash ≥ 0.85, which it most
resembles in recall, it makes 946 fewer removals outside the answer key and **39% fewer
label-contradicting removals** (54 vs 88). Against SemHash ≥ 0.8, which beats it on recall by
7.5 points, it makes a quarter as many label-contradicting removals.

Requiring two independent signals to agree is a cheap way to buy precision: the lexical check
costs one MinHash signature comparison per candidate pair the embedding has already proposed.

## 6. What exactly does SemHash remove that MinHash does not?

57,454 wikitext lines and 7,145 ag_news records are removed by SemHash ≥ 0.9 and kept by
MinHash ≥ 0.7. Characterising them settles the question better than any aggregate.

| | wikitext disputed | wikitext agreed by both | wikitext kept by both |
|---|---|---|---|
| records | 57,454 | 206,996 | 896,599 |
| median length | **3 tokens** | 1 token | **86 tokens** |
| under 10 tokens | 70.8% | 92.7% | 17.2% |

On wikitext the disagreement is almost entirely about fragments. The lines SemHash removes
alone have a median length of three tokens, against 86 for the lines both methods keep. A
three-word section heading carries almost no information beyond its topic, so its embedding is
dominated by that topic and any two headings about adjacent subjects land close together.
Real examples from the run:

> removed `= = = Diamond Jubilee = = =` as a duplicate of `= = = Diamond = = =`
> removed `= = Campaign development = =` as a duplicate of `= = Campaign = =`

These are not duplicates. They are different sections of different articles.

On ag_news the disputed removals are full-length prose (median 39 tokens, the same as
everything else), and they split in two. 39.3% have strong lexical overlap (Jaccard ≥ 0.5) with
a kept record — genuine near-duplicates that MinHash's 0.7 threshold simply missed. 56.4% share
almost no wording (below 0.2), and those are a mixture of real paraphrase and real error:

> **Correctly merged** — `Stocks Sag as Colgate Warns, Oil Jumps` with
> `Stocks Slip After Colgate Warns`. Different headlines, same wire story.
>
> **Wrongly merged** — `Telekom Austria Plans to Buy Mobiltel for $1.5 Bln` with
> `Telekom Austria Wins Mobiltel Bid Option for $2.1 Bln`. Same companies, two different events
> at two different prices.

So SemHash's extra removals on ag_news are substantially real and its extra removals on
wikitext are substantially spurious. The corpus decides, which is why a single published
removal rate is not a quality claim.

## 7. Does any of it help downstream?

A deduplicator is only useful if the cleaned corpus trains a better model, so I trained a
TF-IDF + logistic-regression topic classifier on ag_news under each deduplication setting.

Deduplication also *shrinks* the training set, and less data costs accuracy on its own. Every
deduplicated run is therefore paired with a random subsample of exactly the same size, drawn
five times so the comparison carries an error bar.

| training set | size | deduplicated | random subsample | difference |
|---|---|---|---|---|
| full corpus | 120,000 | 92.55% | — | — |
| exact, normalised | 119,782 | 92.41% | 92.53% ± 0.05 | −0.12 (beyond 2σ) |
| SimHash ≤ 3 | 118,143 | 92.41% | 92.52% ± 0.10 | −0.11 (noise) |
| MinHash ≥ 0.7 | 114,810 | 92.46% | 92.49% ± 0.06 | −0.03 (noise) |
| SemHash ≥ 0.9 | 107,874 | 92.29% | 92.35% ± 0.05 | −0.06 (noise) |
| **SemHash ≥ 0.8** | **90,459** | **92.33%** | 92.02% ± 0.10 | **+0.31 (beyond 2σ)** |

**Deduplication is a budget allocator, not a free win.** At removal rates under 10% it changes
nothing a classifier can detect: every difference is inside the noise of simply drawing fewer
records. But when a quarter of the corpus has to go, dropping duplicates beats dropping at
random by 0.31 points — the only comparison here that clears two standard deviations in the
positive direction. The honest claim is conditional: deduplication earns its place when you are
already discarding data, which is exactly the pretraining situation it was designed for.

The one measurable *cost* appears at the other end. Removing 218 exact duplicates scored 0.12
points **below** removing 218 records at random, outside 2σ. Duplicated records act as mild
upweighting of whatever they say; strip them and that signal goes too.

### Train/test contamination does not inflate this benchmark

7.8% of the ag_news test set duplicates a training record according to MinHash ≥ 0.7, and 18.5%
according to SemHash ≥ 0.9. The standard worry is that a model scores inflated accuracy on
those records because it memorised them in training. It does not:

| test slice | MinHash-flagged | SemHash-flagged |
|---|---|---|
| accuracy on contaminated records | 92.10% | 92.88% |
| accuracy on clean records | 92.59% | 92.48% |

The contaminated slice is no easier than the clean one — for MinHash it is marginally harder.
A 120,000-example linear classifier over a 4-way topic task has no capacity to memorise
individual documents, so contamination buys it nothing. This result is specific to that
setting: it is evidence against assuming leakage inflates every benchmark, not evidence that
leakage is harmless for large generative models, where verbatim memorisation is the actual
mechanism of concern.

### Label agreement is a weak proxy for semantic methods

A true duplicate of a World-news item is also World news, so I measured how often removed pairs
share a label. Against a 25% random-pair baseline, every method scores 87–92%: exact 87.2%,
MinHash 90.7%, SimHash 92.2%, SemHash 91.3–91.6%. The proxy does not discriminate, and for a
clear reason — same-topic stories share a label by construction, which is precisely the case
where semantic deduplication errs. It is informative in one direction only: ag_news contains
byte-identical texts carrying contradicting labels, which is annotation noise rather than a
deduplication finding.
