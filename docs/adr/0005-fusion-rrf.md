# ADR 0005 — Fusion via Reciprocal Rank Fusion, k = 60

**Status:** Accepted · 2026-05-18

## Decision

Fuse the three ranked lists (text-vec ANN, audio-vec ANN, BM25 FTS) with RRF, `k = 60`, top-30 per source → top-10 final.

```python
def rrf(ranked_lists: list[list[str]], k: int = 60, top_k: int = 10) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

## Context

Three heterogeneous signals with non-comparable scores (cosine ∈ [-1, 1], BM25 unbounded ≥ 0). We need fusion that does not require score calibration and that we have no training data to learn.

## Alternatives

| Option | Why rejected |
|---|---|
| Weighted score sum | requires per-source score normalization; weights need tuning data we lack |
| Cascade reranker (BM25 → dense → cross-encoder) | adds a cross-encoder model + inference cost; ≥ 50 LOC; deferred |
| Learned linear / LambdaMART | needs labelled training data; not available |

## Why k = 60

Cormack et al. 2009 default. Empirically near-optimal across IR benchmarks. Smaller k (≈ 10) over-weights rank-1 hits; larger k flattens the distribution. We do **not** tune k on the eval set — that leaks the probe distribution into the algorithm.

## Consequences

- 5 LOC fusion
- Trivial ablation: drop a list, re-fuse → `text only` vs `text + audio` vs `text + audio + bm25`. This is the demo story.
- No score normalization code anywhere
- Future: replace with a learned reranker once we have labelled (query, relevant_clip) pairs at scale
