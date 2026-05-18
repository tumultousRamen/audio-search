# ADR 0006 — Evaluation: transcript-as-gold + ablation table

**Status:** Accepted · 2026-05-18

## Decision

Two probe sets evaluated against three fusion configs. Single source of truth: the shipped transcript.

| Probe set | Size | How generated |
|---|---|---|
| `auto` | 200 | random clip → take 3–7-word substring of transcript → query; gold = source clip |
| `hand` | 10 | hand-written paraphrase / accent / cross-dataset queries; gold = curated clip(s) |

Probes sampled with `seed=42`. Same probes across runs → comparable numbers.

| Fusion config | Sources fused |
|---|---|
| `baseline` | text-vec only |
| `+audio`   | text-vec + audio-vec |
| `+bm25`    | text-vec + audio-vec + bm25 (full) |

## Metrics

Recall@1 / Recall@5 / Recall@10 (primary), MRR (secondary). Reported overall **and** per `source` (CommonVoice / LibriSpeech / AudioCaps).

## Why transcript-as-gold

Both primary datasets ship hand-validated transcripts. A query that is a substring of clip _X_'s transcript is unambiguously about clip _X_ — strict relevance, no labelling cost, no LLM-judge ceiling.

```mermaid
flowchart LR
    T["transcript: 'thank you for calling our support line'"]
    T -->|substring 3-7 words| Q["query: 'thank you for calling'"]
    Q --> S[search system]
    S --> R[top-k clips]
    R -->|gold = source clip| M[Recall@k, MRR]
```

## Alternatives

| Option | Why rejected |
|---|---|
| LLM-as-judge on top-k | research notes warn judge sets its own ceiling + overconfidence bias; deferred to stretch |
| Human-curated qrel set | too slow for a day prototype |
| Whole-transcript as query | trivial exact-match; doesn't test paraphrase |
| Random-mask BERT cloze queries | uniform-random masks generate ungrammatical queries; sub-phrase keeps queries natural |

## Consequences

- Eval runs in < 1 min for 210 probes × 3 configs = 630 queries
- Single-pass ablation produces the headline table for the README + presentation
- Drift / online eval: out of scope for prototype; sketched in [ADR 0009](0009-stretch-and-future-work.md)
