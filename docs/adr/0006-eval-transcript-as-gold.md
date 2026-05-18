# ADR 0006 — Evaluation: transcript-as-gold + ablation table

**Status:** Accepted · 2026-05-18 (revised 2026-05-18 — non-speech corpus, per-source design insight)

## Decision

Two probe sets evaluated against three fusion configs. Single source of truth: the shipped transcript (for ESC-50, the humanised class name serves as the transcript).

| Probe set | Size | How generated |
|---|---|---|
| `auto` | 100 | random clip from `clips_text` → 3–7-word substring of transcript → query; gold = source clip |
| `hand` | 20 | hand-written: 10 LibriSpeech (substring + paraphrase + abstract semantic) + 10 ESC-50 (class-name + acoustic paraphrase); gold = curated clip id list |

Probes sampled with `seed=42`. Same probes across runs → comparable numbers.

| Fusion config | Sources fused |
|---|---|
| `baseline` | text-vec only |
| `+audio`   | text-vec + audio-vec |
| `+bm25`    | text-vec + audio-vec + bm25 (full) |

## Metrics

Recall@1 / Recall@5 / Recall@10 (primary), MRR (secondary). Reported overall **and** per `source` (`librispeech` / `esc50` / `fleurs`).

## Per-corpus design insight (post-ingestion of ESC-50)

The auto probe set is biased toward LibriSpeech because `generate_auto_probes` filters transcripts with `<3` words → ESC-50's 1-2-word class names are almost entirely excluded. The 120-probe run on the mixed index (LS 200 + ESC-50 2 000) yields **107 LS / 13 ESC** probes. Per-source numbers remain fair; the overall row tilts toward speech.

The richer headline that emerges from the per-source view:

- **Speech corpus + substring queries** → text-only baseline R@1 0.71; `+audio` drags to 0.36 because CLAP is uncorrelated with text-substring queries on read-speech.
- **Non-speech corpus + class-label transcripts** → text-only baseline R@1 0.77; `+audio` 0.69 (within noise). CLAP doesn't drag here, but the class name in the transcript means text wins via the back door — we can't isolate CLAP's text-to-audio contribution from this corpus alone.
- **Audio-to-audio path** (`/search-by-audio`, not in the table) hits 30/30 same-class retrieval across 6 ESC-50 classes. CLAP's slot is in audio queries, not text-to-audio fusion.

## Follow-up experiments (deferred)

- **Lower the word-count filter** to `<1` for short-transcript corpora; OR sample auto probes with per-source quota → fair representation in the auto set.
- **Ablate-transcripts** on ESC-50 (re-ingest with empty transcripts). Forces CLAP's text-tower to do retrieval without a text-dense shortcut. Cleanest measurement of CLAP joint-space quality for text→audio queries; tracked in [ADR 0009](0009-stretch-and-future-work.md).
- **LLM-as-judge** for non-speech captions when AudioCaps lands.

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

- Eval runs in ~5 min for 120 probes × 3 configs = 360 queries on the mixed 2 200-clip index (CLAP encode + 3-source RRF + LanceDB hybrid is the bottleneck, not the probe count)
- Single-pass ablation produces the headline table for the README + presentation
- Per-source breakdown is the load-bearing artefact — overall numbers tilted by auto-probe filter, per-source rows reveal the content-dependent fusion story
- Drift / online eval: out of scope for prototype; sketched in [ADR 0009](0009-stretch-and-future-work.md)
