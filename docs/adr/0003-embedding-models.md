# ADR 0003 — Embedding models: bge-small (text) + CLAP (audio)

**Status:** Accepted · 2026-05-18

## Decision

| Modality | Model | Dim | Why |
|---|---|---|---|
| Text | `BAAI/bge-small-en-v1.5` | 384 | top-tier MTEB retrieval, ~5 ms/sentence CPU, ~1 ms MPS |
| Audio | `laion/clap-htsat-unfused` | 512 | **joint** text-audio space → text query directly hits audio vectors; covers speech + AudioCaps stretch |

## Context

We need (a) a strong text retrieval signal on transcripts and (b) an audio signal that a text query can search directly without a second encoder.

## Alternatives

| Audio option | Why rejected |
|---|---|
| Whisper-encoder mean-pool | audio-only space, no text→audio path without going via transcript route |
| WavLM / HuBERT | strong on speech tasks, but no joint-text head → same blocker |
| MS-CLAP | comparable to LAION-CLAP, slightly larger; LAION ships fused/unfused, lighter HF load |
| AudioCLIP, MuLan | weaker open weights / licence friction |

| Text option | Why rejected |
|---|---|
| all-MiniLM-L6-v2 | smaller MTEB score; ≈10 % retrieval gap |
| nomic-embed-text-v1.5 | 768-dim, 2× memory; marginal quality for transcripts |
| bge-large | 1024-dim, 3× embed latency; not needed at this corpus size |

## Consequences

- Query path needs **two** encoders loaded: `bge-small` for text-to-text-vec; `clap-text` for text-to-audio-vec
- Both fit in <2 GB RAM, load once at API startup
- Memory budget: 1000 clips × (384 + 512) dims × fp32 = ~3.5 MB raw vectors, negligible
