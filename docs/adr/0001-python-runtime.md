# ADR 0001 — Python runtime for the full stack

**Status:** Accepted · 2026-05-18

## Decision

Single-language Python stack: `uv` for deps, FastAPI for HTTP, Typer for CLI, PyTorch for embedding models.

## Context

Day-long build. Audio embedding models (Whisper, WavLM, CLAP) are first-class PyTorch citizens. Node alternatives carry hidden cost.

## Alternatives

| Option | Why rejected |
|---|---|
| TS/Node + ONNX Runtime | Model conversion + numerical-parity debugging ≈ 2 hr |
| TS/Node + remote embedding API | $/call, network latency, breaks offline demo |
| Node API in front of Python sidecar | ≈ 1 hr extra glue; typed surface not load-bearing for backend track |

## Consequences

- One process holds models + serves API → simple ops
- `uv` for fast installs and reproducible lockfile
- Swap to remote inference later if scale demands; service boundary already exists at `embed.py`
