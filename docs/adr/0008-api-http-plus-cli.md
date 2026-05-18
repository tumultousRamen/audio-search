# ADR 0008 — Surface: FastAPI service + thin Typer CLI

**Status:** Accepted · 2026-05-18

## Decision

Primary surface is HTTP (FastAPI on `:8000`). A Typer CLI is a thin client that calls the same endpoints over `httpx` and renders a `rich` table.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | body: `{dataset, limit, resume?}` → blocks until done, returns counts |
| `GET`  | `/search` | `?q=&k=10&sources=text,audio,bm25` → ranked hits + per-source ranks + timing |
| `POST` | `/search-by-audio` | body: `{audio_path | base64}` → top-k via `clips_audio` (stretch path lit up day-1) |
| `GET`  | `/eval` | `?config=baseline|+audio|+bm25&probe_set=auto|hand|all` → metrics dict |
| `GET`  | `/clip/{id}` | metadata + transcript + audio_path (so demo can `afplay`) |
| `GET`  | `/health` | model load status, ns sizes |

## Response shape (search)

```python
class SearchHit(BaseModel):
    id: str
    rrf_score: float
    text_rank: int | None
    audio_rank: int | None
    bm25_rank: int | None
    transcript: str
    source: str
    duration_s: float
    audio_path: str

class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    timing_ms: dict[str, float]   # {"text": 12, "audio": 18, "bm25": 8, "rrf": 1, "total": 22}
```

Returning per-source ranks + timing makes the system **inspectable**: at demo time we can show "this clip jumped from text_rank=18 to top-3 because audio_rank=1." Story-telling artefact, not optional.

## Alternatives

| Option | Why rejected |
|---|---|
| CLI only | breaks the "ingestion + search service" framing in the brief |
| HTTP only | live-demo stage friction; CLI's 10 extra LOC is cheap insurance |
| GraphQL | over-engineered for 5 endpoints |

## Consequences

- One process holds the encoders + tpuf client + serves both surfaces
- CLI talks to the running service → no model double-load, no state divergence
- Future: drop a web UI on top of the same HTTP contract; clients are interchangeable
