from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from audio_search.config import get_settings
from audio_search.embed import warmup
from audio_search.eval import run_full_eval
from audio_search.index import list_clips, namespace_counts
from audio_search.ingest import ingest as run_ingest
from audio_search.search import ALL_SOURCES, SourceName, search, search_by_audio


# ---------- Lifespan ---------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm models so first /search isn't slow
    info = warmup()
    app.state.warmup = info
    yield


app = FastAPI(title="audio-search", version="0.1.0", lifespan=lifespan)


# ---------- Pydantic models ---------- #


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
    timing_ms: dict[str, float]


class SearchByAudioHit(BaseModel):
    id: str
    audio_rank: int
    transcript: str
    source: str
    duration_s: float
    audio_path: str


class SearchByAudioResponse(BaseModel):
    query_audio_path: str
    hits: list[SearchByAudioHit]
    timing_ms: dict[str, float]


class IngestRequest(BaseModel):
    dataset: Literal["librispeech", "fleurs", "esc50"]
    limit: int | None = None
    batch_size: int = 32
    resume: bool = False
    skip_audio: bool = False


class IngestResponse(BaseModel):
    dataset: str
    total: int
    text_upserts: int
    audio_upserts: int
    elapsed_s: float
    throughput_clips_per_s: float | None


class HealthResponse(BaseModel):
    status: str
    device: str
    text_dim: int
    audio_dim: int
    namespace_counts: dict[str, int]


# ---------- Endpoints ---------- #


@app.get("/health", response_model=HealthResponse)
def health():
    info = app.state.warmup
    return HealthResponse(
        status="ok",
        device=info["device"],
        text_dim=info["text_dim"],
        audio_dim=info["audio_dim"],
        namespace_counts=namespace_counts(),
    )


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    sources: str = Query("text,audio,bm25", description="comma-separated subset of text,audio,bm25"),
    source_filter: str | None = Query(None, description="filter by clip source: librispeech|fleurs|esc50"),
):
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    bad = [s for s in src_list if s not in ALL_SOURCES]
    if bad:
        raise HTTPException(400, f"unknown sources: {bad}; valid: {list(ALL_SOURCES)}")
    where = {"source": source_filter} if source_filter else None
    result = search(query=q, top_k=k, sources=src_list, where=where)  # type: ignore[arg-type]
    return result


class SearchByAudioRequest(BaseModel):
    audio_path: str
    k: int = 10
    source_filter: str | None = None


@app.post("/search-by-audio", response_model=SearchByAudioResponse)
def search_by_audio_endpoint(req: SearchByAudioRequest):
    p = Path(req.audio_path)
    if not p.exists():
        raise HTTPException(400, f"audio_path not found: {req.audio_path}")
    where = {"source": req.source_filter} if req.source_filter else None
    return search_by_audio(audio_path=str(p), top_k=req.k, where=where)


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    return run_ingest(
        dataset=req.dataset,
        limit=req.limit,
        batch_size=req.batch_size,
        resume=req.resume,
        skip_audio=req.skip_audio,
    )


class EvalRequest(BaseModel):
    n_auto: int = 200
    seed: int = 42
    configs: list[str] | None = None


@app.post("/eval")
def eval_endpoint(req: EvalRequest):
    cfgs = tuple(req.configs) if req.configs else ("baseline", "+audio", "+bm25")
    return run_full_eval(n_auto=req.n_auto, seed=req.seed, configs=cfgs)


class ClipSummary(BaseModel):
    id: str
    transcript: str
    speaker_id: str
    accent: str
    gender: str
    source: str
    lang: str
    duration_s: float
    audio_path: str


class ClipsListResponse(BaseModel):
    clips: list[ClipSummary]
    next_cursor: str | None
    limit: int
    total: int


@app.get("/clips", response_model=ClipsListResponse)
def list_clips_endpoint(
    limit: int = Query(20, ge=1, le=200, description="page size"),
    cursor: str | None = Query(None, description="opaque cursor = last id of previous page"),
    source: str | None = Query(None, description="filter: commonvoice | librispeech | fleurs | audiocaps"),
):
    """Cursor-paginated browse of indexed clips, ordered by id ascending.

    Pass `next_cursor` from the response as `cursor` in the next request.
    A null `next_cursor` indicates the end of the listing.
    """
    return list_clips(limit=limit, cursor=cursor, source_filter=source)


@app.get("/clip/{clip_id}")
def clip_endpoint(clip_id: str, play: int = Query(0, description="if 1, stream the wav file")):
    from audio_search.index import get_by_id

    row = get_by_id(clip_id)
    if not row:
        raise HTTPException(404, f"clip {clip_id!r} not found")

    if play and row.get("audio_path"):
        p = Path(row["audio_path"])
        if p.exists():
            return FileResponse(p, media_type="audio/wav")
    return row
