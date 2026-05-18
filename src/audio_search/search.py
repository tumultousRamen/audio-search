from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable, Literal

from audio_search.embed import encode_text_for_audio_space, encode_texts
from audio_search.index import query_audio, query_bm25, query_text


SourceName = Literal["text", "audio", "bm25"]
ALL_SOURCES: tuple[SourceName, ...] = ("text", "audio", "bm25")


def rrf(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack 2009).

    Each input list is a ranked sequence of doc ids. Output is the fused top-k
    (doc_id, score) sorted by descending score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


def search(
    query: str,
    top_k: int = 10,
    sources: Iterable[SourceName] = ALL_SOURCES,
    where: dict | None = None,
    per_source_k: int = 30,
    rrf_k: int = 60,
) -> dict:
    """Run the configured retrieval sources, fuse with RRF, return the hit dict.

    Shape returned:
      {
        "query": str,
        "hits": [
            {"id", "rrf_score", "text_rank", "audio_rank", "bm25_rank",
             "transcript", "source", "duration_s", "audio_path"}, ...
        ],
        "timing_ms": {"text", "audio", "bm25", "rrf", "total"},
      }
    """
    sources = tuple(sources)
    timing: dict[str, float] = {}
    ranked: dict[str, list[dict]] = {}

    t_total = time.perf_counter()

    if "text" in sources:
        t0 = time.perf_counter()
        qvec = encode_texts([query])[0]
        ranked["text"] = query_text(qvec, top_k=per_source_k, where=where)
        timing["text"] = round((time.perf_counter() - t0) * 1000, 2)

    if "audio" in sources:
        t0 = time.perf_counter()
        qvec_aud = encode_text_for_audio_space([query])[0]
        ranked["audio"] = query_audio(qvec_aud, top_k=per_source_k, where=where)
        timing["audio"] = round((time.perf_counter() - t0) * 1000, 2)

    if "bm25" in sources:
        t0 = time.perf_counter()
        ranked["bm25"] = query_bm25(query, top_k=per_source_k, where=where)
        timing["bm25"] = round((time.perf_counter() - t0) * 1000, 2)

    # Build rank maps for each source
    rank_maps: dict[str, dict[str, int]] = {
        src: {row["id"]: i for i, row in enumerate(ranked[src])} for src in ranked
    }

    # Hit attributes from any source that returned them (text > audio > bm25)
    attrs_by_id: dict[str, dict] = {}
    for src in ("text", "audio", "bm25"):
        for row in ranked.get(src, []):
            if row["id"] not in attrs_by_id:
                attrs_by_id[row["id"]] = row

    # RRF fusion
    t0 = time.perf_counter()
    fused = rrf([[r["id"] for r in ranked[src]] for src in ranked], k=rrf_k, top_k=top_k)
    timing["rrf"] = round((time.perf_counter() - t0) * 1000, 2)

    hits = []
    for doc_id, score in fused:
        a = attrs_by_id.get(doc_id, {})
        hits.append({
            "id": doc_id,
            "rrf_score": round(float(score), 6),
            "text_rank": rank_maps.get("text", {}).get(doc_id),
            "audio_rank": rank_maps.get("audio", {}).get(doc_id),
            "bm25_rank": rank_maps.get("bm25", {}).get(doc_id),
            "transcript": a.get("transcript", ""),
            "source": a.get("source", ""),
            "duration_s": a.get("duration_s", 0.0),
            "audio_path": a.get("audio_path", ""),
        })

    timing["total"] = round((time.perf_counter() - t_total) * 1000, 2)
    return {"query": query, "hits": hits, "timing_ms": timing}


def search_by_audio(
    audio_path: str,
    top_k: int = 10,
    where: dict | None = None,
    per_source_k: int = 30,
) -> dict:
    """Audio-to-audio: embed input audio with CLAP, query clips_audio."""
    from audio_search.embed import encode_audios

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    qvec = encode_audios([audio_path])[0]
    embed_ms = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    rows = query_audio(qvec, top_k=per_source_k, where=where)
    query_ms = round((time.perf_counter() - t0) * 1000, 2)

    hits = []
    for i, row in enumerate(rows[:top_k]):
        hits.append({
            "id": row["id"],
            "audio_rank": i,
            "transcript": row.get("transcript", ""),
            "source": row.get("source", ""),
            "duration_s": row.get("duration_s", 0.0),
            "audio_path": row.get("audio_path", ""),
        })

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)
    return {
        "query_audio_path": audio_path,
        "hits": hits,
        "timing_ms": {"embed": embed_ms, "audio": query_ms, "total": total_ms},
    }
