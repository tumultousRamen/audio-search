"""LanceDB vector store wrapper.

Two tables, both keyed by `id`. Mirrored scalar attributes so each can be queried
+ filtered independently. BM25 / full-text search is a native LanceDB index over
the `transcript` column of `clips_text`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from audio_search.adapters.base import Clip
from audio_search.config import get_settings


# ---------- Schemas ---------- #


def _scalar_fields() -> list[pa.Field]:
    return [
        pa.field("id", pa.string()),
        pa.field("transcript", pa.string()),
        pa.field("speaker_id", pa.string()),
        pa.field("accent", pa.string()),
        pa.field("gender", pa.string()),
        pa.field("source", pa.string()),
        pa.field("lang", pa.string()),
        pa.field("duration_s", pa.float32()),
        pa.field("audio_path", pa.string()),
    ]


def _text_schema(dim: int = 384) -> pa.Schema:
    return pa.schema(_scalar_fields() + [pa.field("vector", pa.list_(pa.float32(), dim))])


def _audio_schema(dim: int = 512) -> pa.Schema:
    return pa.schema(_scalar_fields() + [pa.field("vector", pa.list_(pa.float32(), dim))])


# ---------- Connection ---------- #


@lru_cache(maxsize=1)
def _db():
    import lancedb

    s = get_settings()
    path = Path(s.lancedb_path)
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def _get_or_create(name: str, schema: pa.Schema):
    db = _db()
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema, mode="create")


def get_text_table():
    s = get_settings()
    return _get_or_create(s.tpuf_ns_text, _text_schema(384))


def get_audio_table():
    s = get_settings()
    return _get_or_create(s.tpuf_ns_audio, _audio_schema(512))


# ---------- Row construction ---------- #


def _row(clip: Clip, vec: np.ndarray) -> dict[str, Any]:
    a = clip.attrs()
    return {
        "id": clip.id,
        "vector": vec.astype(np.float32).tolist(),
        **{k: a[k] for k in [
            "transcript", "speaker_id", "accent", "gender",
            "source", "lang", "duration_s", "audio_path",
        ]},
    }


def _merge_upsert(tbl, records: list[dict]) -> int:
    if not records:
        return 0
    (
        tbl.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(records)
    )
    return len(records)


# ---------- Upsert ---------- #


def upsert_text(rows: list[tuple[Clip, np.ndarray]]) -> int:
    """rows: list of (clip, 384-dim text_vec)."""
    if not rows:
        return 0
    tbl = get_text_table()
    records = [_row(c, v) for c, v in rows]
    n = _merge_upsert(tbl, records)
    _ensure_fts_index(tbl)
    return n


def upsert_audio(rows: list[tuple[Clip, np.ndarray]]) -> int:
    """rows: list of (clip, 512-dim audio_vec)."""
    if not rows:
        return 0
    tbl = get_audio_table()
    return _merge_upsert(tbl, [_row(c, v) for c, v in rows])


def _ensure_fts_index(tbl) -> None:
    """Create the native (Lance) FTS index on `transcript` if not already built."""
    try:
        existing = [ix.name for ix in tbl.list_indices()]
        if any("transcript" in n or "fts" in n.lower() for n in existing):
            return
        tbl.create_fts_index("transcript", use_tantivy=False, replace=False)
    except Exception:
        # Index may already exist or table may be too small; non-fatal
        pass


# ---------- Query ---------- #


def _where_to_sql(where: dict | None) -> str | None:
    if not where:
        return None
    clauses = []
    for k, v in where.items():
        if isinstance(v, str):
            clauses.append(f"{k} = '{v}'")
        elif isinstance(v, (int, float)):
            clauses.append(f"{k} = {v}")
    return " AND ".join(clauses) if clauses else None


_RETURN_COLS = [
    "id", "transcript", "speaker_id", "accent", "gender",
    "source", "lang", "duration_s", "audio_path",
]


def query_text(qvec: np.ndarray, top_k: int = 30, where: dict | None = None) -> list[dict]:
    tbl = get_text_table()
    q = tbl.search(qvec.astype(np.float32)).limit(top_k).select(_RETURN_COLS)
    sql_where = _where_to_sql(where)
    if sql_where:
        q = q.where(sql_where)
    return [_clean(r) for r in q.to_list()]


def query_audio(qvec: np.ndarray, top_k: int = 30, where: dict | None = None) -> list[dict]:
    tbl = get_audio_table()
    q = tbl.search(qvec.astype(np.float32)).limit(top_k).select(_RETURN_COLS)
    sql_where = _where_to_sql(where)
    if sql_where:
        q = q.where(sql_where)
    return [_clean(r) for r in q.to_list()]


def query_bm25(query_str: str, top_k: int = 30, where: dict | None = None) -> list[dict]:
    tbl = get_text_table()
    try:
        q = tbl.search(query_str, query_type="fts").limit(top_k).select(_RETURN_COLS)
    except Exception:
        # FTS index not built yet (empty table or recent upsert) — return empty
        return []
    sql_where = _where_to_sql(where)
    if sql_where:
        q = q.where(sql_where)
    return [_clean(r) for r in q.to_list()]


def _clean(row: dict) -> dict:
    # Lance returns float32 numpy scalars + the `_distance`/`_score` column we don't expose
    return {
        "id": row.get("id"),
        "transcript": row.get("transcript", ""),
        "speaker_id": row.get("speaker_id", ""),
        "accent": row.get("accent", ""),
        "gender": row.get("gender", ""),
        "source": row.get("source", ""),
        "lang": row.get("lang", ""),
        "duration_s": float(row.get("duration_s", 0.0) or 0.0),
        "audio_path": row.get("audio_path", ""),
    }


def namespace_counts() -> dict[str, int]:
    s = get_settings()
    out = {}
    db = _db()
    for label, name in [("clips_text", s.tpuf_ns_text), ("clips_audio", s.tpuf_ns_audio)]:
        try:
            out[label] = db.open_table(name).count_rows() if name in db.table_names() else 0
        except Exception:
            out[label] = -1
    return out


def list_clips(
    limit: int = 20,
    cursor: str | None = None,
    source_filter: str | None = None,
) -> dict:
    """Cursor-paginated browse over `clips_text`.

    Cursor is the last `id` of the previous page. Ordering is by `id` ascending,
    which is stable and deterministic given the deterministic clip-id scheme
    (`ls_<spk>_<chap>_<utt>`, `fl_<config>_<row>`).

    Implementation note: we sort client-side over the full table because LanceDB
    does not expose an inline ORDER BY at the time of writing. At corpus scale
    > ~10^5 clips this should be replaced with a server-side ordered scan
    (LanceDB roadmap) or a maintained secondary index of sorted ids.
    """
    s = get_settings()
    db = _db()
    if s.tpuf_ns_text not in db.table_names():
        return {"clips": [], "next_cursor": None, "limit": limit, "total": 0}

    tbl = db.open_table(s.tpuf_ns_text)
    where_parts: list[str] = []
    if source_filter:
        where_parts.append(f"source = '{source_filter}'")

    q = tbl.search().select(_RETURN_COLS)
    if where_parts:
        q = q.where(" AND ".join(where_parts))
    rows = q.to_list()
    rows.sort(key=lambda r: r["id"])

    if cursor:
        rows = [r for r in rows if r["id"] > cursor]

    page = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = page[-1]["id"] if (has_more and page) else None

    return {
        "clips": [_clean(r) for r in page],
        "next_cursor": next_cursor,
        "limit": limit,
        "total": tbl.count_rows(),
    }


def get_by_id(clip_id: str) -> dict | None:
    s = get_settings()
    db = _db()
    if s.tpuf_ns_text not in db.table_names():
        return None
    tbl = db.open_table(s.tpuf_ns_text)
    rows = tbl.search().where(f"id = '{clip_id}'").limit(1).select(_RETURN_COLS).to_list()
    return _clean(rows[0]) if rows else None
