"""Eval harness — transcript-as-gold for speech corpora.

Probe generation:
  - auto: random clip → 3–7-word substring of its transcript → query; gold = source clip.
  - hand: a tiny curated set committed to disk (eval/hand_probes.json), gold ids referenced by id.

Configs evaluated (ablation):
  - baseline : text-vec only
  - +audio   : text-vec + audio-vec
  - +bm25    : text-vec + audio-vec + bm25

Metrics: Recall@1, Recall@5, Recall@10, MRR. Reported overall and per source.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from audio_search.config import get_settings
from audio_search.index import get_text_table
from audio_search.search import ALL_SOURCES, search


# ---------- Probe schema ---------- #


@dataclass(slots=True)
class Probe:
    qid: str                 # stable probe id ("auto_001" / "hand_dog_barking")
    query: str               # the user-side query string
    gold_ids: list[str]      # one or more clip ids considered relevant
    source: str | None = None  # for per-source breakdown ("librispeech" | "fleurs" | None for hand)
    note: str | None = None  # human note for hand probes


# ---------- Auto probe generation ---------- #


def _take_subphrase(text: str, n_min: int = 3, n_max: int = 7, rng: random.Random | None = None) -> str:
    """Pick a contiguous 3–7-word window of the transcript."""
    rng = rng or random.Random()
    words = [w for w in text.split() if w]
    if len(words) <= n_min:
        return text.strip()
    n = rng.randint(n_min, min(n_max, len(words)))
    start = rng.randint(0, len(words) - n)
    return " ".join(words[start : start + n])


def generate_auto_probes(n: int = 200, seed: int = 42) -> list[Probe]:
    """Sample N clips uniformly from clips_text and craft a sub-phrase probe each."""
    tbl = get_text_table()
    total = tbl.count_rows()
    if total == 0:
        return []
    take = min(n * 3, total)  # over-sample to filter trivially short transcripts
    rows = (
        tbl.search()
        .limit(take)
        .select(["id", "transcript", "source"])
        .to_list()
    )
    rng = random.Random(seed)
    rng.shuffle(rows)
    probes: list[Probe] = []
    for r in rows:
        t = (r.get("transcript") or "").strip()
        if len(t.split()) < 3:
            continue
        q = _take_subphrase(t, rng=rng)
        if not q:
            continue
        probes.append(Probe(
            qid=f"auto_{len(probes):03d}",
            query=q,
            gold_ids=[r["id"]],
            source=r.get("source") or None,
        ))
        if len(probes) >= n:
            break
    return probes


def load_hand_probes(path: Path | None = None) -> list[Probe]:
    p = path or (get_settings().eval_dir / "hand_probes.json")
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    return [Probe(**r) for r in raw]


# ---------- Metric computation ---------- #


def _recall_at_k(gold: Iterable[str], ranked_ids: list[str], k: int) -> float:
    gold_set = set(gold)
    return 1.0 if any(rid in gold_set for rid in ranked_ids[:k]) else 0.0


def _mrr(gold: Iterable[str], ranked_ids: list[str]) -> float:
    gold_set = set(gold)
    for i, rid in enumerate(ranked_ids):
        if rid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


CONFIGS: dict[str, tuple[str, ...]] = {
    "baseline": ("text",),
    "+audio":   ("text", "audio"),
    "+bm25":    ("text", "audio", "bm25"),
}


@dataclass(slots=True)
class ProbeResult:
    qid: str
    query: str
    gold_ids: list[str]
    source: str | None
    top_ids: list[str]
    hit_rank: int | None    # 0-indexed rank of first gold hit in top-10, or None
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def run_config(probes: list[Probe], config: str, top_k: int = 10) -> list[ProbeResult]:
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; valid: {list(CONFIGS)}")
    sources = CONFIGS[config]
    out: list[ProbeResult] = []
    for p in probes:
        r = search(query=p.query, top_k=top_k, sources=sources)
        ids = [h["id"] for h in r["hits"]]
        gold = set(p.gold_ids)
        hit_rank = next((i for i, rid in enumerate(ids) if rid in gold), None)
        out.append(ProbeResult(
            qid=p.qid,
            query=p.query,
            gold_ids=p.gold_ids,
            source=p.source,
            top_ids=ids,
            hit_rank=hit_rank,
            recall_at_1=_recall_at_k(gold, ids, 1),
            recall_at_5=_recall_at_k(gold, ids, 5),
            recall_at_10=_recall_at_k(gold, ids, 10),
            mrr=_mrr(gold, ids),
        ))
    return out


def aggregate(results: list[ProbeResult], by_source: bool = False) -> dict:
    def _agg(rs: list[ProbeResult]) -> dict:
        if not rs:
            return {"n": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        n = len(rs)
        return {
            "n": n,
            "recall@1": round(sum(r.recall_at_1 for r in rs) / n, 4),
            "recall@5": round(sum(r.recall_at_5 for r in rs) / n, 4),
            "recall@10": round(sum(r.recall_at_10 for r in rs) / n, 4),
            "mrr": round(sum(r.mrr for r in rs) / n, 4),
        }

    overall = _agg(results)
    if not by_source:
        return overall
    by_src: dict[str, dict] = {}
    sources = sorted({r.source or "_unknown" for r in results})
    for src in sources:
        by_src[src] = _agg([r for r in results if (r.source or "_unknown") == src])
    return {"overall": overall, "by_source": by_src}


def run_full_eval(
    n_auto: int = 200,
    seed: int = 42,
    configs: tuple[str, ...] = ("baseline", "+audio", "+bm25"),
    out_path: Path | None = None,
) -> dict:
    """Generate probes, run every config, write structured results."""
    auto = generate_auto_probes(n=n_auto, seed=seed)
    hand = load_hand_probes()
    probes = auto + hand

    if not probes:
        return {"error": "no probes — is the index empty?"}

    summary: dict = {
        "n_probes_total": len(probes),
        "n_auto": len(auto),
        "n_hand": len(hand),
        "configs": {},
    }
    raw_results: dict[str, list[dict]] = {}

    for cfg in configs:
        results = run_config(probes, cfg)
        summary["configs"][cfg] = aggregate(results, by_source=True)
        raw_results[cfg] = [asdict(r) for r in results]

    payload = {"summary": summary, "raw": raw_results, "seed": seed}
    p = out_path or (get_settings().eval_dir / "results.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return summary
