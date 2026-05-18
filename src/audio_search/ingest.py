from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from audio_search.adapters.base import Clip, DatasetAdapter
from audio_search.adapters.fleurs import FleursAdapter
from audio_search.adapters.librispeech import LibriSpeechAdapter
from audio_search.config import get_settings
from audio_search.embed import encode_audios, encode_texts
from audio_search.index import upsert_audio, upsert_text


console = Console()


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "fleurs": FleursAdapter,
    "librispeech": LibriSpeechAdapter,
}


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def _checkpoint_path(dataset: str) -> Path:
    return get_settings().cache_dir / f"checkpoint_{dataset}.json"


def _load_checkpoint(dataset: str) -> set[str]:
    p = _checkpoint_path(dataset)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()).get("ids", []))
    except Exception:
        return set()


def _save_checkpoint(dataset: str, seen_ids: set[str]) -> None:
    p = _checkpoint_path(dataset)
    p.write_text(json.dumps({"ids": sorted(seen_ids)}))


def ingest(
    dataset: str,
    limit: int | None = None,
    batch_size: int = 32,
    skip_audio: bool = False,
    resume: bool = False,
) -> dict:
    """Batched-sync ingestion: stream clips → embed text + audio → upsert both namespaces.

    Idempotent: clip ids are deterministic; tpuf upsert by id is a no-op for unchanged rows.
    Resumable: on `resume=True`, ids in the local checkpoint are skipped.
    """
    if dataset not in ADAPTERS:
        raise ValueError(f"unknown dataset {dataset!r}; available: {list(ADAPTERS)}")
    adapter = ADAPTERS[dataset]()

    seen = _load_checkpoint(dataset) if resume else set()
    if seen:
        console.log(f"resume: skipping {len(seen)} previously ingested ids")

    t_start = time.perf_counter()
    n_total = 0
    n_text = 0
    n_audio = 0

    iter_clips = adapter.iter_clips(limit=limit)
    if seen:
        iter_clips = (c for c in iter_clips if c.id not in seen)

    with Progress(
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"ingest {dataset}", total=limit)

        for batch in _batched(iter_clips, batch_size):
            transcripts = [c.transcript for c in batch]
            text_vecs = encode_texts(transcripts)

            text_rows = list(zip(batch, text_vecs))
            n_text += upsert_text(text_rows)

            if not skip_audio:
                audio_vecs = encode_audios([c.audio_path for c in batch])
                audio_rows = list(zip(batch, audio_vecs))
                n_audio += upsert_audio(audio_rows)

            for c in batch:
                seen.add(c.id)
            n_total += len(batch)
            _save_checkpoint(dataset, seen)
            progress.update(task, advance=len(batch))

    elapsed = round(time.perf_counter() - t_start, 2)
    return {
        "dataset": dataset,
        "total": n_total,
        "text_upserts": n_text,
        "audio_upserts": n_audio,
        "elapsed_s": elapsed,
        "throughput_clips_per_s": round(n_total / elapsed, 2) if elapsed > 0 else None,
    }
