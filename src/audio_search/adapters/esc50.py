"""ESC-50 adapter — 2000 environmental sound clips across 50 classes.

This is the non-speech corpus that lets CLAP earn its retrieval slot. The class
`category` ("dog", "rain", "glass_breaking", ...) doubles as the "transcript",
so all three retrieval sources — text-dense / BM25 / CLAP audio — have signal
on queries like "dog barking" or "rainfall".

Source: <https://huggingface.co/datasets/ashraq/esc50>
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from audio_search.adapters.base import Clip


def _humanize(category: str) -> str:
    """`glass_breaking` -> `glass breaking`; suitable as a search target."""
    return category.replace("_", " ").strip().lower()


class ESC50Adapter:
    source_name = "esc50"

    def __init__(
        self,
        split: str = "train",  # ESC-50 ships a single split
        cache_audio_dir: Path = Path("./data/esc50_audio"),
        seed: int = 42,
    ) -> None:
        self.split = split
        self.cache_audio_dir = cache_audio_dir
        self.cache_audio_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def iter_clips(self, limit: int | None = None) -> Iterator[Clip]:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset("ashraq/esc50", split=self.split)
        ds = ds.shuffle(seed=self.seed)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        for row in ds:
            audio = row["audio"]
            samples = audio.get_all_samples()
            sample_rate = int(samples.sample_rate)
            arr = samples.data.squeeze().cpu().numpy()
            duration_s = float(arr.shape[-1] / sample_rate)

            filename = row["filename"]  # e.g. "1-100032-A-0.wav"
            stem = filename.rsplit(".", 1)[0]
            clip_id = f"esc_{stem}"

            local_path = self.cache_audio_dir / f"{clip_id}.wav"
            if not local_path.exists():
                sf.write(local_path, arr, sample_rate, subtype="PCM_16")

            category = row.get("category") or ""
            transcript = _humanize(category)  # used by BM25 + bge as the textual signal

            yield Clip(
                id=clip_id,
                audio_path=local_path,
                transcript=transcript,
                duration_s=duration_s,
                source=self.source_name,
                lang="n/a",
                speaker_id=None,
                accent=None,
                gender=None,
                extra={
                    "category": category,
                    "target": row.get("target"),
                    "fold": row.get("fold"),
                    "esc10": row.get("esc10"),
                },
            )
