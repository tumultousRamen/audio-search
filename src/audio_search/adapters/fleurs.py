"""FLEURS adapter — replaces CommonVoice (which Mozilla pulled from HF in Oct 2025).

`google/fleurs` is a 102-language read-speech corpus, ~10s clips with transcripts.
Default config is `en_us`; multiple configs (e.g. `hi_in`, `de_de`) can be concatenated
upstream for multilingual demos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from audio_search.adapters.base import Clip


def _norm(s: str) -> str:
    return s.strip().lower()


class FleursAdapter:
    source_name = "fleurs"

    def __init__(
        self,
        config: str = "en_us",
        split: str = "validation",
        cache_audio_dir: Path = Path("./data/fleurs_audio"),
        seed: int = 42,
    ) -> None:
        self.config = config
        self.split = split
        self.cache_audio_dir = cache_audio_dir
        self.cache_audio_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def iter_clips(self, limit: int | None = None) -> Iterator[Clip]:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset("google/fleurs", self.config, split=self.split)
        ds = ds.shuffle(seed=self.seed)
        if limit is not None:
            ds = ds.select(range(limit))

        for row in ds:
            audio = row["audio"]
            samples = audio.get_all_samples()
            sample_rate = int(samples.sample_rate)
            arr = samples.data.squeeze().cpu().numpy()
            duration_s = float(arr.shape[-1] / sample_rate)

            row_id = str(row.get("id", row.get("path", int(duration_s * 1000))))
            clip_id = f"fl_{self.config}_{row_id}"

            local_path = self.cache_audio_dir / f"{clip_id}.wav"
            if not local_path.exists():
                sf.write(local_path, arr, sample_rate, subtype="PCM_16")

            transcript = row.get("raw_transcription") or row.get("transcription") or ""

            yield Clip(
                id=clip_id,
                audio_path=local_path,
                transcript=_norm(transcript),
                duration_s=duration_s,
                source=self.source_name,
                lang=row.get("language", self.config.split("_")[0]),
                speaker_id=None,
                accent=self.config,  # locale doubles as accent label
                gender=row.get("gender") if isinstance(row.get("gender"), str) else None,
                extra={"lang_id": row.get("lang_id"), "fleurs_id": row.get("id")},
            )
