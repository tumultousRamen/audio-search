from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol


@dataclass(slots=True)
class Clip:
    """Unified record for one audio clip across all dataset sources."""

    id: str                       # stable, deterministic — "cv_<client>_<row>" / "ls_<spk>_<chap>_<utt>"
    audio_path: Path              # local file path (mp3 / flac / wav)
    transcript: str               # normalized lowercase text
    duration_s: float
    source: str                   # "commonvoice" | "librispeech" | "audiocaps"
    lang: str = "en"
    speaker_id: str | None = None
    accent: str | None = None
    gender: str | None = None
    extra: dict = field(default_factory=dict)

    def attrs(self) -> dict:
        """Attributes payload mirrored into both turbopuffer namespaces."""
        return {
            "transcript": self.transcript,
            "speaker_id": self.speaker_id or "",
            "accent": self.accent or "",
            "gender": self.gender or "",
            "source": self.source,
            "lang": self.lang,
            "duration_s": float(self.duration_s),
            "audio_path": str(self.audio_path),
        }


class DatasetAdapter(Protocol):
    """One adapter per dataset. Yields Clip records lazily."""

    source_name: str

    def iter_clips(self, limit: int | None = None) -> Iterator[Clip]: ...
