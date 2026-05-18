from __future__ import annotations

from pathlib import Path
from typing import Iterator

from audio_search.adapters.base import Clip


def _norm(s: str) -> str:
    return s.strip().lower()


class LibriSpeechAdapter:
    """OpenSLR LibriSpeech via HuggingFace `datasets`.

    Uses `openslr/librispeech_asr`, `clean` config, `validation` split (dev-clean).
    """

    source_name = "librispeech"

    def __init__(
        self,
        version: str = "openslr/librispeech_asr",
        config: str = "clean",
        split: str = "validation",
        cache_audio_dir: Path = Path("./data/librispeech_audio"),
        seed: int = 42,
    ) -> None:
        self.version = version
        self.config = config
        self.split = split
        self.cache_audio_dir = cache_audio_dir
        self.cache_audio_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def iter_clips(self, limit: int | None = None) -> Iterator[Clip]:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset(self.version, self.config, split=self.split)
        ds = ds.shuffle(seed=self.seed)
        if limit is not None:
            ds = ds.select(range(limit))

        for row in ds:
            audio = row["audio"]
            samples = audio.get_all_samples()
            sample_rate = int(samples.sample_rate)
            arr = samples.data.squeeze().cpu().numpy()
            duration_s = float(arr.shape[-1] / sample_rate)

            spk = str(row.get("speaker_id", "unk"))
            chap = str(row.get("chapter_id", "0"))
            utt = str(row.get("id") or row.get("file") or int(duration_s * 1000))
            utt_short = Path(utt).stem
            clip_id = f"ls_{spk}_{chap}_{utt_short}"

            local_path = self.cache_audio_dir / f"{clip_id}.wav"
            if not local_path.exists():
                sf.write(local_path, arr, sample_rate, subtype="PCM_16")

            yield Clip(
                id=clip_id,
                audio_path=local_path,
                transcript=_norm(row["text"]),
                duration_s=duration_s,
                source=self.source_name,
                lang="en",
                speaker_id=spk,
                accent=None,
                gender=None,
                extra={"chapter_id": chap},
            )
