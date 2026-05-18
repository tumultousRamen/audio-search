from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    lancedb_path: str = Field(default="./data/lancedb", alias="LANCEDB_PATH")

    # Logical table names (kept under the `tpuf_ns_*` attrs for backwards-compat after
    # the LanceDB pivot; renamed lazily as touched.)
    tpuf_ns_text: str = Field(default="clips_text", alias="TABLE_TEXT")
    tpuf_ns_audio: str = Field(default="clips_audio", alias="TABLE_AUDIO")

    text_encoder: str = Field(default="BAAI/bge-small-en-v1.5", alias="TEXT_ENCODER")
    audio_encoder: str = Field(default="laion/clap-htsat-unfused", alias="AUDIO_ENCODER")

    device: str | None = Field(default=None, alias="DEVICE")
    hf_home: str = Field(default="./.cache/huggingface", alias="HF_HOME")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    data_dir: Path = Path("./data")
    eval_dir: Path = Path("./eval")
    cache_dir: Path = Path("./eval/cache")

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    os.environ.setdefault("HF_HOME", s.hf_home)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    s.eval_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
