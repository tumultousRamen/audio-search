from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from audio_search.config import get_settings


# ---------- Text encoder ---------- #


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    return SentenceTransformer(s.text_encoder, device=s.resolve_device())


def encode_texts(texts: list[str], normalize: bool = True) -> np.ndarray:
    """Returns (N, 384) float32 array of L2-normalized embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _text_model().encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32, copy=False)


# ---------- Audio encoder (CLAP) ---------- #


@lru_cache(maxsize=1)
def _clap_bundle():
    """Returns (model, processor, target_sample_rate)."""
    import torch
    from transformers import ClapModel, ClapProcessor

    s = get_settings()
    device = s.resolve_device()
    model = ClapModel.from_pretrained(s.audio_encoder).to(device).eval()
    processor = ClapProcessor.from_pretrained(s.audio_encoder)
    target_sr = int(processor.feature_extractor.sampling_rate)
    return model, processor, target_sr, device, torch


def _load_audio(path: Path | str, target_sr: int) -> np.ndarray:
    import librosa

    arr, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return arr.astype(np.float32)


def encode_audios(paths: list[Path | str], normalize: bool = True) -> np.ndarray:
    """Returns (N, 512) float32 array."""
    if not paths:
        return np.zeros((0, 512), dtype=np.float32)
    model, processor, target_sr, device, torch = _clap_bundle()
    arrays = [_load_audio(p, target_sr) for p in paths]
    inputs = processor(audio=arrays, sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_audio_features(**inputs)
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    vecs = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
    return vecs


def encode_text_for_audio_space(queries: list[str], normalize: bool = True) -> np.ndarray:
    """Embed a text query into the CLAP joint space, for querying clips_audio."""
    if not queries:
        return np.zeros((0, 512), dtype=np.float32)
    model, processor, target_sr, device, torch = _clap_bundle()
    inputs = processor(text=queries, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    vecs = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
    return vecs


# ---------- Smoke ---------- #


def smoke_test_text() -> tuple[int, int]:
    vecs = encode_texts(["hello world", "thank you for calling"])
    return vecs.shape


def warmup() -> dict:
    """Load both models, return dims. Used by /health and during startup."""
    t = encode_texts(["warmup"])
    return {"text_dim": int(t.shape[1]), "audio_dim": 512, "device": get_settings().resolve_device()}
