"""Step C of the pipeline: stitch converted chunks back into a WAV."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import librosa
import numpy as np
import soundfile as sf

from .manifests import ConvertedManifest, ChunkPayload, load_converted_manifest


@dataclass
class StitchConfig:
    sample_rate: int = 16_000
    crossfade_seconds: float = 1.5


def stitch_manifest_to_file(
    manifest_path: Path,
    output_path: Path,
    config: Optional[StitchConfig] = None,
    manifest: Optional[ConvertedManifest] = None,
) -> Path:
    cfg = config or StitchConfig()
    manifest_obj = manifest or load_converted_manifest(manifest_path)
    audio = stitch_chunks(manifest_obj.converted_chunks, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, cfg.sample_rate)
    return output_path


def stitch_chunks(chunks: Iterable[ChunkPayload], cfg: StitchConfig) -> np.ndarray:
    ordered = sorted(chunks, key=lambda chunk: chunk.start)
    crossfade = int(cfg.crossfade_seconds * cfg.sample_rate)

    buffer = np.zeros(1, dtype=np.float32)
    for item in ordered:
        chunk_audio = _decode_chunk(item, cfg.sample_rate)
        buffer = _place_chunk(buffer, chunk_audio, start=item.start, crossfade=crossfade)
    return buffer


def _decode_chunk(chunk: ChunkPayload, target_sr: int) -> np.ndarray:
    audio_bytes = base64.b64decode(chunk.audio_b64)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return np.asarray(data, dtype=np.float32)


def _place_chunk(existing: np.ndarray, chunk: np.ndarray, start: int, crossfade: int) -> np.ndarray:
    if chunk.size == 0:
        return existing

    end = start + chunk.size
    if end > existing.size:
        pad = end - existing.size
        existing = np.pad(existing, (0, pad))

    overlap = min(crossfade, chunk.size, existing.size - start)
    if overlap > 0:
        fade = np.linspace(0.0, 1.0, num=overlap, dtype=np.float32)
        keep = 1.0 - fade
        existing[start : start + overlap] = (
            existing[start : start + overlap] * keep + chunk[:overlap] * fade
        )
        existing[start + overlap : end] = chunk[overlap:]
    else:
        existing[start:end] = chunk

    return existing


__all__ = ["StitchConfig", "stitch_manifest_to_file", "stitch_chunks"]
