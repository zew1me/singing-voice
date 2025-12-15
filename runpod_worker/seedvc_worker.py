"""Reference implementation for the Seed-VC RunPod worker (Step B)."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

try:
    from seed_vc.inference import inference_pipeline
except ImportError:  # pragma: no cover - the actual worker image installs this.
    inference_pipeline = None  # type: ignore[assignment]

from singing_voice.manifests import ChunkPayload, ConvertedManifest, PreprocessManifest

DEFAULT_SAMPLE_RATE = 16_000


def convert_manifest(
    manifest: PreprocessManifest,
    target_voice_path: Path,
    model_path: Path,
) -> ConvertedManifest:
    if inference_pipeline is None:  # pragma: no cover - guard for local linting
        raise RuntimeError("seed_vc is not available. Install it inside the RunPod image.")

    target_voice, _ = librosa.load(target_voice_path, sr=manifest.sample_rate)

    converted = []
    for chunk in manifest.chunks:
        source_audio = _decode_chunk(chunk, manifest.sample_rate)
        converted_audio = inference_pipeline(
            source_audio,
            target_voice,
            model_path=str(model_path),
        )
        converted.append(
            ChunkPayload(
                chunk_id=chunk.chunk_id,
                start=chunk.start,
                end=chunk.end,
                duration=len(converted_audio) / manifest.sample_rate,
                audio_b64=_encode_wav(converted_audio, manifest.sample_rate),
            )
        )

    return ConvertedManifest(sample_rate=manifest.sample_rate, converted_chunks=converted)


def _decode_chunk(chunk: ChunkPayload, sample_rate: int) -> np.ndarray:
    audio_bytes = base64.b64decode(chunk.audio_b64)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
    return np.asarray(data, dtype=np.float32)


def _encode_wav(audio: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


__all__ = ["convert_manifest"]
