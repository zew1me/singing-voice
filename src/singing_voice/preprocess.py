"""Step A of the pipeline: trim, chunk, and serialize audio with librosa."""
from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from .manifests import ChunkPayload, PreprocessManifest, save_manifest


@dataclass
class PreprocessorConfig:
    sample_rate: int = 16_000
    chunk_seconds: float = 12.0
    overlap_seconds: float = 1.5
    silence_threshold_db: float = -40.0
    min_chunk_seconds: float = 0.15
    frame_length: int = 1024
    hop_length: int = 256


class SilenceDetector:
    """Encapsulates librosa-based silence trimming."""

    def __init__(self, threshold_db: float, frame_length: int, hop_length: int) -> None:
        self.threshold_db = threshold_db
        self.frame_length = frame_length
        self.hop_length = hop_length

    def trim_bounds(self, audio: np.ndarray) -> Optional[Tuple[int, int]]:
        if audio.size == 0:
            return None

        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=True,
        )[0]
        if not np.any(rms):
            return None

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        mask = rms_db > self.threshold_db
        if not np.any(mask):
            return None

        frames = np.flatnonzero(mask)
        start = librosa.frames_to_samples(frames[0], hop_length=self.hop_length)
        end = librosa.frames_to_samples(frames[-1], hop_length=self.hop_length) + self.frame_length
        return max(start, 0), min(int(end), audio.shape[-1])


def preprocess_audio_file(
    audio_path: Path,
    manifest_path: Path,
    config: Optional[PreprocessorConfig] = None,
) -> PreprocessManifest:
    """Load ``audio_path`` and write a manifest that RunPod pods can consume."""

    cfg = config or PreprocessorConfig()
    audio, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    detector = SilenceDetector(cfg.silence_threshold_db, cfg.frame_length, cfg.hop_length)

    chunk_payloads = _chunk_audio(audio, cfg, detector)
    manifest = PreprocessManifest(
        sample_rate=cfg.sample_rate,
        source=str(audio_path.resolve()),
        chunk_seconds=cfg.chunk_seconds,
        overlap_seconds=cfg.overlap_seconds,
        created_at=datetime.utcnow().isoformat(),
        chunks=chunk_payloads,
    )

    save_manifest(manifest, manifest_path)
    return manifest


def _chunk_audio(
    audio: np.ndarray,
    cfg: PreprocessorConfig,
    detector: SilenceDetector,
) -> List[ChunkPayload]:
    chunk_size = int(cfg.chunk_seconds * cfg.sample_rate)
    overlap = int(cfg.overlap_seconds * cfg.sample_rate)
    min_len = int(cfg.min_chunk_seconds * cfg.sample_rate)

    cursor = 0
    total = len(audio)
    chunks: List[ChunkPayload] = []

    while cursor < total:
        start = max(cursor - overlap, 0)
        end = min(cursor + chunk_size, total)
        segment = audio[start:end]

        trim = detector.trim_bounds(segment)
        if trim is None:
            cursor += chunk_size
            continue

        trim_start, trim_end = trim
        trimmed = segment[trim_start:trim_end]
        if trimmed.size < min_len:
            cursor += chunk_size
            continue

        chunk_id = uuid.uuid4().hex
        absolute_start = start + trim_start
        absolute_end = start + trim_end

        wav_b64 = _encode_wav(trimmed, cfg.sample_rate)
        chunks.append(
            ChunkPayload(
                chunk_id=chunk_id,
                start=int(absolute_start),
                end=int(absolute_end),
                duration=float(trimmed.size / cfg.sample_rate),
                audio_b64=wav_b64,
            )
        )

        cursor += chunk_size

    return chunks


def _encode_wav(audio: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


__all__ = ["PreprocessorConfig", "preprocess_audio_file"]
