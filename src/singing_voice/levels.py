"""Level and spectrum measurements for scoring.

``active_rms_db`` is the same measure the worker uses to level its output
(``runpod_worker.engine.active_rms_db``): RMS over the 50 ms windows within 30 dB of the
loudest, i.e. the level while singing. It is repeated here because the worker image loads its
engine before this package is installed.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Vocal bands, Hz. Band levels are relative to the whole spectrum, so they describe tone
# colour, not loudness: a darker voice has more "low_mid" and less "presence".
BANDS_HZ = (
    ("low", 0.0, 250.0),
    ("low_mid", 250.0, 2_000.0),
    ("presence", 2_000.0, 5_000.0),
    ("sibilance", 5_000.0, 10_000.0),
    ("air", 10_000.0, 16_000.0),
)
CLIP_LEVEL = 0.999


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _db(value: float) -> float:
    return float(10 * np.log10(value))


def active_rms_db(audio: np.ndarray, sample_rate: int, window_s: float = 0.05, range_db: float = 30.0) -> Optional[float]:
    """RMS level (dBFS) over the 50 ms windows within ``range_db`` of the loudest."""
    audio = to_mono(audio)
    size = max(1, int(sample_rate * window_s))
    frames = len(audio) // size
    if frames == 0:
        return None
    power = np.mean(audio[: frames * size].reshape(frames, size).astype(np.float64) ** 2, axis=1)
    if not np.any(power > 0):
        return None
    loud = power[power >= power.max() * 10 ** (-range_db / 10)]
    return _db(np.mean(loud))


def level_summary(audio: np.ndarray, sample_rate: int) -> Dict[str, Optional[float]]:
    audio = to_mono(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    level = active_rms_db(audio, sample_rate)
    return {
        "duration_s": round(len(audio) / sample_rate, 3),
        "active_rms_db": None if level is None else round(level, 1),
        "peak_dbfs": None if peak <= 0 else round(20 * np.log10(peak), 1),
        "clipped_samples": int(np.sum(np.abs(audio) >= CLIP_LEVEL)),
    }


def band_profile(audio: np.ndarray, sample_rate: int, n_fft: int = 2048, range_db: float = 30.0) -> Optional[Dict[str, float]]:
    """Each band's share of the spectrum in dB, plus the spectral centroid, over the sung frames.

    Frames more than ``range_db`` below the loudest are left out so silences and breaths do not
    pull the profile toward the noise floor. Bands above Nyquist are omitted.
    """
    audio = to_mono(audio).astype(np.float64)
    hop = n_fft // 2
    if len(audio) < n_fft:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop]
    spectra = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    energy = spectra.sum(axis=1)
    if not np.any(energy > 0):
        return None
    active = spectra[energy >= energy.max() * 10 ** (-range_db / 10)]
    power = active.mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    total = power[freqs > 0].sum()
    profile: Dict[str, float] = {}
    for name, low, high in BANDS_HZ:
        if high > sample_rate / 2:
            continue
        band = power[(freqs > low) & (freqs <= high)].sum()
        profile[f"{name}_db"] = round(_db(max(band, 1e-20) / total), 1) + 0.0  # no "-0.0"
    profile["centroid_hz"] = round(float(np.sum(freqs * power) / power.sum()), 0)
    return profile


def profile_distance_db(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[float]:
    """RMS difference in dB over the bands both profiles have."""
    if not a or not b:
        return None
    keys = [k for k in a if k.endswith("_db") and k in b]
    if not keys:
        return None
    return round(float(np.sqrt(np.mean([(a[k] - b[k]) ** 2 for k in keys]))), 2)


__all__ = [
    "BANDS_HZ",
    "active_rms_db",
    "band_profile",
    "level_summary",
    "profile_distance_db",
    "to_mono",
]
