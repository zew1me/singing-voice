"""Sibilance-focused analysis helpers for vocal matching workflows."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np

SIBILANCE_BAND_HZ = (4_000.0, 10_000.0)
# The full sibilance band must sit below Nyquist for the ratio to mean the same thing at every rate.
MIN_SAMPLE_RATE = 20_000
# Frames more than this far below the loudest frame count as silence.
ACTIVE_FRAME_FLOOR_DB = -60.0


class OpenAIRecommendationError(RuntimeError):
    """The optional OpenAI enrichment failed; the local analysis is still valid."""


@dataclass
class SibilanceMetrics:
    file: str
    duration_seconds: float
    sample_rate: int
    sibilance_ratio_db: float
    sibilant_frame_ratio: float
    sibilance_peak_hz: float
    sibilant_centroid_hz: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "sibilance_ratio_db": self.sibilance_ratio_db,
            "sibilant_frame_ratio": self.sibilant_frame_ratio,
            "sibilance_peak_hz": self.sibilance_peak_hz,
            "sibilant_centroid_hz": self.sibilant_centroid_hz,
        }


def analyze_sibilance(path: Path, sample_rate: int = 44_100) -> SibilanceMetrics:
    if sample_rate < MIN_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate {sample_rate} Hz cannot represent the {SIBILANCE_BAND_HZ[0]:.0f}-"
            f"{SIBILANCE_BAND_HZ[1]:.0f} Hz sibilance band; use at least {MIN_SAMPLE_RATE} Hz."
        )
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file has no samples: {path}")

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    energy = np.square(magnitude)
    total_energy_per_frame = np.sum(energy, axis=0)
    # Silent frames have no spectrum to compare; left in, they read as all-sibilance or skew the quantiles.
    active = total_energy_per_frame > max(
        float(np.max(total_energy_per_frame)) * 10 ** (ACTIVE_FRAME_FLOOR_DB / 10), 1e-10
    )
    if not np.any(active):
        raise ValueError(f"No sibilance estimate: audio is silent: {path}")
    magnitude = magnitude[:, active]
    energy = energy[:, active]
    total_energy_per_frame = total_energy_per_frame[active]

    sib_mask = (freqs >= SIBILANCE_BAND_HZ[0]) & (freqs <= SIBILANCE_BAND_HZ[1])
    sib_energy_per_frame = np.sum(energy[sib_mask, :], axis=0)

    sib_ratio = np.clip(sib_energy_per_frame / total_energy_per_frame, 1e-12, 1.0)
    sibilance_ratio_db = float(10.0 * np.log10(np.mean(sib_ratio)))

    sibilant_frames = sib_ratio > np.quantile(sib_ratio, 0.8)
    sibilant_frame_ratio = float(np.mean(sibilant_frames))

    sib_profile = np.mean(magnitude[sib_mask, :], axis=1)
    sib_freqs = freqs[sib_mask]
    peak_idx = int(np.argmax(sib_profile))
    sibilance_peak_hz = float(sib_freqs[peak_idx])

    centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    if np.any(sibilant_frames):
        sibilant_centroid_hz = float(np.median(centroids[sibilant_frames]))
    else:
        sibilant_centroid_hz = float(np.median(centroids))

    return SibilanceMetrics(
        file=str(path),
        duration_seconds=float(audio.size / sr),
        sample_rate=sr,
        sibilance_ratio_db=sibilance_ratio_db,
        sibilant_frame_ratio=sibilant_frame_ratio,
        sibilance_peak_hz=sibilance_peak_hz,
        sibilant_centroid_hz=sibilant_centroid_hz,
    )


def build_local_recommendations(source: SibilanceMetrics, target: SibilanceMetrics) -> Dict[str, Any]:
    delta_db = target.sibilance_ratio_db - source.sibilance_ratio_db
    peak_delta = target.sibilance_peak_hz - source.sibilance_peak_hz
    centroid_delta = target.sibilant_centroid_hz - source.sibilant_centroid_hz

    if delta_db > 1.5:
        gain_hint = "Increase source sibilance presence by ~2-4 dB around target peak."
    elif delta_db < -1.5:
        gain_hint = "Reduce source sibilance by ~2-5 dB using a de-esser around source peak."
    else:
        gain_hint = "Sibilance level is close; use only subtle EQ/de-ess moves."

    q_hint = "Use Q around 3-6 for surgical EQ moves."
    timing_hint = "For sharper S, use faster attack; for smoother tails, use slower release."

    return {
        "delta_sibilance_ratio_db": round(delta_db, 3),
        "delta_peak_hz": round(peak_delta, 2),
        "delta_centroid_hz": round(centroid_delta, 2),
        "source_peak_hz": round(source.sibilance_peak_hz, 2),
        "target_peak_hz": round(target.sibilance_peak_hz, 2),
        "suggested_actions": [gain_hint, q_hint, timing_hint],
    }


def analyze_with_openai(
    source: SibilanceMetrics,
    target: SibilanceMetrics,
    local_recommendations: Dict[str, Any],
    api_key: str,
    model: str = "gpt-4.1-mini",
    timeout: int = 60,
) -> str:
    try:
        import requests
    except ImportError as exc:
        raise OpenAIRecommendationError("Install optional dependency group with requests to call OpenAI.") from exc

    prompt = {
        "task": "Recommend practical Ableton + iZotope settings to make source vocal sibilance match target vocal.",
        "constraints": [
            "Assume Ableton Live 12 Standard and iZotope Nectar/Neutron/RX are available.",
            "Give concrete frequency bands, gain moves, and de-esser settings.",
            "Prefer actionable steps and avoid vague advice.",
        ],
        "source_metrics": source.to_dict(),
        "target_metrics": target.to_dict(),
        "local_recommendations": local_recommendations,
    }

    try:
        response = requests.post(
        "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": json.dumps(prompt, indent=2),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise OpenAIRecommendationError(f"OpenAI request failed: {exc}") from exc
    return _extract_output_text(payload)


def _extract_output_text(payload: Any) -> str:
    """Pull the message text out of a Responses API payload, rejecting malformed shapes."""
    if not isinstance(payload, dict):
        raise OpenAIRecommendationError("OpenAI response was not a JSON object.")

    status = payload.get("status")
    if status is not None and status != "completed":
        details = payload.get("incomplete_details") or payload.get("error")
        reason = details.get("reason") or details.get("message") if isinstance(details, dict) else None
        raise OpenAIRecommendationError(
            f"OpenAI response status was {status!r}" + (f" ({reason})" if reason else "") + "; no recommendations used."
        )

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output", [])
    if not isinstance(output, list):
        raise OpenAIRecommendationError("OpenAI response 'output' was not a list.")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise OpenAIRecommendationError("OpenAI response output item was not an object.")
        if item.get("type", "message") != "message":
            continue  # reasoning, tool calls, ...
        content = item.get("content", [])
        if not isinstance(content, list):
            raise OpenAIRecommendationError("OpenAI response message content was not a list.")
        for part in content:
            if not isinstance(part, dict):
                raise OpenAIRecommendationError("OpenAI response content part was not an object.")
            if part.get("type", "output_text") != "output_text":
                continue  # refusals and other non-text parts
            text = part.get("text")
            if not isinstance(text, str):
                raise OpenAIRecommendationError("OpenAI response output_text part had no string text.")
            if text:
                parts.append(text)
    text = "\n".join(parts).strip()
    if text:
        return text

    raise OpenAIRecommendationError("OpenAI response did not include text output.")


def write_sibilance_report(
    path: Path,
    source: SibilanceMetrics,
    target: SibilanceMetrics,
    local_recommendations: Dict[str, Any],
    openai_recommendations: Optional[str] = None,
    openai_error: Optional[str] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "source": source.to_dict(),
        "target": target.to_dict(),
        "local_recommendations": local_recommendations,
    }
    if openai_recommendations:
        payload["openai_recommendations"] = openai_recommendations
    if openai_error:
        payload["openai_error"] = openai_error
    path.write_text(json.dumps(payload, indent=2))
    return path


__all__ = [
    "MIN_SAMPLE_RATE",
    "OpenAIRecommendationError",
    "SibilanceMetrics",
    "analyze_sibilance",
    "build_local_recommendations",
    "analyze_with_openai",
    "write_sibilance_report",
]
