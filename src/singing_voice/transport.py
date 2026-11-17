"""Audio in and out of the RunPod worker: inline FLAC (base64) or URLs.

A job names each audio input as either ``{"b64": "<base64 audio file>"}`` or
``{"url": "https://..."}`` (for example a presigned GET). The result comes back inline as
base64 FLAC, or is PUT to ``output.put_url`` (a presigned PUT) when the caller gives one, so
the worker never holds storage credentials.

RunPod caps a ``/run`` request at 10 MB, and results are capped at about the same size, so
inline transport suits excerpts; full songs need URLs.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

RUN_PAYLOAD_LIMIT_BYTES = 10_000_000
# Leave room for the rest of the JSON envelope.
INLINE_AUDIO_LIMIT_BYTES = 9_500_000
DOWNLOAD_TIMEOUT_S = 300


class TransportError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def encode_flac_b64(audio: np.ndarray, sample_rate: int, subtype: str = "PCM_24") -> str:
    """Base64 of a FLAC file. Samples outside [-1, 1] are clipped (FLAC is integer PCM)."""
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0), sample_rate, format="FLAC", subtype=subtype)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_audio_bytes(data: bytes, label: str = "audio") -> Tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises several types for unreadable data
        raise TransportError("BAD_AUDIO", f"{label} is not a readable audio file: {exc}") from exc
    if audio.size == 0:
        raise TransportError("BAD_AUDIO", f"{label} has no samples")
    return audio, int(sample_rate)


def decode_audio_b64(value: str, label: str = "audio") -> Tuple[np.ndarray, int]:
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise TransportError("BAD_BASE64", f"{label}.b64 is not valid base64") from exc
    return decode_audio_bytes(data, label)


def read_audio_spec(spec: Any, label: str) -> Tuple[np.ndarray, int]:
    """Decode ``{"b64": ...}`` or download ``{"url": ...}``."""
    if not isinstance(spec, dict) or not ({"b64", "url"} & set(spec)):
        raise TransportError("MISSING_INPUT", f"'{label}' must be an object with 'b64' or 'url'")
    if isinstance(spec.get("b64"), str):
        return decode_audio_b64(spec["b64"], label)
    url = spec.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise TransportError("BAD_URL", f"{label}.url must be an http(s) URL")
    import requests  # noqa: PLC0415 - only the worker and the client need it

    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TransportError("DOWNLOAD_FAILED", f"could not download {label}: {exc}") from exc
    return decode_audio_bytes(response.content, label)


def deliver_audio(audio: np.ndarray, sample_rate: int, output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """PUT FLAC to ``output['put_url']`` if given, else return it inline."""
    flac_b64 = encode_flac_b64(audio, sample_rate)
    put_url = (output or {}).get("put_url")
    if put_url:
        import requests  # noqa: PLC0415

        try:
            response = requests.put(
                put_url,
                data=base64.b64decode(flac_b64),
                headers={"Content-Type": "audio/flac"},
                timeout=DOWNLOAD_TIMEOUT_S,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TransportError("UPLOAD_FAILED", f"could not PUT the result: {exc}") from exc
        return {"output_url": put_url.split("?", 1)[0], "format": "flac"}
    if len(flac_b64) > INLINE_AUDIO_LIMIT_BYTES:
        raise TransportError(
            "OUTPUT_TOO_LARGE",
            f"result is {len(flac_b64) / 1e6:.1f} MB as base64 FLAC, over RunPod's result limit; "
            "send output.put_url (a presigned PUT) or a shorter source",
        )
    return {"audio_b64": flac_b64, "format": "flac"}


__all__ = [
    "INLINE_AUDIO_LIMIT_BYTES",
    "RUN_PAYLOAD_LIMIT_BYTES",
    "TransportError",
    "decode_audio_b64",
    "decode_audio_bytes",
    "deliver_audio",
    "encode_flac_b64",
    "read_audio_spec",
]
