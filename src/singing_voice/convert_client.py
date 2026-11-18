"""Client for the Seed-VC RunPod endpoint: send a vocal and a reference, save the result."""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf

from .transport import RUN_PAYLOAD_LIMIT_BYTES, TransportError, decode_audio_b64, encode_flac_b64

RUNPOD_API = "https://api.runpod.ai/v2"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


@dataclass
class ConvertRequest:
    source: Path
    reference: Path
    options: Dict[str, Any] = field(default_factory=dict)
    start_s: float = 0.0
    duration_s: Optional[float] = None


def read_excerpt(path: Path, start_s: float = 0.0, duration_s: Optional[float] = None) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    start = int(round(start_s * info.samplerate))
    if start >= info.frames:
        raise TransportError("BAD_EXCERPT", f"--start-s {start_s} is past the end of {path.name} ({info.duration:.1f} s)")
    frames = -1 if duration_s is None else int(round(duration_s * info.samplerate))
    audio, sample_rate = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def build_payload(request: ConvertRequest) -> Dict[str, Any]:
    """Job input with both audio files inline as FLAC. Refuses payloads over RunPod's /run limit."""
    source, source_sr = read_excerpt(request.source, request.start_s, request.duration_s)
    reference, reference_sr = read_excerpt(request.reference)
    payload = {
        "source": {"b64": encode_flac_b64(source, source_sr)},
        "reference": {"b64": encode_flac_b64(reference, reference_sr)},
        "options": dict(request.options),
    }
    size = len(json.dumps({"input": payload}))
    if size > RUN_PAYLOAD_LIMIT_BYTES:
        raise TransportError(
            "INPUT_TOO_LARGE",
            f"request is {size / 1e6:.1f} MB, over RunPod's 10 MB /run limit; use --duration-s for an "
            "excerpt (about 60 s fits) until URL transport is set up",
        )
    return payload


def _session(api_key: str):
    import requests  # noqa: PLC0415

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    return session


def run_job(
    endpoint_id: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: float = 1800,
    poll_s: float = 3.0,
    on_status: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Submit with /run and poll /status until the job finishes. Returns RunPod's status object."""
    session = _session(api_key)
    response = session.post(f"{RUNPOD_API}/{endpoint_id}/run", json={"input": payload}, timeout=120)
    response.raise_for_status()
    job_id = response.json()["id"]
    started = time.time()
    status: Dict[str, Any] = {"id": job_id, "status": "IN_QUEUE"}
    while time.time() - started < timeout_s:
        time.sleep(poll_s)
        response = session.get(f"{RUNPOD_API}/{endpoint_id}/status/{job_id}", timeout=120)
        response.raise_for_status()
        status = response.json()
        if on_status:
            on_status(status.get("status", "?"), time.time() - started)
        if status.get("status") in TERMINAL_STATES:
            return status
    session.post(f"{RUNPOD_API}/{endpoint_id}/cancel/{job_id}", timeout=60)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s:.0f} s; cancelled it")


def next_revision_path(directory: Path, base: str, suffix: str = ".wav") -> Path:
    """``<base>_rNN<suffix>`` with the next free NN; never an existing file."""
    pattern = re.compile(rf"^{re.escape(base)}_r(\d+){re.escape(suffix)}$")
    taken = [int(m.group(1)) for p in directory.glob(f"{base}_r*{suffix}") if (m := pattern.match(p.name))]
    return directory / f"{base}_r{max(taken, default=0) + 1:02d}{suffix}"


def save_result(output: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    """Write the returned audio as 24-bit WAV plus a JSON sidecar. Refuses to overwrite."""
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
    if "audio_b64" not in output:
        raise TransportError("NO_AUDIO", f"result has no inline audio (keys: {sorted(output)})")
    audio, sample_rate = decode_audio_b64(output["audio_b64"], "result")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate, subtype="PCM_24")
    meta = {k: v for k, v in output.items() if k != "audio_b64"}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class ConvertFailed(RuntimeError):
    def __init__(self, status: Optional[str], error: Any) -> None:
        super().__init__(f"job {status or 'failed'}: {error}")
        self.status = status
        self.error = error


def convert_to_file(
    request: ConvertRequest,
    out_path: Path,
    endpoint_id: str,
    api_key: str,
    timeout_s: float = 1800,
    on_status: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Run one conversion and write ``out_path`` plus a sidecar that names the source excerpt,
    the reference and the options, which is all ``svtool score`` needs."""
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
    payload = build_payload(request)
    status = run_job(endpoint_id, api_key, payload, timeout_s=timeout_s, on_status=on_status)
    output = status.get("output") or {}
    if status.get("status") != "COMPLETED" or not isinstance(output, dict) or "error" in output:
        error = output.get("error") if isinstance(output, dict) else output
        raise ConvertFailed(status.get("status"), error or status.get("error"))
    meta = save_result(output, out_path)
    meta["source_excerpt"] = {"path": str(request.source), "start_s": request.start_s, "duration_s": request.duration_s}
    meta["reference"] = str(request.reference)
    meta["job"] = {"id": status.get("id"), "delay_ms": status.get("delayTime"), "execution_ms": status.get("executionTime")}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


__all__ = [
    "ConvertFailed",
    "ConvertRequest",
    "convert_to_file",
    "build_payload",
    "next_revision_path",
    "read_excerpt",
    "run_job",
    "save_result",
]
