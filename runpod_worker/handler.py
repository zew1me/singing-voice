"""RunPod serverless entrypoint: sing a vocal in a reference voice with Seed-VC.

Job input::

    {
      "source":    {"b64": "<base64 audio file>"} | {"url": "https://..."},
      "reference": {"b64": "..."} | {"url": "..."},
      "output":    {"put_url": "https://..."},            # optional; default is inline FLAC
      "options":   {"diffusion_steps": 40, "inference_cfg_rate": 0.7, "semitone_shift": 0}
    }

Success output: ``sample_rate`` (44100), ``num_samples`` (always the source's duration at
44.1 kHz, so the result lines up with the input), ``audio_b64`` or ``output_url``, the
options used, engine details and timings. Failure output: ``{"error": "CODE: message"}``,
which RunPod reports as a failed job.

The engine (several GB of weights) is loaded once per worker, at startup.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - runpod only exists inside the worker image.
    import runpod  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    runpod = None

from runpod_worker.engine import (
    OUTPUT_SAMPLE_RATE,
    ConvertOptions,
    EngineError,
    SeedVCEngine,
)
from singing_voice.transport import TransportError, deliver_audio, read_audio_spec

SEED_VC_HOME_ENV = "SEED_VC_HOME"
CHECKPOINT_ENV = "SEEDVC_CHECKPOINT"
CONFIG_ENV = "SEEDVC_CONFIG"
SEED_VC_REF_ENV = "SEED_VC_REF"

_ENGINE: Optional[SeedVCEngine] = None


def get_engine() -> SeedVCEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SeedVCEngine(
            os.environ.get(SEED_VC_HOME_ENV, "/opt/seed-vc"),
            checkpoint=os.environ.get(CHECKPOINT_ENV) or None,
            config=os.environ.get(CONFIG_ENV) or None,
        )
    return _ENGINE


def handler(event: Dict[str, Any], engine_factory: Callable[[], Any] = get_engine) -> Dict[str, Any]:
    inputs = event.get("input") if isinstance(event, dict) else None
    if not isinstance(inputs, dict):
        return {"error": "MISSING_INPUT: job has no 'input' object"}
    started = time.time()
    try:
        options = ConvertOptions.from_dict(inputs.get("options"))
        source, source_sr = read_audio_spec(inputs.get("source"), "source")
        reference, reference_sr = read_audio_spec(inputs.get("reference"), "reference")
        loaded = time.time()
        engine = engine_factory()
        audio = engine.convert(source, source_sr, reference, reference_sr, options)
        converted = time.time()
        delivered = deliver_audio(audio, OUTPUT_SAMPLE_RATE, inputs.get("output"))
    except (EngineError, TransportError) as exc:
        return {"error": f"{exc.code}: {exc}"}

    source_seconds = len(source) / source_sr
    convert_seconds = converted - loaded
    return {
        "status": "success",
        "sample_rate": OUTPUT_SAMPLE_RATE,
        "num_samples": int(len(audio)),
        "duration_s": round(len(audio) / OUTPUT_SAMPLE_RATE, 3),
        **delivered,
        "options": options.to_dict(),
        "levels": dict(getattr(engine, "last_report", {}) or {}),
        "engine": {**engine.describe(), "seed_vc_ref": os.environ.get(SEED_VC_REF_ENV)},
        "timing_s": {
            "inputs": round(loaded - started, 2),
            "convert": round(convert_seconds, 2),
            "output": round(time.time() - converted, 2),
            "realtime_factor": round(convert_seconds / source_seconds, 3) if source_seconds else None,
        },
    }


if __name__ == "__main__" and runpod is not None:  # pragma: no cover - worker image only
    get_engine()
    runpod.serverless.start({"handler": handler})
