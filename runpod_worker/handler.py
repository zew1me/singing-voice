"""RunPod serverless entrypoint for the Seed-VC worker."""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, cast

try:  # pragma: no cover - runpod only exists inside the worker image.
    import runpod  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - keep local linting light.
    runpod = cast(Any, None)

try:  # pragma: no cover - requests is only required inside the worker image.
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency locally.
    requests = None  # type: ignore[assignment]

from singing_voice.manifests import PreprocessManifest
from runpod_worker.seedvc_worker import convert_manifest

TARGET_VOICE_ENV = "TARGET_VOICE_PATH"
MODEL_PATH_ENV = "SEEDVC_MODEL_PATH"
DEFAULT_TARGET_NAME = "target_voice.wav"
DEFAULT_MODEL_NAME = "seedvc_model.pt"


def _extract_inputs(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    payload = event.get("input")
    if isinstance(payload, dict):
        return payload
    return event


def _extract_manifest(inputs: Dict[str, Any]) -> PreprocessManifest:
    manifest_payload: Optional[Dict[str, Any]] = None
    candidate = inputs.get("manifest")
    if isinstance(candidate, dict):
        manifest_payload = candidate
    elif "chunks" in inputs and "sample_rate" in inputs:
        manifest_payload = inputs

    if not manifest_payload:
        raise ValueError("Input payload must contain a Step A manifest under 'manifest'.")

    return PreprocessManifest.from_dict(manifest_payload)


def _download_file(url: str, destination: Path) -> None:
    if requests is None:  # pragma: no cover - guarded for local linting.
        raise RuntimeError("requests is required inside the worker image to download files.")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _resolve_asset(
    *,
    env_var: str,
    b64_key: str,
    url_key: str,
    default_name: str,
    inputs: Dict[str, Any],
    tmp_dir: Path,
) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        env_path = Path(env_value)
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"{env_var} points to a missing file: {env_path}")

    b64_value = inputs.get(b64_key)
    if isinstance(b64_value, str):
        destination = tmp_dir / default_name
        destination.write_bytes(base64.b64decode(b64_value))
        return destination

    url_value = inputs.get(url_key)
    if isinstance(url_value, str):
        destination = tmp_dir / default_name
        _download_file(url_value, destination)
        return destination

    raise ValueError(
        f"Provide '{b64_key}' or '{url_key}', or set the {env_var} environment variable."
    )


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inputs = _extract_inputs(event)

    try:
        manifest = _extract_manifest(inputs)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            target_voice_path = _resolve_asset(
                env_var=TARGET_VOICE_ENV,
                b64_key="target_voice_b64",
                url_key="target_voice_url",
                default_name=DEFAULT_TARGET_NAME,
                inputs=inputs,
                tmp_dir=tmp_dir,
            )
            model_path = _resolve_asset(
                env_var=MODEL_PATH_ENV,
                b64_key="model_b64",
                url_key="model_url",
                default_name=DEFAULT_MODEL_NAME,
                inputs=inputs,
                tmp_dir=tmp_dir,
            )

            converted = convert_manifest(manifest, target_voice_path, model_path)
            return {"status": "success", "converted_manifest": converted.to_dict()}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


if runpod:
    runpod.serverless.start({"handler": handler})
