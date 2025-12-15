"""Helpers for interacting with the Seed-VC RunPod worker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .manifests import ConvertedManifest, save_converted_manifest


def submit_to_runpod(
    manifest_path: Path,
    endpoint: str,
    api_key: Optional[str] = None,
    timeout: int = 600,
    output_path: Optional[Path] = None,
) -> ConvertedManifest:
    """Send ``manifest_path`` to the Seed-VC endpoint and return the response."""

    payload = json.loads(manifest_path.read_text())
    response_data = _post_json(endpoint, payload=payload, api_key=api_key, timeout=timeout)
    converted = ConvertedManifest.from_dict(response_data)

    if output_path:
        save_converted_manifest(converted, output_path)
    return converted


def _post_json(endpoint: str, payload: dict, api_key: Optional[str], timeout: int) -> dict:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional dependency hook
        raise RuntimeError(
            "The optional 'runpod' dependency group is required for HTTP submissions."
        ) from exc

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


__all__ = ["submit_to_runpod"]
