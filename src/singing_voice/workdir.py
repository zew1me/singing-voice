"""Helpers for keeping every generated file inside a dedicated working directory."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_WORKDIR = Path("working")


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


@dataclass
class SessionPaths:
    """Small helper that keeps the important paths for a single run."""

    base: Path
    manifest: Path

    @classmethod
    def create(
        cls,
        label: str,
        workdir: Optional[Path] = None,
        manifest_name: Optional[str] = None,
    ) -> "SessionPaths":
        root = Path(workdir or DEFAULT_WORKDIR)
        root.mkdir(parents=True, exist_ok=True)

        session_dir = root / f"{label}_{_timestamp()}"
        session_dir.mkdir(parents=True, exist_ok=True)

        manifest_filename = manifest_name or "manifest.json"
        manifest_path = session_dir / manifest_filename
        return cls(base=session_dir, manifest=manifest_path)


__all__ = ["SessionPaths", "DEFAULT_WORKDIR"]
