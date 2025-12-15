"""Typed helpers for the JSON exchanged between the pipeline steps."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class ChunkPayload:
    chunk_id: str
    start: int
    end: int
    duration: float
    audio_b64: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkPayload":
        return cls(
            chunk_id=data["chunk_id"],
            start=int(data["start"]),
            end=int(data["end"]),
            duration=float(data.get("duration", 0.0)),
            audio_b64=data["audio_b64"],
        )


@dataclass
class PreprocessManifest:
    sample_rate: int
    source: str
    chunk_seconds: float
    overlap_seconds: float
    created_at: str
    chunks: List[ChunkPayload] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "source": self.source,
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "created_at": self.created_at,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessManifest":
        return cls(
            sample_rate=int(data["sample_rate"]),
            source=str(data.get("source", "")),
            chunk_seconds=float(data.get("chunk_seconds", 0.0)),
            overlap_seconds=float(data.get("overlap_seconds", 0.0)),
            created_at=str(data.get("created_at", datetime.utcnow().isoformat())),
            chunks=[ChunkPayload.from_dict(item) for item in data.get("chunks", [])],
        )


@dataclass
class ConvertedManifest:
    sample_rate: int
    converted_chunks: List[ChunkPayload]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "converted_chunks": [chunk.to_dict() for chunk in self.converted_chunks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvertedManifest":
        return cls(
            sample_rate=int(data["sample_rate"]),
            converted_chunks=[ChunkPayload.from_dict(item) for item in data.get("converted_chunks", [])],
        )


def save_manifest(manifest: PreprocessManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(manifest.to_dict()))
    return path


def save_converted_manifest(manifest: ConvertedManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(manifest.to_dict()))
    return path


def _json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)


def load_manifest(path: Path) -> PreprocessManifest:
    import json

    data = json.loads(path.read_text())
    if "converted_chunks" in data and "chunks" not in data:
        # If a converted manifest is passed to the preprocessor loader we
        # still return a preprocess manifest so downstream tooling can inspect
        # original metadata.
        return PreprocessManifest(
            sample_rate=int(data["sample_rate"]),
            source=str(path),
            chunk_seconds=float(data.get("chunk_seconds", 0.0)),
            overlap_seconds=float(data.get("overlap_seconds", 0.0)),
            created_at=str(data.get("created_at", "")),
            chunks=[ChunkPayload.from_dict(item) for item in data.get("converted_chunks", [])],
        )
    return PreprocessManifest.from_dict(data)


def load_converted_manifest(path: Path) -> ConvertedManifest:
    import json

    data = json.loads(path.read_text())
    if "converted_chunks" not in data:
        raise ValueError("Expected a converted manifest with 'converted_chunks'")
    return ConvertedManifest.from_dict(data)


__all__ = [
    "ChunkPayload",
    "PreprocessManifest",
    "ConvertedManifest",
    "save_manifest",
    "save_converted_manifest",
    "load_manifest",
    "load_converted_manifest",
]
