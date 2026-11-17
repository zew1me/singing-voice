"""Command line interface for the singing-voice pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .convert_cli import register as register_convert
from .preprocess import PreprocessorConfig, preprocess_audio_file
from .stitch import StitchConfig, stitch_manifest_to_file
from .workdir import DEFAULT_WORKDIR, SessionPaths

app = typer.Typer(help="Librosa preprocessing + Seed-VC stitching utilities.")
register_convert(app)


@app.command()
def preprocess(
    audio_path: Path = typer.Argument(..., help="Input WAV/AIFF/MP3 file."),
    manifest_path: Optional[Path] = typer.Option(
        None,
        "--manifest-path",
        help="Where to store the manifest JSON. Defaults to working/<stem>/manifest.json",
    ),
    workdir: Path = typer.Option(DEFAULT_WORKDIR, help="Root directory for generated files."),
    chunk_seconds: float = typer.Option(12.0, help="Nominal chunk duration in seconds."),
    overlap_seconds: float = typer.Option(1.5, help="Overlap duration between chunks."),
    silence_threshold_db: float = typer.Option(-40.0, help="dB threshold for detecting audio."),
    min_chunk_seconds: float = typer.Option(0.15, help="Minimum trimmed chunk length."),
    sample_rate: int = typer.Option(16_000, help="Sample rate for librosa.load and output files."),
):
    """Chunk + serialize audio so the RunPod worker can run inference."""

    audio_path = audio_path.expanduser().resolve()
    config = PreprocessorConfig(
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        silence_threshold_db=silence_threshold_db,
        min_chunk_seconds=min_chunk_seconds,
    )

    manifest_target = manifest_path
    if manifest_target is None:
        session = SessionPaths.create(audio_path.stem, workdir=workdir)
        manifest_target = session.manifest
    else:
        manifest_target = manifest_target.expanduser().resolve()
        manifest_target.parent.mkdir(parents=True, exist_ok=True)

    manifest = preprocess_audio_file(audio_path, manifest_target, config=config)
    typer.echo(
        f"Wrote manifest with {len(manifest.chunks)} chunks to {manifest_target}"
    )


@app.command()
def stitch(
    converted_manifest: Path = typer.Argument(..., help="JSON output from the Seed-VC worker."),
    output_path: Path = typer.Option(
        DEFAULT_WORKDIR / "stitched.wav",
        "--output",
        help="Where to place the stitched WAV file.",
    ),
    crossfade_seconds: float = typer.Option(1.5, help="Crossfade duration for overlaps."),
    sample_rate: int = typer.Option(16_000, help="Sample rate for the output WAV."),
):
    """Stitch converted chunks into a single WAV file."""

    cfg = StitchConfig(sample_rate=sample_rate, crossfade_seconds=crossfade_seconds)
    path = stitch_manifest_to_file(converted_manifest, output_path, config=cfg)
    typer.echo(f"Created {path}")


if __name__ == "__main__":
    app()
