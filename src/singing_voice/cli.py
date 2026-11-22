"""Command line interface for the singing-voice pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .convert_cli import register as register_convert
from .score_cli import register as register_score
from .preprocess import PreprocessorConfig, preprocess_audio_file
from .sibilance import (
    MIN_SAMPLE_RATE,
    analyze_sibilance,
    analyze_with_openai,
    build_local_recommendations,
    write_sibilance_report,
)
from .stitch import StitchConfig, stitch_manifest_to_file
from .workdir import DEFAULT_WORKDIR, SessionPaths

app = typer.Typer(help="Librosa preprocessing + Seed-VC stitching utilities.")
register_convert(app)
register_score(app)


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


def _same_file(a: Path, b: Path) -> bool:
    if a == b:
        return True
    try:
        return a.samefile(b)
    except OSError:
        return False


@app.command("analyze-sibilance")
def analyze_sibilance_cmd(
    source_audio: Path = typer.Argument(..., help="Voice recording to shape."),
    target_audio: Path = typer.Argument(..., help="Reference voice recording."),
    sample_rate: int = typer.Option(
        44_100,
        min=MIN_SAMPLE_RATE,
        help=f"Analysis sample rate in Hz (at least {MIN_SAMPLE_RATE} to cover the 4-10 kHz band).",
    ),
    output_path: Path = typer.Option(
        DEFAULT_WORKDIR / "sibilance_report.json",
        "--output",
        help="Where to write the analysis JSON.",
    ),
    use_openai: bool = typer.Option(
        False,
        "--openai",
        help="Request extra recommendations from OpenAI using local metrics.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        envvar="OPENAI_API_KEY",
        help="OpenAI API key (or set OPENAI_API_KEY).",
    ),
    model: str = typer.Option("gpt-4.1-mini", help="OpenAI model for remote recommendations."),
    timeout: int = typer.Option(60, help="OpenAI HTTP timeout in seconds."),
):
    """Analyze and compare sibilance between two voice recordings."""

    source_audio = source_audio.expanduser().resolve()
    target_audio = target_audio.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    for audio in (source_audio, target_audio):
        if _same_file(output_path, audio):
            raise typer.BadParameter(
                f"Report path would overwrite input recording {audio}.",
                param_hint="--output",
            )

    source_metrics = analyze_sibilance(source_audio, sample_rate=sample_rate)
    target_metrics = analyze_sibilance(target_audio, sample_rate=sample_rate)
    local_recommendations = build_local_recommendations(source_metrics, target_metrics)

    openai_recommendations = None
    if use_openai:
        if not api_key:
            raise typer.BadParameter(
                "OPENAI_API_KEY is required when using --openai.",
                param_hint="--openai",
            )
        openai_recommendations = analyze_with_openai(
            source=source_metrics,
            target=target_metrics,
            local_recommendations=local_recommendations,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )

    write_sibilance_report(
        output_path,
        source=source_metrics,
        target=target_metrics,
        local_recommendations=local_recommendations,
        openai_recommendations=openai_recommendations,
    )
    typer.echo(f"Wrote sibilance report to {output_path}")


if __name__ == "__main__":
    app()
