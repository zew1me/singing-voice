"""``svtool convert``: one command from a sung vocal to the same take in a reference voice."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .convert_client import ConvertRequest, build_payload, next_revision_path, run_job, save_result
from .transport import TransportError
from .workdir import DEFAULT_WORKDIR


def convert(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Sung vocal (a Demucs stem is fine)."),
    reference: Path = typer.Option(..., "--reference", exists=True, dir_okay=False, help="Clean clip of the target voice, 1-25 s used."),
    endpoint_id: str = typer.Option(..., envvar="RUNPOD_ENDPOINT_ID", help="RunPod serverless endpoint id."),
    api_key: str = typer.Option(..., envvar="RUNPOD_API_KEY", help="RunPod API key."),
    start_s: float = typer.Option(0.0, "--start-s", help="Excerpt start in the source, seconds."),
    duration_s: Optional[float] = typer.Option(None, "--duration-s", help="Excerpt length, seconds (inline transport fits about 60 s)."),
    diffusion_steps: int = typer.Option(40, help="Seed-VC diffusion steps; upstream suggests 30-50 for singing."),
    inference_cfg_rate: float = typer.Option(0.7, help="Classifier-free guidance rate (0-1)."),
    semitone_shift: int = typer.Option(0, help="Pitch shift in semitones; whole octaves only (-24, -12, 0, 12, 24)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output WAV. Default: working/<source>_seedvc_rNN.wav."),
    timeout_s: float = typer.Option(1800, "--timeout-s", help="Give up (and cancel the job) after this many seconds."),
) -> None:
    """Sing SOURCE in the voice of --reference on the RunPod Seed-VC endpoint."""
    source, reference = source.expanduser().resolve(), reference.expanduser().resolve()
    if out is None:
        out = next_revision_path(DEFAULT_WORKDIR.resolve(), f"{source.stem}_seedvc")
    out = out.expanduser().resolve()
    if out.exists():
        raise typer.BadParameter(f"{out} exists; outputs are never overwritten", param_hint="--out")

    request = ConvertRequest(
        source=source,
        reference=reference,
        options={
            "diffusion_steps": diffusion_steps,
            "inference_cfg_rate": inference_cfg_rate,
            "semitone_shift": semitone_shift,
        },
        start_s=start_s,
        duration_s=duration_s,
    )
    try:
        payload = build_payload(request)
    except TransportError as exc:
        raise typer.BadParameter(str(exc)) from exc

    def report(state: str, elapsed: float) -> None:
        typer.echo(f"  {state} after {elapsed:.0f} s", err=True)

    status = run_job(endpoint_id, api_key, payload, timeout_s=timeout_s, on_status=report)
    output = status.get("output") or {}
    if status.get("status") != "COMPLETED" or "error" in output:
        typer.echo(json.dumps({"ok": False, "status": status.get("status"), "error": output.get("error") or status.get("error")}))
        raise typer.Exit(1)

    meta = save_result(output, out)
    meta["source_excerpt"] = {"path": str(source), "start_s": start_s, "duration_s": duration_s}
    meta["reference"] = str(reference)
    meta["job"] = {"id": status.get("id"), "delay_ms": status.get("delayTime"), "execution_ms": status.get("executionTime")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    typer.echo(json.dumps({"ok": True, "output": str(out), **meta}))


def register(app: typer.Typer) -> None:
    app.command("convert")(convert)


__all__ = ["convert", "register"]
