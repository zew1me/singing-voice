"""``svtool score`` and ``svtool bench``: judge conversions by numbers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .bench import TestSet, TestSetError, compare, format_table, load_summary, new_run_dir, rescore_run, run_bench
from .convert_client import ConvertRequest, convert_to_file
from .score import ScoreInputs, report_path, score, write_report
from .workdir import DEFAULT_WORKDIR

TIMBRE_HELP = "Speaker-embedding similarity (needs the score extra)."
WORDS_HELP = "Whisper transcripts and word error rate (needs the score extra)."
WHISPER_HELP = "faster-whisper model size: tiny, base, small, medium, large-v3."


def _load_models(timbre: bool, words: bool, whisper_model: str):
    from .score_models import load_models  # noqa: PLC0415 - keeps torch out of other commands

    models = load_models(timbre=timbre, words=words, whisper_model=whisper_model)
    for section, reason in models.skipped.items():
        typer.echo(f"  {section}: skipped ({reason})", err=True)
    return models


def score_command(
    output: Path = typer.Argument(..., exists=True, dir_okay=False, help="Converted WAV (from svtool convert)."),
    source: Optional[Path] = typer.Option(None, "--source", exists=True, dir_okay=False, help="Input vocal. Default: from OUTPUT's .json sidecar."),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, dir_okay=False, help="Reference voice. Default: from the sidecar."),
    start_s: Optional[float] = typer.Option(None, "--start-s", help="Excerpt start in the source, seconds. Default: from the sidecar, else 0."),
    duration_s: Optional[float] = typer.Option(None, "--duration-s", help="Excerpt length, seconds. Default: from the sidecar, else OUTPUT's length."),
    semitone_shift: Optional[int] = typer.Option(None, "--semitone-shift", help="Shift the conversion applied, semitones. Default: from the sidecar, else 0."),
    timbre: bool = typer.Option(True, "--timbre/--no-timbre", help=TIMBRE_HELP),
    words: bool = typer.Option(True, "--words/--no-words", help=WORDS_HELP),
    whisper_model: str = typer.Option("small", "--whisper-model", help=WHISPER_HELP),
    report: Optional[Path] = typer.Option(None, "--report", help="Report JSON. Default: <output>.score.json (replaced if present)."),
) -> None:
    """Score a converted vocal against its input and the reference voice."""
    output = output.expanduser().resolve()
    if source is None:
        try:
            inputs = ScoreInputs.from_sidecar(output)
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--source") from exc
    else:
        import soundfile as sf  # noqa: PLC0415

        inputs = ScoreInputs(output=output, source=source.expanduser().resolve(), duration_s=sf.info(str(output)).duration)
    if reference is not None:
        inputs.reference = reference.expanduser().resolve()
    if start_s is not None:
        inputs.start_s = start_s
    if duration_s is not None:
        inputs.duration_s = duration_s
    if semitone_shift is not None:
        inputs.semitone_shift = semitone_shift

    models = _load_models(timbre, words, whisper_model)
    result = score(inputs, models, timbre=timbre, words=words)
    path = write_report(result, (report or report_path(output)).expanduser().resolve())
    typer.echo(json.dumps({"ok": True, "report": str(path), "summary": result["summary"]}))


bench_app = typer.Typer(help="Convert and score a fixed test set; compare runs.")


@bench_app.command("run")
def bench_run(
    testset_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Test set JSON."),
    label: str = typer.Option(..., "--label", help="Short name for this run, e.g. baseline or steps50."),
    endpoint_id: Optional[str] = typer.Option(None, envvar="RUNPOD_ENDPOINT_ID", help="RunPod serverless endpoint id."),
    api_key: Optional[str] = typer.Option(None, envvar="RUNPOD_API_KEY", help="RunPod API key."),
    diffusion_steps: int = typer.Option(40, help="Seed-VC diffusion steps."),
    inference_cfg_rate: float = typer.Option(0.7, help="Classifier-free guidance rate (0-1)."),
    semitone_shift: int = typer.Option(0, help="Pitch shift in semitones; whole octaves only."),
    timeout_s: float = typer.Option(1800, "--timeout-s", help="Per-clip job timeout, seconds."),
    workdir: Path = typer.Option(DEFAULT_WORKDIR / "bench", "--workdir", help="Runs go in <workdir>/<timestamp>_<label>/."),
    timbre: bool = typer.Option(True, "--timbre/--no-timbre", help=TIMBRE_HELP),
    words: bool = typer.Option(True, "--words/--no-words", help=WORDS_HELP),
    whisper_model: str = typer.Option("small", "--whisper-model", help=WHISPER_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check the test set and stop."),
) -> None:
    """Convert every clip of TESTSET with the same options, score each, write summary.json."""
    try:
        testset = TestSet.load(testset_path)
    except TestSetError as exc:
        raise typer.BadParameter(str(exc), param_hint="TESTSET") from exc
    problems = testset.problems()
    for problem in problems:
        typer.echo(f"  {problem}", err=True)
    if problems:
        raise typer.Exit(1)
    if dry_run:
        typer.echo(json.dumps({"ok": True, "testset": testset.name, "clips": [c.id for c in testset.clips]}))
        return
    if not endpoint_id or not api_key:
        raise typer.BadParameter("set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (or pass --endpoint-id/--api-key)")

    options = {"diffusion_steps": diffusion_steps, "inference_cfg_rate": inference_cfg_rate, "semitone_shift": semitone_shift}
    run_dir = new_run_dir(workdir.expanduser().resolve(), label)
    models = _load_models(timbre, words, whisper_model)

    def convert_clip(clip, out: Path):
        request = ConvertRequest(clip.source, clip.reference, dict(options), start_s=clip.start_s, duration_s=clip.duration_s)
        return convert_to_file(
            request, out, endpoint_id, api_key, timeout_s=timeout_s,
            on_status=lambda state, elapsed: typer.echo(f"  {state} after {elapsed:.0f} s", err=True),
        )

    summary = run_bench(
        testset, run_dir, convert_clip,
        lambda inputs: score(inputs, models, timbre=timbre, words=words),
        label=label, options=options, log=lambda line: typer.echo(line, err=True),
    )
    typer.echo(json.dumps({"ok": True, "run": str(run_dir), "summary": summary}))


@bench_app.command("rescore")
def bench_rescore(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="A run directory from bench run."),
    timbre: bool = typer.Option(True, "--timbre/--no-timbre", help=TIMBRE_HELP),
    words: bool = typer.Option(True, "--words/--no-words", help=WORDS_HELP),
    whisper_model: str = typer.Option("small", "--whisper-model", help=WHISPER_HELP),
) -> None:
    """Score a run's converted clips again (e.g. after installing the score extra)."""
    models = _load_models(timbre, words, whisper_model)
    summary = rescore_run(
        run_dir.expanduser().resolve(),
        lambda inputs: score(inputs, models, timbre=timbre, words=words),
        log=lambda line: typer.echo(line, err=True),
    )
    typer.echo(json.dumps({"ok": True, "run": str(run_dir), "summary": summary}))


@bench_app.command("compare")
def bench_compare(
    before: Path = typer.Argument(..., exists=True, help="Baseline run directory (or its summary.json)."),
    after: Path = typer.Argument(..., exists=True, help="Run to judge against it."),
    as_json: bool = typer.Option(False, "--json", help="Print rows as JSON instead of a table."),
) -> None:
    """Per clip and metric: before, after, delta and whether it got better."""
    a, b = load_summary(before), load_summary(after)
    rows = compare(a, b)
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
    else:
        typer.echo(format_table(rows, str(a.get("label") or "before"), str(b.get("label") or "after")))


def register(app: typer.Typer) -> None:
    app.command("score")(score_command)
    app.add_typer(bench_app, name="bench")


__all__ = ["bench_app", "register", "score_command"]
