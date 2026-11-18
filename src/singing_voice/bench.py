"""A fixed test set and benchmark runs, so a pipeline change can be judged by its scores.

A test set is a JSON file naming a few vocal excerpts and the reference voice::

    {
      "name": "m1",
      "reference": "../test_files/reference.wav",
      "clips": [
        {"id": "clean", "kind": "clean", "source": "../test_files/clean.wav", "start_s": 30, "duration_s": 25},
        {"id": "bleed", "kind": "bleed", "source": "../test_files/stem.wav", "start_s": 60, "duration_s": 25}
      ]
    }

Paths are relative to the test set file. A clip may name its own ``reference``. A run converts
every clip with the same options into ``working/bench/<timestamp>_<label>/`` and writes
``<clip>.wav``, its convert sidecar ``<clip>.json``, ``<clip>.score.json`` and a
``summary.json`` that ``svtool bench compare`` reads.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .score import SUMMARY_METRICS, ScoreInputs, report_path

CLIP_KINDS = ("clean", "bleed", "harmonies", "other")
# Inline transport fits about a minute of vocal (RunPod's 10 MB /run cap).
MAX_CLIP_SECONDS = 60.0
SUMMARY_FILE = "summary.json"
RUN_FILE = "run.json"


class TestSetError(ValueError):
    __test__ = False  # not a pytest class


@dataclass
class Clip:
    id: str
    source: Path
    reference: Path
    duration_s: float
    start_s: float = 0.0
    kind: str = "other"
    notes: str = ""


@dataclass
class TestSet:
    __test__ = False  # not a pytest class

    name: str
    path: Path
    clips: List[Clip] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "TestSet":
        path = path.expanduser().resolve()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TestSetError(f"cannot read test set {path}: {exc}") from exc
        base = path.parent

        def resolve(value: Any, what: str) -> Path:
            if not isinstance(value, str) or not value:
                raise TestSetError(f"{what} must be a path")
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else base / candidate).resolve()

        default_reference = data.get("reference")
        clips: List[Clip] = []
        for index, raw in enumerate(data.get("clips") or []):
            where = f"clips[{index}]"
            if not isinstance(raw, dict):
                raise TestSetError(f"{where} must be an object")
            clip_id = raw.get("id")
            if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
                raise TestSetError(f"{where}.id must be letters, digits, '-' or '_' (it names files)")
            if clip_id in {c.id for c in clips}:
                raise TestSetError(f"duplicate clip id {clip_id!r}")
            kind = raw.get("kind", "other")
            if kind not in CLIP_KINDS:
                raise TestSetError(f"{where}.kind must be one of {CLIP_KINDS}")
            reference = raw.get("reference", default_reference)
            if reference is None:
                raise TestSetError(f"{where} has no reference and the test set has no default 'reference'")
            try:
                duration = float(raw["duration_s"])
                start = float(raw.get("start_s", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise TestSetError(f"{where} needs a numeric duration_s (and start_s, if given)") from exc
            if not 0 < duration <= MAX_CLIP_SECONDS:
                raise TestSetError(f"{where}.duration_s must be in (0, {MAX_CLIP_SECONDS:.0f}] s; inline transport fits about a minute")
            if start < 0:
                raise TestSetError(f"{where}.start_s must not be negative")
            clips.append(
                Clip(
                    id=clip_id,
                    source=resolve(raw.get("source"), f"{where}.source"),
                    reference=resolve(reference, f"{where}.reference"),
                    duration_s=duration,
                    start_s=start,
                    kind=kind,
                    notes=str(raw.get("notes", "")),
                )
            )
        if not clips:
            raise TestSetError(f"{path.name} has no clips")
        return cls(name=str(data.get("name") or path.stem), path=path, clips=clips)

    def problems(self) -> List[str]:
        """Missing files and excerpts that run past the end of their source."""
        import soundfile as sf  # noqa: PLC0415

        found: List[str] = []
        for clip in self.clips:
            for label, file in (("source", clip.source), ("reference", clip.reference)):
                if not file.is_file():
                    found.append(f"{clip.id}: {label} {file} does not exist")
            if clip.source.is_file():
                length = sf.info(str(clip.source)).duration
                if clip.start_s + clip.duration_s > length + 1e-3:
                    found.append(
                        f"{clip.id}: excerpt {clip.start_s:g}-{clip.start_s + clip.duration_s:g} s runs past the end "
                        f"of {clip.source.name} ({length:.1f} s)"
                    )
        return found


def new_run_dir(root: Path, label: str, now: Optional[datetime] = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "run"
    run_dir = root / f"{stamp}_{safe}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists")
    return run_dir


# convert(clip, out_path) -> convert sidecar dict; score(inputs) -> report dict
ConvertFn = Callable[[Clip, Path], Dict[str, Any]]
ScoreFn = Callable[[ScoreInputs], Dict[str, Any]]


def run_bench(
    testset: TestSet,
    run_dir: Path,
    convert: ConvertFn,
    score: ScoreFn,
    label: str,
    options: Dict[str, Any],
    log: Callable[[str], None] = lambda _: None,
) -> Dict[str, Any]:
    """Convert and score every clip. A failing clip is recorded and the run goes on."""
    run_dir.mkdir(parents=True, exist_ok=False)
    run_info = {
        "label": label,
        "testset": {"name": testset.name, "path": str(testset.path)},
        "options": options,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": {clip.id: {**asdict(clip), "source": str(clip.source), "reference": str(clip.reference)} for clip in testset.clips},
        "errors": {},
    }
    for clip in testset.clips:
        out = run_dir / f"{clip.id}.wav"
        log(f"[{clip.id}] converting {clip.source.name} {clip.start_s:g}+{clip.duration_s:g} s")
        try:
            convert(clip, out)
        except Exception as exc:  # noqa: BLE001 - one bad clip should not lose the others
            run_info["errors"][clip.id] = f"convert: {type(exc).__name__}: {exc}"
            log(f"[{clip.id}] convert failed: {exc}")
    (run_dir / RUN_FILE).write_text(json.dumps(run_info, indent=2) + "\n")
    return rescore_run(run_dir, score, log=log)


def rescore_run(run_dir: Path, score: ScoreFn, log: Callable[[str], None] = lambda _: None) -> Dict[str, Any]:
    """(Re)score every converted clip in a run and rewrite ``summary.json``."""
    run_info = json.loads((run_dir / RUN_FILE).read_text())
    errors = dict(run_info.get("errors") or {})
    for clip_id in run_info["clips"]:
        out = run_dir / f"{clip_id}.wav"
        if not out.is_file():
            continue
        errors.pop(clip_id, None)
        log(f"[{clip_id}] scoring")
        try:
            report = score(ScoreInputs.from_sidecar(out))
        except Exception as exc:  # noqa: BLE001
            errors[clip_id] = f"score: {type(exc).__name__}: {exc}"
            log(f"[{clip_id}] scoring failed: {exc}")
            continue
        report_path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    run_info["errors"] = errors
    (run_dir / RUN_FILE).write_text(json.dumps(run_info, indent=2) + "\n")
    summary = summarize_run(run_dir)
    (run_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    run_info = json.loads((run_dir / RUN_FILE).read_text())
    clips: Dict[str, Any] = {}
    for clip_id, clip in run_info["clips"].items():
        entry: Dict[str, Any] = {"kind": clip.get("kind")}
        report_file = report_path(run_dir / f"{clip_id}.wav")
        if clip_id in run_info.get("errors", {}):
            entry["error"] = run_info["errors"][clip_id]
        elif report_file.is_file():
            entry.update(json.loads(report_file.read_text()).get("summary") or {})
        else:
            entry["error"] = "not scored"
        clips[clip_id] = entry
    return {
        "label": run_info.get("label"),
        "testset": run_info.get("testset"),
        "options": run_info.get("options"),
        "created_at": run_info.get("created_at"),
        "clips": clips,
    }


def load_summary(path: Path) -> Dict[str, Any]:
    """A run directory or its ``summary.json``."""
    path = path.expanduser()
    if path.is_dir():
        path = path / SUMMARY_FILE
    return json.loads(path.read_text())


def _verdict(direction: Any, before: float, after: float) -> str:
    if direction == 0 or before == after:
        return ""
    if direction == "abs":
        better = abs(after) < abs(before)
    else:
        better = (after - before) * direction > 0
    return "better" if better else "worse"


def compare(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per clip and metric present in either run."""
    rows: List[Dict[str, Any]] = []
    clip_ids = list(before.get("clips", {})) + [c for c in after.get("clips", {}) if c not in before.get("clips", {})]
    for clip_id in clip_ids:
        a = before.get("clips", {}).get(clip_id, {})
        b = after.get("clips", {}).get(clip_id, {})
        for key, _, direction in SUMMARY_METRICS:
            va, vb = a.get(key), b.get(key)
            if va is None and vb is None:
                continue
            row = {"clip": clip_id, "metric": key, "before": va, "after": vb, "delta": None, "verdict": ""}
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                row["delta"] = round(vb - va, 4)
                row["verdict"] = _verdict(direction, va, vb)
            rows.append(row)
    return rows


def format_table(rows: List[Dict[str, Any]], before_label: str, after_label: str) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        return f"{value:g}" if isinstance(value, (int, float)) else str(value)

    header = ["clip", "metric", before_label, after_label, "delta", ""]
    body = [[r["clip"], r["metric"], cell(r["before"]), cell(r["after"]), "-" if r["delta"] is None else f"{r['delta']:+g}", r["verdict"]] for r in rows]
    widths = [max(len(str(line[i])) for line in [header, *body]) for i in range(len(header))]
    lines = ["  ".join(str(value).ljust(widths[i]) for i, value in enumerate(line)).rstrip() for line in [header, *body]]
    lines.insert(1, "  ".join("-" * w for w in widths).rstrip())
    return "\n".join(lines)


__all__ = [
    "CLIP_KINDS",
    "Clip",
    "TestSet",
    "TestSetError",
    "compare",
    "format_table",
    "load_summary",
    "new_run_dir",
    "rescore_run",
    "run_bench",
    "summarize_run",
]
