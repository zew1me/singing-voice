"""Stage 5: score a conversion against its input and the reference voice.

One report per converted file, so a change to the pipeline can be judged by numbers:

- ``melody``: f0 of the output against the input, in cents, on frames both sing; plus
  voiced/unvoiced agreement. The conversion should keep the melody exactly (allowing for a
  requested octave shift).
- ``alignment``: lag of the output's energy envelope against the input's. Should be 0.
- ``levels`` and ``spectrum``: active RMS, peak, clipping, and a level-independent band
  profile for input, output and reference, with band distances. A cheap timbre proxy: the
  output's spectrum should move from the input's toward the reference's.
- ``timbre`` (optional model): speaker-embedding cosine similarity of the output to the
  reference (want high) and to the input (want low), with the input-to-reference similarity as
  the baseline.
- ``words`` (optional model): Whisper transcripts of input and output, and the word error rate
  of the output against the input.

The model-based sections are skipped, with the reason in the report, when their packages
(the ``score`` extra) are missing.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .levels import band_profile, level_summary, profile_distance_db, to_mono

SCHEMA = "svtool.score/1"
ANALYSIS_SAMPLE_RATE = 16_000
HOP = 160  # 10 ms at 16 kHz
FMIN_HZ = 65.0  # C2
FMAX_HZ = 1_100.0  # about C#6
REFERENCE_MAX_SECONDS = 25  # what the worker uses of the reference
MAX_LAG_S = 0.5


class Embedder(Protocol):
    name: str

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: Optional[str] = None) -> Tuple[str, Optional[str]]: ...


@dataclass
class Models:
    """Loaded scoring models, or why each one is missing."""

    embedder: Optional[Embedder] = None
    transcriber: Optional[Transcriber] = None
    skipped: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScoreInputs:
    output: Path
    source: Path
    reference: Optional[Path] = None
    start_s: float = 0.0
    duration_s: Optional[float] = None
    semitone_shift: int = 0

    @classmethod
    def from_sidecar(cls, output: Path) -> "ScoreInputs":
        """Read source excerpt, reference and options from ``svtool convert``'s JSON sidecar."""
        sidecar = output.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"no sidecar {sidecar.name}; pass --source (and --reference)")
        meta = json.loads(sidecar.read_text())
        excerpt = meta.get("source_excerpt") or {}
        if not excerpt.get("path"):
            raise ValueError(f"{sidecar.name} does not name the source; pass --source")
        return cls(
            output=output,
            source=Path(excerpt["path"]),
            reference=Path(meta["reference"]) if meta.get("reference") else None,
            start_s=float(excerpt.get("start_s") or 0.0),
            duration_s=excerpt.get("duration_s"),
            semitone_shift=int((meta.get("options") or {}).get("semitone_shift", 0)),
        )


# --- audio helpers -------------------------------------------------------------------------


def resample(audio: np.ndarray, sample_rate: int, target: int = ANALYSIS_SAMPLE_RATE) -> np.ndarray:
    audio = to_mono(audio)
    if sample_rate == target:
        return audio
    import librosa  # noqa: PLC0415

    return librosa.resample(audio, orig_sr=sample_rate, target_sr=target).astype(np.float32)


def read_audio(path: Path, start_s: float = 0.0, duration_s: Optional[float] = None) -> Tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    info = sf.info(str(path))
    start = int(round(start_s * info.samplerate))
    frames = -1 if duration_s is None else int(round(float(duration_s) * info.samplerate))
    audio, sample_rate = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
    return to_mono(audio), int(sample_rate)


# --- melody --------------------------------------------------------------------------------


@dataclass
class PitchTrack:
    f0_hz: np.ndarray  # NaN where unvoiced
    voiced: np.ndarray  # bool

    @property
    def hop_s(self) -> float:
        return HOP / ANALYSIS_SAMPLE_RATE


def track_pitch(audio_16k: np.ndarray, semitone_shift: int = 0) -> PitchTrack:
    """pYIN f0 at 10 ms hops over a singing range (C2 to C#6), moved by ``semitone_shift``.

    The output of an octave-shifted conversion is tracked over the shifted range, so a note
    sung an octave up near the top of the range is not lost above ``fmax``.
    """
    import librosa  # noqa: PLC0415

    ratio = 2 ** (semitone_shift / 12)
    fmin, fmax = FMIN_HZ * ratio, FMAX_HZ * ratio
    # pYIN needs frame_length - frame_length // 2 > the longest period (sr / fmin).
    frame_length = 1024
    while frame_length // 2 <= ANALYSIS_SAMPLE_RATE / fmin + 1:
        frame_length *= 2
    f0, voiced, _ = librosa.pyin(
        np.asarray(audio_16k, dtype=np.float32),
        fmin=fmin,
        fmax=min(fmax, ANALYSIS_SAMPLE_RATE / 2 - 1),
        sr=ANALYSIS_SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=HOP,
        center=True,
    )
    return PitchTrack(f0_hz=np.asarray(f0, dtype=np.float64), voiced=np.asarray(voiced, dtype=bool))


def _pct(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator == 0 else round(100.0 * numerator / denominator, 1)


def melody_metrics(source: PitchTrack, output: PitchTrack, semitone_shift: int = 0) -> Dict[str, Any]:
    """Output f0 against input f0 in cents (after the requested shift), and voicing agreement."""
    # Voicing covers the longer timeline; frames one track lacks count as unvoiced.
    n = max(len(source.voiced), len(output.voiced))
    sv = np.pad(source.voiced, (0, n - len(source.voiced)))
    ov = np.pad(output.voiced, (0, n - len(output.voiced)))
    f0_s = np.pad(source.f0_hz, (0, n - len(source.f0_hz)), constant_values=np.nan)
    f0_o = np.pad(output.f0_hz, (0, n - len(output.f0_hz)), constant_values=np.nan)
    both = sv & ov & np.isfinite(f0_s) & np.isfinite(f0_o)
    cents = 1200 * np.log2(f0_o[both] / f0_s[both]) - 100 * semitone_shift
    abs_cents = np.abs(cents)
    have = cents.size > 0
    return {
        "frames": int(n),
        "source_voiced_frames": int(sv.sum()),
        "output_voiced_frames": int(ov.sum()),
        "compared_frames": int(cents.size),
        "median_abs_cents": round(float(np.median(abs_cents)), 1) if have else None,
        "mean_cents": round(float(np.mean(cents)), 1) if have else None,
        "within_50_cents_pct": _pct(float(np.sum(abs_cents <= 50)), cents.size),
        "gross_errors_pct": _pct(float(np.sum(abs_cents > 600)), cents.size),
        "voicing_agreement_pct": _pct(float(np.sum(sv == ov)), n),
        "voicing_recall_pct": _pct(float(np.sum(sv & ov)), float(sv.sum())),
        "voicing_false_alarm_pct": _pct(float(np.sum(ov & ~sv)), float((~sv).sum())),
    }


# --- alignment -----------------------------------------------------------------------------


def energy_envelope_db(audio_16k: np.ndarray, floor_db: float = 50.0) -> np.ndarray:
    audio = np.asarray(audio_16k, dtype=np.float64)
    frames = len(audio) // HOP
    if frames == 0:
        return np.zeros(0)
    power = np.mean(audio[: frames * HOP].reshape(frames, HOP) ** 2, axis=1) + 1e-12
    db = 10 * np.log10(power)
    return np.maximum(db, db.max() - floor_db)


def alignment_lag(source_16k: np.ndarray, output_16k: np.ndarray, max_lag_s: float = MAX_LAG_S) -> Dict[str, Any]:
    """Lag (ms) that best lines the output's energy envelope up with the input's; positive = output late."""
    a, b = energy_envelope_db(source_16k), energy_envelope_db(output_16k)
    n = min(len(a), len(b))
    max_lag = int(max_lag_s * ANALYSIS_SAMPLE_RATE / HOP)
    if n < 2 * max_lag + 10:
        max_lag = max(0, n // 4)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    if n == 0 or not np.any(a) or not np.any(b):  # silence or a constant level: nothing to line up
        return {"lag_ms": None, "envelope_correlation": None}
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[: n - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: n + lag])
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
        corr = float(np.sum(x * y) / denom) if denom > 0 else 0.0
        if corr > best:
            best_lag, best = lag, corr
    return {
        "lag_ms": round(best_lag * 1000 * HOP / ANALYSIS_SAMPLE_RATE, 1),
        "envelope_correlation": None if not np.isfinite(best) else round(best, 3),
    }


# --- words ---------------------------------------------------------------------------------


def normalize_words(text: str) -> List[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> Optional[float]:
    """Levenshtein distance over words divided by the reference length."""
    if not reference:
        return None
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
        previous = current
    return round(previous[-1] / len(reference), 3)


def words_section(transcriber: Transcriber, source_16k: np.ndarray, output_16k: np.ndarray) -> Dict[str, Any]:
    source_text, language = transcriber.transcribe(source_16k, ANALYSIS_SAMPLE_RATE)
    output_text, _ = transcriber.transcribe(output_16k, ANALYSIS_SAMPLE_RATE, language=language)
    return {
        "model": transcriber.name,
        "language": language,
        "source_text": source_text,
        "output_text": output_text,
        "wer": word_error_rate(normalize_words(source_text), normalize_words(output_text)),
    }


# --- timbre --------------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)


def timbre_section(embedder: Embedder, source_16k: np.ndarray, output_16k: np.ndarray, reference_16k: Optional[np.ndarray]) -> Dict[str, Any]:
    source = embedder.embed(source_16k, ANALYSIS_SAMPLE_RATE)
    output = embedder.embed(output_16k, ANALYSIS_SAMPLE_RATE)
    section: Dict[str, Any] = {"model": embedder.name, "output_to_source": cosine(output, source)}
    if reference_16k is not None:
        reference = embedder.embed(reference_16k, ANALYSIS_SAMPLE_RATE)
        section["output_to_reference"] = cosine(output, reference)
        section["source_to_reference"] = cosine(source, reference)
        section["moved_toward_reference"] = round(section["output_to_reference"] - section["source_to_reference"], 4)
    return section


# --- report --------------------------------------------------------------------------------

# (summary key, dotted path in the report, which way is better: +1 higher, -1 lower,
# "abs" closer to zero, 0 a baseline that is not judged)
SUMMARY_METRICS: Tuple[Tuple[str, str, Any], ...] = (
    ("timbre_to_reference", "timbre.output_to_reference", +1),
    ("timbre_to_source", "timbre.output_to_source", -1),
    ("timbre_source_to_reference", "timbre.source_to_reference", 0),
    ("spectrum_to_reference_db", "spectrum.output_to_reference_db", -1),
    ("spectrum_source_to_reference_db", "spectrum.source_to_reference_db", 0),
    ("pitch_median_abs_cents", "melody.median_abs_cents", -1),
    ("pitch_within_50c_pct", "melody.within_50_cents_pct", +1),
    ("pitch_gross_errors_pct", "melody.gross_errors_pct", -1),
    ("voicing_agreement_pct", "melody.voicing_agreement_pct", +1),
    ("wer", "words.wer", -1),
    ("alignment_lag_ms", "alignment.lag_ms", "abs"),
    ("output_clipped_samples", "levels.output.clipped_samples", -1),
    ("level_change_db", "levels.output_minus_source_db", "abs"),
)


def _lookup(report: Dict[str, Any], dotted: str) -> Any:
    node: Any = report
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _lookup(report, path) for key, path, _ in SUMMARY_METRICS}


def _guarded(name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Model sections must not sink the rest of the report (e.g. a failed model download)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}


def score(inputs: ScoreInputs, models: Optional[Models] = None, timbre: bool = True, words: bool = True) -> Dict[str, Any]:
    """Build the report for one converted file."""
    models = models or Models()
    source, source_sr = read_audio(inputs.source, inputs.start_s, inputs.duration_s)
    output, output_sr = read_audio(inputs.output)
    reference = reference_sr = None
    if inputs.reference is not None:
        reference, reference_sr = read_audio(inputs.reference, 0.0, REFERENCE_MAX_SECONDS)

    source_16k, output_16k = resample(source, source_sr), resample(output, output_sr)
    reference_16k = None if reference is None else resample(reference, reference_sr)

    levels = {"source": level_summary(source, source_sr), "output": level_summary(output, output_sr)}
    if reference is not None:
        levels["reference"] = level_summary(reference, reference_sr)
    if levels["source"]["active_rms_db"] is not None and levels["output"]["active_rms_db"] is not None:
        levels["output_minus_source_db"] = round(levels["output"]["active_rms_db"] - levels["source"]["active_rms_db"], 1)

    profiles = {"source": band_profile(source, source_sr), "output": band_profile(output, output_sr)}
    spectrum: Dict[str, Any] = {"bands": profiles, "output_to_source_db": profile_distance_db(profiles["output"], profiles["source"])}
    if reference is not None:
        profiles["reference"] = band_profile(reference, reference_sr)
        spectrum["output_to_reference_db"] = profile_distance_db(profiles["output"], profiles["reference"])
        spectrum["source_to_reference_db"] = profile_distance_db(profiles["source"], profiles["reference"])

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(inputs).items()},
        "levels": levels,
        "spectrum": spectrum,
        "alignment": alignment_lag(source_16k, output_16k),
        "melody": melody_metrics(track_pitch(source_16k), track_pitch(output_16k, inputs.semitone_shift), inputs.semitone_shift),
    }
    if not timbre:
        report["timbre"] = {"skipped": "turned off"}
    elif models.embedder is None:
        report["timbre"] = {"skipped": models.skipped.get("timbre", "no speaker-embedding model loaded")}
    else:
        report["timbre"] = _guarded("timbre", lambda: timbre_section(models.embedder, source_16k, output_16k, reference_16k))
    if not words:
        report["words"] = {"skipped": "turned off"}
    elif models.transcriber is None:
        report["words"] = {"skipped": models.skipped.get("words", "no transcription model loaded")}
    else:
        report["words"] = _guarded("words", lambda: words_section(models.transcriber, source_16k, output_16k))
    report["summary"] = summarize(report)
    return report


def report_path(output: Path) -> Path:
    return output.with_suffix(".score.json")


def write_report(report: Dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "Embedder",
    "Models",
    "PitchTrack",
    "SUMMARY_METRICS",
    "ScoreInputs",
    "Transcriber",
    "alignment_lag",
    "melody_metrics",
    "normalize_words",
    "report_path",
    "score",
    "summarize",
    "track_pitch",
    "word_error_rate",
    "write_report",
]
