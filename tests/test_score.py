import json

import numpy as np
import pytest
import soundfile as sf

from singing_voice.score import (
    ANALYSIS_SAMPLE_RATE,
    Models,
    PitchTrack,
    ScoreInputs,
    alignment_lag,
    melody_metrics,
    normalize_words,
    score,
    summarize,
    track_pitch,
    word_error_rate,
)

from .conftest import PHRASE, sung

SR = ANALYSIS_SAMPLE_RATE


def track(f0):
    f0 = np.asarray(f0, dtype=float)
    return PitchTrack(f0_hz=f0, voiced=np.isfinite(f0))


def test_melody_metrics_measure_cents_on_frames_both_sing():
    source = track([220, 220, np.nan, 440, 440])
    output = track([220 * 2 ** (30 / 1200), 220, 300, np.nan, 440 * 2])
    m = melody_metrics(source, output)
    assert m["compared_frames"] == 3
    assert m["median_abs_cents"] == 30.0
    assert m["gross_errors_pct"] == pytest.approx(33.3, abs=0.1)  # the octave jump
    assert m["voicing_agreement_pct"] == 60.0
    assert m["voicing_recall_pct"] == 75.0
    assert m["voicing_false_alarm_pct"] == 100.0


def test_melody_metrics_allow_for_the_requested_octave_shift():
    source = track([220, 330, 440])
    output = track([440, 660, 880])
    assert melody_metrics(source, output, semitone_shift=12)["median_abs_cents"] == 0.0
    assert melody_metrics(source, output)["median_abs_cents"] == 1200.0


def test_voicing_counts_frames_missing_from_the_shorter_track_as_unvoiced():
    m = melody_metrics(track([220, 220, 440, 440]), track([220, 220]))
    assert m["frames"] == 4 and m["compared_frames"] == 2
    assert m["voicing_recall_pct"] == 50.0 and m["voicing_agreement_pct"] == 50.0
    m = melody_metrics(track([220, 220]), track([220, 220, 440, 440]))
    assert m["voicing_false_alarm_pct"] == 100.0 and m["voicing_agreement_pct"] == 50.0


def test_melody_metrics_with_nothing_sung():
    m = melody_metrics(track([np.nan] * 4), track([np.nan] * 4))
    assert m["median_abs_cents"] is None and m["within_50_cents_pct"] is None
    assert m["voicing_agreement_pct"] == 100.0


def test_pitch_tracking_follows_a_sung_line():
    clean = track_pitch(sung(PHRASE, SR))
    sharp = track_pitch(sung(PHRASE, SR, harmonics=(1.0, 0.2), cents=40, seed=1))
    m = melody_metrics(clean, sharp)
    assert m["compared_frames"] > 150
    assert m["median_abs_cents"] == pytest.approx(40, abs=8)
    assert m["voicing_agreement_pct"] > 90


def test_pitch_tracking_follows_an_octave_shift_above_the_unshifted_range():
    high = [(76, 0.6), (None, 0.2), (79, 0.6)]  # E5, G5: an octave up is 1319 and 1568 Hz
    m = melody_metrics(track_pitch(sung(high, SR)), track_pitch(sung(high, SR, cents=1200, seed=1), semitone_shift=12), semitone_shift=12)
    assert m["median_abs_cents"] == pytest.approx(0, abs=10)
    assert m["voicing_recall_pct"] > 85


def test_pitch_tracking_two_octaves_down_still_runs():
    low = [(45, 0.8)]  # A2, two octaves down is 27.5 Hz
    m = melody_metrics(track_pitch(sung(low, SR)), track_pitch(sung(low, SR, cents=-2400, seed=1), semitone_shift=-24), semitone_shift=-24)
    assert m["compared_frames"] > 0


def test_alignment_of_silence_is_unavailable():
    silence = np.zeros(2 * SR, dtype=np.float32)
    assert alignment_lag(silence, silence) == {"lag_ms": None, "envelope_correlation": None}
    assert alignment_lag(sung(PHRASE, SR), silence)["lag_ms"] is None


@pytest.mark.parametrize("delay_s", [0.0, 0.05, -0.12])
def test_alignment_finds_the_lag(delay_s):
    source = sung(PHRASE, SR)
    if delay_s >= 0:
        output = sung(PHRASE, SR, delay_s=delay_s, seed=1)
    else:
        output = np.concatenate([source[int(-delay_s * SR):], np.zeros(int(-delay_s * SR), dtype=np.float32)])
    result = alignment_lag(source, output)
    assert result["lag_ms"] == pytest.approx(delay_s * 1000, abs=10)
    assert result["envelope_correlation"] > 0.8


def test_word_error_rate():
    ref = normalize_words("Hold me close, don't let go!")
    assert ref == ["hold", "me", "close", "don't", "let", "go"]
    assert word_error_rate(ref, ref) == 0.0
    assert word_error_rate(ref, normalize_words("hold me close let it go")) == pytest.approx(2 / 6, abs=1e-3)
    assert word_error_rate(ref, []) == 1.0
    assert word_error_rate([], ["anything"]) is None


class FakeEmbedder:
    """Embeds by spectral brightness, enough to tell the fake voices apart."""

    name = "fake-embedder"

    def embed(self, audio, sample_rate):
        assert sample_rate == SR
        spectrum = np.abs(np.fft.rfft(audio))
        bands = np.array_split(spectrum, 8)
        return np.log10(np.array([b.sum() for b in bands]) + 1e-9)


class FakeTranscriber:
    name = "fake-whisper"

    def __init__(self, texts):
        self.texts = list(texts)
        self.languages = []

    def transcribe(self, audio, sample_rate, language=None):
        self.languages.append(language)
        return self.texts.pop(0), language or "en"


class BrokenEmbedder:
    name = "broken"

    def embed(self, audio, sample_rate):
        raise RuntimeError("weights missing")


def write_clips(tmp_path):
    long_line = PHRASE * 2
    source = sung(long_line, 48_000, harmonics=(1.0, 0.5, 0.3, 0.2))
    # The "source" file has a second of something else before the excerpt.
    sf.write(tmp_path / "src.wav", np.concatenate([np.full(48_000, 0.5, dtype=np.float32), source]), 48_000)
    sf.write(tmp_path / "ref.wav", sung(long_line[::-1], 44_100, harmonics=(1.0, 0.1), seed=2), 44_100)
    sf.write(tmp_path / "out.wav", sung(long_line, 44_100, harmonics=(1.0, 0.15), seed=3), 44_100)
    excerpt_s = len(source) / 48_000
    (tmp_path / "out.json").write_text(json.dumps({
        "options": {"semitone_shift": 0},
        "source_excerpt": {"path": str(tmp_path / "src.wav"), "start_s": 1.0, "duration_s": excerpt_s},
        "reference": str(tmp_path / "ref.wav"),
    }))
    return tmp_path / "out.wav"


def test_inputs_come_from_the_convert_sidecar(tmp_path):
    out = write_clips(tmp_path)
    inputs = ScoreInputs.from_sidecar(out)
    assert inputs.source == tmp_path / "src.wav" and inputs.reference == tmp_path / "ref.wav"
    assert inputs.start_s == 1.0 and inputs.semitone_shift == 0
    with pytest.raises(FileNotFoundError):
        ScoreInputs.from_sidecar(tmp_path / "ref.wav")


def test_full_report(tmp_path):
    out = write_clips(tmp_path)
    transcriber = FakeTranscriber(["hold me close", "hold me clothes"])
    models = Models(embedder=FakeEmbedder(), transcriber=transcriber)
    report = score(ScoreInputs.from_sidecar(out), models)

    assert report["schema"] == "svtool.score/1"
    # The excerpt skipped the constant first second, so melody and timing line up.
    assert report["melody"]["median_abs_cents"] < 10
    assert report["alignment"]["lag_ms"] == pytest.approx(0, abs=10)
    assert abs(report["levels"]["output_minus_source_db"]) < 3
    # The output's tone is the reference's (both dark), not the source's.
    spectrum = report["spectrum"]
    assert spectrum["output_to_reference_db"] < spectrum["source_to_reference_db"]
    timbre = report["timbre"]
    assert timbre["output_to_reference"] > timbre["output_to_source"]
    assert timbre["moved_toward_reference"] > 0
    words = report["words"]
    assert words["wer"] == pytest.approx(1 / 3, abs=1e-3)
    assert transcriber.languages == [None, "en"]  # output transcribed in the input's language
    assert report["summary"] == summarize(report)
    assert report["summary"]["wer"] == words["wer"]
    json.dumps(report)  # serialisable


def test_missing_or_failing_models_are_recorded_not_fatal(tmp_path):
    out = write_clips(tmp_path)
    report = score(
        ScoreInputs.from_sidecar(out),
        Models(embedder=BrokenEmbedder(), skipped={"words": "faster_whisper is not installed"}),
    )
    assert "weights missing" in report["timbre"]["error"]
    assert report["words"] == {"skipped": "faster_whisper is not installed"}
    assert report["summary"]["timbre_to_reference"] is None
    assert report["summary"]["pitch_median_abs_cents"] is not None

    off = score(ScoreInputs.from_sidecar(out), Models(embedder=FakeEmbedder()), timbre=False, words=False)
    assert off["timbre"] == {"skipped": "turned off"} and off["words"] == {"skipped": "turned off"}


def test_load_models_records_why_a_model_is_missing(monkeypatch):
    from singing_voice import score_models

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    def offline(*args, **kwargs):
        raise OSError("couldn't connect to huggingface.co")

    monkeypatch.setattr(score_models, "EcapaEmbedder", missing)
    monkeypatch.setattr(score_models, "WhisperTranscriber", offline)
    models = score_models.load_models()
    assert models.embedder is None and models.transcriber is None
    assert models.skipped["timbre"].startswith("torch is not installed")
    assert "huggingface.co" in models.skipped["words"]
    assert score_models.load_models(timbre=False, words=False).skipped == {}
