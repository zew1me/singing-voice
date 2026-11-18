import json

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from singing_voice import score_cli
from singing_voice.cli import app
from singing_voice.score import Models

from .conftest import PHRASE, sung

runner = CliRunner()


def test_score_command_writes_report_next_to_output(tmp_path, monkeypatch):
    monkeypatch.setattr(score_cli, "_load_models", lambda timbre, words, whisper_model: Models(skipped={"timbre": "not installed"}))
    sf.write(tmp_path / "src.wav", sung(PHRASE, 44_100), 44_100)
    sf.write(tmp_path / "out.wav", sung(PHRASE, 44_100, harmonics=(1.0, 0.2), seed=1), 44_100)
    result = runner.invoke(app, ["score", str(tmp_path / "out.wav"), "--source", str(tmp_path / "src.wav"), "--no-words"])
    assert result.exit_code == 0, result.output
    line = json.loads(result.stdout.strip().splitlines()[-1])
    report = json.loads((tmp_path / "out.score.json").read_text())
    assert line["report"] == str(tmp_path / "out.score.json") and line["summary"] == report["summary"]
    assert report["timbre"] == {"skipped": "not installed"} and report["words"] == {"skipped": "turned off"}
    assert report["melody"]["median_abs_cents"] < 10


def test_source_override_keeps_the_sidecar_reference_and_shift(tmp_path, monkeypatch):
    monkeypatch.setattr(score_cli, "_load_models", lambda timbre, words, whisper_model: Models())
    seen = {}

    def fake_score(inputs, models, timbre, words):
        seen["inputs"] = inputs
        return {"summary": {}}

    monkeypatch.setattr(score_cli, "score", fake_score)
    sf.write(tmp_path / "src.wav", np.zeros(44_100, dtype=np.float32), 44_100)
    sf.write(tmp_path / "ref.wav", np.zeros(44_100, dtype=np.float32), 44_100)
    sf.write(tmp_path / "out.wav", np.zeros(44_100, dtype=np.float32), 44_100)
    (tmp_path / "out.json").write_text(json.dumps({
        "options": {"semitone_shift": 12}, "reference": str(tmp_path / "ref.wav"),
        "source_excerpt": {"path": "/elsewhere/original.wav", "start_s": 30, "duration_s": 1},
    }))
    result = runner.invoke(app, ["score", str(tmp_path / "out.wav"), "--source", str(tmp_path / "src.wav"), "--no-timbre", "--no-words"])
    assert result.exit_code == 0, result.output
    inputs = seen["inputs"]
    assert inputs.source == tmp_path / "src.wav" and inputs.start_s == 0.0 and inputs.duration_s == 1.0
    assert inputs.reference == tmp_path / "ref.wav" and inputs.semitone_shift == 12


def test_score_without_sidecar_or_source_is_a_usage_error(tmp_path):
    sf.write(tmp_path / "out.wav", np.zeros(4410, dtype=np.float32), 44_100)
    result = runner.invoke(app, ["score", str(tmp_path / "out.wav")])
    assert result.exit_code == 2 and "sidecar" in result.output


def test_bench_dry_run_reports_missing_audio(tmp_path):
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"reference": "ref.wav", "clips": [{"id": "clean", "source": "clean.wav", "duration_s": 5}]}))
    result = runner.invoke(app, ["bench", "run", str(testset), "--label", "x", "--dry-run"])
    assert result.exit_code == 1 and "does not exist" in result.output
    sf.write(tmp_path / "clean.wav", np.zeros(44_100 * 6, dtype=np.float32), 44_100)
    sf.write(tmp_path / "ref.wav", np.zeros(44_100, dtype=np.float32), 44_100)
    result = runner.invoke(app, ["bench", "run", str(testset), "--label", "x", "--dry-run"])
    assert result.exit_code == 0, result.output


def test_committed_testset_is_valid():
    from pathlib import Path

    from singing_voice.bench import TestSet

    testset = TestSet.load(Path(__file__).parent.parent / "testsets" / "m1.json")
    assert [(c.id, c.kind) for c in testset.clips] == [("clean", "clean"), ("bleed", "bleed"), ("harmonies", "harmonies")]


def test_bench_compare_prints_a_table(tmp_path):
    for name, cents in (("a", 20.0), ("b", 9.0)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps({"label": name, "clips": {"clean": {"pitch_median_abs_cents": cents}}}))
    result = runner.invoke(app, ["bench", "compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert "pitch_median_abs_cents" in result.output and "-11" in result.output and "better" in result.output
