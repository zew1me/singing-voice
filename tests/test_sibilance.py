import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from singing_voice import sibilance
from singing_voice.cli import app

from .conftest import PHRASE, sung

runner = CliRunner()


def _write_pair(tmp_path, sample_rate=44_100):
    src, tgt = tmp_path / "src.wav", tmp_path / "tgt.wav"
    sf.write(src, sung(PHRASE, sample_rate), sample_rate)
    sf.write(tgt, sung(PHRASE, sample_rate, harmonics=(1.0, 0.2), seed=1), sample_rate)
    return src, tgt


def test_rejects_sample_rate_below_band_before_loading(tmp_path):
    result = runner.invoke(
        app,
        ["analyze-sibilance", str(tmp_path / "missing.wav"), str(tmp_path / "missing2.wav"), "--sample-rate", "6000"],
    )
    assert result.exit_code == 2
    assert "--sample-rate" in result.output
    with pytest.raises(ValueError, match="at least 20000"):
        sibilance.analyze_sibilance(tmp_path / "missing.wav", sample_rate=16_000)


@pytest.mark.parametrize("alias", ["same", "hardlink"])
def test_refuses_to_overwrite_an_input_recording(tmp_path, alias):
    src, tgt = _write_pair(tmp_path)
    before = tgt.read_bytes()
    out = tgt
    if alias == "hardlink":
        out = tmp_path / "link.wav"
        out.hardlink_to(tgt)
    result = runner.invoke(app, ["analyze-sibilance", str(src), str(tgt), "--output", str(out)])
    assert result.exit_code == 2
    assert "--output" in result.output
    assert tgt.read_bytes() == before


def test_silent_tail_does_not_change_the_estimate(tmp_path):
    line = sung(PHRASE, 44_100)
    sf.write(tmp_path / "dry.wav", line, 44_100)
    sf.write(tmp_path / "tail.wav", np.concatenate([line, np.zeros(5 * 44_100, dtype=np.float32)]), 44_100)
    dry = sibilance.analyze_sibilance(tmp_path / "dry.wav")
    tail = sibilance.analyze_sibilance(tmp_path / "tail.wav")
    assert abs(tail.sibilance_ratio_db - dry.sibilance_ratio_db) < 0.5
    assert tail.sibilance_peak_hz == dry.sibilance_peak_hz


def test_silent_input_has_no_estimate(tmp_path):
    src, _ = _write_pair(tmp_path)
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(44_100, dtype=np.float32), 44_100)
    with pytest.raises(ValueError, match="silent"):
        sibilance.analyze_sibilance(silent)
    result = runner.invoke(app, ["analyze-sibilance", str(src), str(silent), "--output", str(tmp_path / "r.json")])
    assert result.exit_code == 1
    assert not (tmp_path / "r.json").exists()


def _message(*parts):
    return {"type": "message", "content": list(parts)}


def test_extracts_message_text_and_skips_non_message_items():
    payload = {
        "output": [
            {"type": "reasoning", "summary": []},
            _message({"type": "output_text", "text": "Cut 7 kHz by 3 dB."}),
        ]
    }
    assert sibilance._extract_output_text(payload) == "Cut 7 kHz by 3 dB."


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"output": None},
        {"output": ["nope"]},
        {"output": [{"type": "message", "content": None}]},
        {"output": [_message({"type": "output_text", "text": 42})]},
    ],
)
def test_malformed_payloads_raise_response_error(payload):
    with pytest.raises(RuntimeError, match="OpenAI response"):
        sibilance._extract_output_text(payload)


def test_completed_status_is_accepted():
    payload = {"status": "completed", "output": [_message({"type": "output_text", "text": "ok"})]}
    assert sibilance._extract_output_text(payload) == "ok"


def test_incomplete_generation_is_not_presented_as_complete():
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [_message({"type": "output_text", "text": "Cut 7 kHz by"})],
    }
    with pytest.raises(RuntimeError, match="'incomplete' \\(max_output_tokens\\)"):
        sibilance._extract_output_text(payload)


def test_whitespace_only_text_counts_as_no_text():
    payload = {"output": [_message({"type": "output_text", "text": " \n "})]}
    with pytest.raises(RuntimeError, match="did not include text"):
        sibilance._extract_output_text(payload)


def test_openai_failure_still_writes_the_local_report(tmp_path, monkeypatch):
    import requests

    class TooManyRequests:
        def raise_for_status(self):
            raise requests.HTTPError("429 Client Error: Too Many Requests")

    monkeypatch.setattr(requests, "post", lambda *a, **k: TooManyRequests())
    src, tgt = _write_pair(tmp_path)
    out = tmp_path / "report.json"
    result = runner.invoke(
        app, ["analyze-sibilance", str(src), str(tgt), "--output", str(out), "--openai", "--api-key", "sk-test"]
    )
    assert result.exit_code == 0, result.output
    assert "429" in result.stderr
    report = json.loads(out.read_text())
    assert "local_recommendations" in report and "openai_recommendations" not in report
    assert "429" in report["openai_error"]
