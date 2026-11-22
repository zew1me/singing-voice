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
