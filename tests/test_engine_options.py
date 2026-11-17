import numpy as np
import pytest

from runpod_worker.engine import ConvertOptions, EngineError, fit_length, output_length, to_mono


def test_defaults_suit_singing():
    options = ConvertOptions.from_dict(None)
    assert options.diffusion_steps == 40 and options.semitone_shift == 0


@pytest.mark.parametrize("shift", [-12, 0, 12, 24, -24])
def test_octave_shifts_allowed(shift):
    assert ConvertOptions.from_dict({"semitone_shift": shift}).semitone_shift == shift


@pytest.mark.parametrize(
    "options",
    [{"semitone_shift": 3}, {"semitone_shift": -7}, {"diffusion_steps": 0}, {"inference_cfg_rate": 1.5}],
)
def test_out_of_key_or_out_of_range_options_rejected(options):
    with pytest.raises(EngineError) as err:
        ConvertOptions.from_dict(options)
    assert err.value.code == "BAD_OPTION"


def test_unknown_option_rejected():
    with pytest.raises(EngineError) as err:
        ConvertOptions.from_dict({"auto_f0_adjust": True})
    assert err.value.code == "UNKNOWN_OPTION"


def test_fit_length_pads_and_trims():
    audio = np.ones(10, dtype=np.float32)
    assert fit_length(audio, 4).tolist() == [1, 1, 1, 1]
    padded = fit_length(audio, 12)
    assert len(padded) == 12 and padded[-2:].tolist() == [0, 0]


def test_output_length_is_source_duration_at_44k():
    assert output_length(48_000, 48_000) == 44_100
    assert output_length(44_100 * 3, 44_100) == 44_100 * 3


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1)
    assert to_mono(stereo).tolist() == [0.5] * 4


def test_active_rms_ignores_silence(tone):
    audio, sr = tone(seconds=1.0, sample_rate=44_100)  # 0.3 amplitude sine: -13.5 dBFS RMS
    padded = np.concatenate([np.zeros(sr * 3, dtype=np.float32), audio])
    from runpod_worker.engine import active_rms_db

    assert active_rms_db(padded, sr) == pytest.approx(-13.5, abs=0.2)
    assert active_rms_db(np.zeros(sr, dtype=np.float32), sr) is None


def test_gain_to_reaches_target(tone):
    from runpod_worker.engine import active_rms_db, gain_to

    audio, sr = tone(seconds=1.0, sample_rate=44_100)
    assert active_rms_db(audio * gain_to(audio, sr, -32.0), sr) == pytest.approx(-32.0, abs=0.1)
