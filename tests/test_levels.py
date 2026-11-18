import numpy as np
import pytest

from runpod_worker import engine
from singing_voice.levels import active_rms_db, band_profile, level_summary, profile_distance_db

from .conftest import PHRASE, sung


def test_active_rms_matches_the_worker(tone):
    # The worker gain-matches with its own copy; scoring must measure the same thing.
    audio, sr = tone(seconds=1.0, sample_rate=44_100)
    padded = np.concatenate([np.zeros(sr, dtype=np.float32), audio, 0.01 * audio])
    assert active_rms_db(padded, sr) == pytest.approx(engine.active_rms_db(padded, sr))


def test_level_summary_counts_clipping(tone):
    audio, sr = tone(seconds=0.5, sample_rate=44_100)
    loud = np.clip(audio * 10, -1, 1)
    summary = level_summary(loud, sr)
    assert summary["clipped_samples"] > 0 and summary["peak_dbfs"] == 0.0
    assert level_summary(audio, sr)["clipped_samples"] == 0
    assert level_summary(audio, sr)["active_rms_db"] == pytest.approx(-13.5, abs=0.2)


def test_band_profile_ignores_level_and_tracks_brightness():
    sr = 44_100
    dark = sung(PHRASE, sr, harmonics=(1.0, 0.2, 0.05))
    bright = sung(PHRASE, sr, harmonics=(1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3))
    dark_profile, quiet_profile = band_profile(dark, sr), band_profile(0.1 * dark, sr)
    assert profile_distance_db(dark_profile, quiet_profile) == pytest.approx(0.0, abs=0.05)
    assert band_profile(bright, sr)["centroid_hz"] > dark_profile["centroid_hz"]
    assert profile_distance_db(dark_profile, band_profile(bright, sr)) > 1.0


def test_band_profile_skips_bands_above_nyquist():
    profile = band_profile(sung(PHRASE, 16_000), 16_000)
    assert "air_db" not in profile and "sibilance_db" not in profile and "presence_db" in profile


def test_band_profile_of_silence_or_too_short_is_none():
    assert band_profile(np.zeros(44_100, dtype=np.float32), 44_100) is None
    assert band_profile(np.ones(100, dtype=np.float32), 44_100) is None
    assert profile_distance_db(None, {"low_db": 0.0}) is None
