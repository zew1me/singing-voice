import base64

import numpy as np
import pytest

from singing_voice import transport
from singing_voice.transport import (
    TransportError,
    decode_audio_b64,
    deliver_audio,
    encode_flac_b64,
    read_audio_spec,
)


def test_flac_round_trip_keeps_rate_channels_and_samples(tone):
    audio, sr = tone(seconds=0.5, sample_rate=48_000, channels=2)
    decoded, decoded_sr = decode_audio_b64(encode_flac_b64(audio, sr))
    assert decoded_sr == sr
    assert decoded.shape == audio.shape
    assert np.max(np.abs(decoded - audio)) < 1e-6  # 24-bit


def test_encode_clips_instead_of_wrapping():
    decoded, _ = decode_audio_b64(encode_flac_b64(np.array([1.5, -1.5, 0.0], dtype=np.float32), 44_100))
    assert decoded.max() <= 1.0 and decoded.min() >= -1.0
    assert decoded[0] > 0.99 and decoded[1] < -0.99


@pytest.mark.parametrize(
    "spec, code",
    [
        (None, "MISSING_INPUT"),
        ({}, "MISSING_INPUT"),
        ({"url": "file:///etc/passwd"}, "BAD_URL"),
        ({"b64": "not base64!!"}, "BAD_BASE64"),
        ({"b64": base64.b64encode(b"not audio").decode()}, "BAD_AUDIO"),
    ],
)
def test_bad_audio_specs_name_the_problem(spec, code):
    with pytest.raises(TransportError) as err:
        read_audio_spec(spec, "source")
    assert err.value.code == code


def test_deliver_inline_by_default(tone):
    audio, _ = tone(seconds=0.2, sample_rate=44_100)
    result = deliver_audio(audio, 44_100, None)
    assert result["format"] == "flac"
    decoded, sr = decode_audio_b64(result["audio_b64"])
    assert sr == 44_100 and len(decoded) == len(audio)


def test_deliver_refuses_results_over_the_inline_limit(tone, monkeypatch):
    monkeypatch.setattr(transport, "INLINE_AUDIO_LIMIT_BYTES", 1_000)
    audio, _ = tone(seconds=1.0, sample_rate=44_100)
    with pytest.raises(TransportError) as err:
        deliver_audio(audio, 44_100, {})
    assert err.value.code == "OUTPUT_TOO_LARGE"
