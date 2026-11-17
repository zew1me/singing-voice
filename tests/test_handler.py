import numpy as np

from runpod_worker.engine import OUTPUT_SAMPLE_RATE, output_length, to_mono
from runpod_worker.handler import handler
from singing_voice.transport import decode_audio_b64, encode_flac_b64


class FakeEngine:
    def __init__(self):
        self.calls = []

    def convert(self, source, source_sr, reference, reference_sr, options):
        self.calls.append((source_sr, reference_sr, options))
        n = output_length(len(to_mono(source)), source_sr)
        return np.full(n, 0.1, dtype=np.float32)

    def describe(self):
        return {"model": "fake", "device": "cpu", "fp16": False, "load_seconds": 0.0}


def job(source, sr, reference, rsr, **extra):
    return {"input": {"source": {"b64": encode_flac_b64(source, sr)}, "reference": {"b64": encode_flac_b64(reference, rsr)}, **extra}}


def test_success_returns_flac_matching_the_source_duration(tone):
    source, sr = tone(seconds=2.0, sample_rate=48_000, channels=2)
    reference, rsr = tone(seconds=3.0, sample_rate=44_100)
    engine = FakeEngine()
    result = handler(job(source, sr, reference, rsr, options={"diffusion_steps": 30}), engine_factory=lambda: engine)
    assert result["status"] == "success"
    assert result["sample_rate"] == OUTPUT_SAMPLE_RATE
    assert result["num_samples"] == 2 * OUTPUT_SAMPLE_RATE
    audio, out_sr = decode_audio_b64(result["audio_b64"])
    assert out_sr == OUTPUT_SAMPLE_RATE and len(audio) == 2 * OUTPUT_SAMPLE_RATE
    assert result["options"]["diffusion_steps"] == 30
    assert engine.calls[0][:2] == (48_000, 44_100)


def test_errors_come_back_as_runpod_failures(tone):
    source, sr = tone()
    engine = FakeEngine()
    assert handler({}, engine_factory=lambda: engine)["error"].startswith("MISSING_INPUT")
    assert handler({"input": {"source": {"b64": encode_flac_b64(source, sr)}}}, engine_factory=lambda: engine)["error"].startswith("MISSING_INPUT")
    bad_shift = handler(job(source, sr, source, sr, options={"semitone_shift": 5}), engine_factory=lambda: engine)
    assert bad_shift["error"].startswith("BAD_OPTION")
    assert engine.calls == []
