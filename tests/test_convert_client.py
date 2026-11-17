import json

import pytest
import soundfile as sf

from singing_voice import convert_client
from singing_voice.convert_client import ConvertRequest, build_payload, next_revision_path, save_result
from singing_voice.transport import TransportError, decode_audio_b64, encode_flac_b64


def write(path, audio, sr):
    sf.write(str(path), audio, sr)
    return path


def test_next_revision_skips_taken_numbers(tmp_path):
    assert next_revision_path(tmp_path, "vox_seedvc").name == "vox_seedvc_r01.wav"
    (tmp_path / "vox_seedvc_r01.wav").touch()
    (tmp_path / "vox_seedvc_r03.wav").touch()
    (tmp_path / "other_r09.wav").touch()
    assert next_revision_path(tmp_path, "vox_seedvc").name == "vox_seedvc_r04.wav"


def test_payload_carries_only_the_excerpt(tmp_path, tone):
    audio, sr = tone(seconds=3.0, sample_rate=48_000)
    request = ConvertRequest(write(tmp_path / "src.wav", audio, sr), write(tmp_path / "ref.wav", audio, sr), {"diffusion_steps": 30}, start_s=1.0, duration_s=0.5)
    payload = build_payload(request)
    excerpt, excerpt_sr = decode_audio_b64(payload["source"]["b64"])
    assert excerpt_sr == sr and len(excerpt) == sr // 2
    assert payload["options"] == {"diffusion_steps": 30}


def test_payload_over_runpod_limit_is_refused(tmp_path, tone, monkeypatch):
    monkeypatch.setattr(convert_client, "RUN_PAYLOAD_LIMIT_BYTES", 1_000)
    audio, sr = tone(seconds=1.0)
    request = ConvertRequest(write(tmp_path / "src.wav", audio, sr), write(tmp_path / "ref.wav", audio, sr))
    with pytest.raises(TransportError) as err:
        build_payload(request)
    assert err.value.code == "INPUT_TOO_LARGE"


def test_excerpt_past_the_end_is_refused(tmp_path, tone):
    audio, sr = tone(seconds=1.0)
    request = ConvertRequest(write(tmp_path / "src.wav", audio, sr), write(tmp_path / "ref.wav", audio, sr), start_s=5.0)
    with pytest.raises(TransportError):
        build_payload(request)


def test_save_result_writes_wav_and_sidecar_and_never_overwrites(tmp_path, tone):
    audio, _ = tone(seconds=0.5, sample_rate=44_100)
    output = {"audio_b64": encode_flac_b64(audio, 44_100), "sample_rate": 44_100, "options": {"diffusion_steps": 40}}
    out = tmp_path / "vox_seedvc_r01.wav"
    meta = save_result(output, out)
    assert sf.info(str(out)).samplerate == 44_100 and sf.info(str(out)).subtype == "PCM_24"
    assert json.loads(out.with_suffix(".json").read_text()) == meta and "audio_b64" not in meta
    with pytest.raises(FileExistsError):
        save_result(output, out)
