"""The fixed-window splitter (Step A) and stitcher (Step C), until phrase splitting replaces them."""
import base64
import io

import numpy as np
import soundfile as sf

from singing_voice.manifests import ChunkPayload, ConvertedManifest, load_manifest, save_converted_manifest
from singing_voice.preprocess import PreprocessorConfig, preprocess_audio_file
from singing_voice.stitch import StitchConfig, stitch_chunks, stitch_manifest_to_file

SR = 16_000


def chunk_audio(chunk):
    audio, sr = sf.read(io.BytesIO(base64.b64decode(chunk.audio_b64)), dtype="float32")
    assert sr == SR
    return audio


def write_vocal(tmp_path, seconds=10.0, silent=((3.0, 5.0),)):
    rng = np.random.default_rng(0)
    audio = (0.2 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
    for start, end in silent:
        audio[int(start * SR) : int(end * SR)] = 0
    path = tmp_path / "vocal.wav"
    sf.write(path, audio, SR, subtype="FLOAT")
    return path, audio


def test_split_keeps_exact_offsets_and_skips_silent_windows(tmp_path):
    path, audio = write_vocal(tmp_path, seconds=10.0, silent=((3.0, 8.5),))
    cfg = PreprocessorConfig(sample_rate=SR, chunk_seconds=4.0, overlap_seconds=0.5)
    manifest = preprocess_audio_file(path, tmp_path / "manifest.json", cfg)

    assert manifest.sample_rate == SR and len(manifest.chunks) == 2  # the 3.5-8 s window is silent
    for chunk in manifest.chunks:
        np.testing.assert_allclose(chunk_audio(chunk), audio[chunk.start : chunk.end], atol=1e-4)
        assert chunk.duration == (chunk.end - chunk.start) / SR
    first, last = manifest.chunks
    assert first.start == 0 and 0 <= first.end - 3 * SR <= 2 * 1024  # trimmed at the silence, within a frame or two
    assert abs(last.start - int(8.5 * SR)) <= 1024 and last.end == len(audio)
    assert load_manifest(tmp_path / "manifest.json").chunks[0].start == first.start


def test_split_trims_silence_inside_a_window(tmp_path):
    path, audio = write_vocal(tmp_path, seconds=4.0, silent=((0.0, 1.0),))
    manifest = preprocess_audio_file(path, tmp_path / "m.json", PreprocessorConfig(sample_rate=SR, chunk_seconds=4.0))
    (chunk,) = manifest.chunks
    assert abs(chunk.start - SR) <= 256 + 1024  # within a frame of the onset


def encode(audio, sr=SR):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="FLOAT")
    return base64.b64encode(buffer.getvalue()).decode()


def test_reassembly_places_chunks_at_their_offsets(tmp_path):
    a, b = np.full(SR, 0.5, dtype=np.float32), np.full(SR, -0.25, dtype=np.float32)
    chunks = [ChunkPayload("b", 3 * SR, 4 * SR, 1.0, encode(b)), ChunkPayload("a", 0, SR, 1.0, encode(a))]
    out = stitch_chunks(chunks, StitchConfig(sample_rate=SR, crossfade_seconds=0.0))
    assert len(out) == 4 * SR
    np.testing.assert_allclose(out[:SR], 0.5)
    np.testing.assert_allclose(out[SR : 3 * SR], 0.0)
    np.testing.assert_allclose(out[3 * SR :], -0.25)


def test_reassembly_crossfades_only_where_chunks_overlap(tmp_path):
    overlap = SR // 4
    a, b = np.ones(SR, dtype=np.float32), np.zeros(SR, dtype=np.float32)
    chunks = [ChunkPayload("a", 0, SR, 1.0, encode(a)), ChunkPayload("b", SR - overlap, 2 * SR - overlap, 1.0, encode(b))]
    out = stitch_chunks(chunks, StitchConfig(sample_rate=SR, crossfade_seconds=0.25))
    assert len(out) == 2 * SR - overlap
    fade = out[SR - overlap : SR]
    assert fade[0] == 1.0 and fade[-1] == 0.0 and np.all(np.diff(fade) <= 0)
    np.testing.assert_allclose(out[: SR - overlap], 1.0)
    np.testing.assert_allclose(out[SR:], 0.0)


def test_first_chunk_and_chunks_after_gaps_start_at_full_level():
    a = np.ones(SR, dtype=np.float32)
    chunks = [ChunkPayload("a", SR // 2, SR + SR // 2, 1.0, encode(a)), ChunkPayload("b", 3 * SR, 4 * SR, 1.0, encode(a))]
    out = stitch_chunks(chunks, StitchConfig(sample_rate=SR, crossfade_seconds=1.5))
    np.testing.assert_allclose(out[: SR // 2], 0.0)
    np.testing.assert_allclose(out[SR // 2 : SR + SR // 2], 1.0)
    np.testing.assert_allclose(out[3 * SR :], 1.0)


def test_reassembly_resamples_and_writes_a_file(tmp_path):
    tone = np.sin(2 * np.pi * 220 * np.arange(32_000) / 32_000).astype(np.float32) * 0.3
    manifest = ConvertedManifest(sample_rate=SR, converted_chunks=[ChunkPayload("a", 0, SR, 1.0, encode(tone, 32_000))])
    save_converted_manifest(manifest, tmp_path / "converted.json")
    out = stitch_manifest_to_file(tmp_path / "converted.json", tmp_path / "out.wav", StitchConfig(sample_rate=SR))
    audio, sr = sf.read(out)
    assert sr == SR and abs(len(audio) - SR) <= 1
