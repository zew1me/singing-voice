import numpy as np
import pytest


@pytest.fixture
def tone():
    def make(seconds: float = 1.0, sample_rate: int = 48_000, channels: int = 1, freq: float = 220.0):
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        mono = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return (np.stack([mono] * channels, axis=1) if channels > 1 else mono), sample_rate

    return make


def sung(notes, sample_rate, harmonics=(1.0, 0.5, 0.3, 0.2), cents=0.0, delay_s=0.0, seed=0, noise=1e-3):
    """A deterministic sung line: harmonic notes with vibrato and soft edges; ``None`` is a rest.

    ``notes`` is a list of (MIDI note or None, seconds).
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for midi, seconds in notes:
        n = int(seconds * sample_rate)
        t = np.arange(n) / sample_rate
        if midi is None:
            pieces.append(np.zeros(n))
            continue
        freq = 440 * 2 ** ((midi - 69) / 12) * 2 ** (cents / 1200) * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * t))
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        wave = sum(a * np.sin(k * phase) for k, a in enumerate(harmonics, start=1))
        edge = np.clip(np.minimum(t / 0.03, (seconds - t) / 0.05), 0, 1)
        pieces.append(0.2 * wave * edge)
    audio = np.concatenate(pieces)
    if delay_s:
        audio = np.concatenate([np.zeros(int(delay_s * sample_rate)), audio])[: len(audio)]
    return (audio + noise * rng.standard_normal(len(audio))).astype(np.float32)


PHRASE = [(60, 0.5), (64, 0.5), (None, 0.25), (67, 0.6), (65, 0.4), (None, 0.25)]


@pytest.fixture
def phrase():
    return PHRASE
