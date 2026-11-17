import numpy as np
import pytest


@pytest.fixture
def tone():
    def make(seconds: float = 1.0, sample_rate: int = 48_000, channels: int = 1, freq: float = 220.0):
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        mono = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return (np.stack([mono] * channels, axis=1) if channels > 1 else mono), sample_rate

    return make
