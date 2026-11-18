"""Optional models for scoring: a speaker-embedding model (timbre) and Whisper (words).

Both come from the ``score`` extra (``uv pip install -e .[score]``) and download their weights
from Hugging Face on first use. Neither is needed for the rest of the report.

- Timbre: SpeechBrain's ECAPA-TDNN trained on VoxCeleb (speech). It is not trained on singing,
  so absolute similarities read lower than for speech; compare runs against each other and
  against the input-to-reference baseline in the same report. It is deliberately not the
  CAM++ encoder Seed-VC conditions on, which would mark the converter's own homework.
- Words: faster-whisper (CTranslate2, int8). Sung lyrics transcribe worse than speech, so read
  WER as "did the words survive", comparing runs, not as an absolute.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .score import ANALYSIS_SAMPLE_RATE, Models

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_WHISPER_MODEL = "small"
CACHE_DIR = Path.home() / ".cache" / "singing-voice"
INSTALL_HINT = "install the score extra: uv pip install -e '.[score]'"


class EcapaEmbedder:
    name = ECAPA_SOURCE

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        import torch  # noqa: PLC0415

        from speechbrain.inference.classifiers import EncoderClassifier  # noqa: PLC0415

        self._torch = torch
        self._model = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE,
            savedir=str(cache_dir / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"},
        )

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != ANALYSIS_SAMPLE_RATE:
            raise ValueError(f"expected {ANALYSIS_SAMPLE_RATE} Hz audio, got {sample_rate}")
        with self._torch.no_grad():
            wav = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
            return self._model.encode_batch(wav).squeeze().cpu().numpy()


class WhisperTranscriber:
    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.name = f"faster-whisper/{model_size}"
        self._model = WhisperModel(model_size, device="auto", compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        if sample_rate != ANALYSIS_SAMPLE_RATE:
            raise ValueError(f"expected {ANALYSIS_SAMPLE_RATE} Hz audio, got {sample_rate}")
        segments, info = self._model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=language,
            beam_size=5,
            # Stops a misheard line from being repeated through the rest of the clip.
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, info.language


def load_models(timbre: bool = True, words: bool = True, whisper_model: str = DEFAULT_WHISPER_MODEL) -> Models:
    """Load what is installed; record why anything is missing instead of failing."""
    models = Models()
    if timbre:
        try:
            models.embedder = EcapaEmbedder()
        except ImportError as exc:
            models.skipped["timbre"] = f"{exc.name or exc} is not installed; {INSTALL_HINT}"
        except Exception as exc:  # noqa: BLE001 - e.g. no network for the first download
            models.skipped["timbre"] = f"could not load {ECAPA_SOURCE}: {type(exc).__name__}: {exc}"
    if words:
        try:
            models.transcriber = WhisperTranscriber(whisper_model)
        except ImportError as exc:
            models.skipped["words"] = f"{exc.name or exc} is not installed; {INSTALL_HINT}"
        except Exception as exc:  # noqa: BLE001
            models.skipped["words"] = f"could not load Whisper {whisper_model}: {type(exc).__name__}: {exc}"
    return models


__all__ = ["DEFAULT_WHISPER_MODEL", "EcapaEmbedder", "WhisperTranscriber", "load_models"]
