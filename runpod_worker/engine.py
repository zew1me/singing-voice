"""Seed-VC singing conversion engine (44.1 kHz, f0-conditioned).

Loads upstream Seed-VC's singing model once and converts whole vocals with upstream's own
30 s context windowing. This is a port of upstream ``inference.py``'s ``main()`` that takes
arrays instead of file paths, loads only the f0 model (upstream's ``SeedVCWrapper`` loads
both models and both vocoders, and its ``convert_voice`` is a generator even when not
streaming), and keeps the song's key: the source melody is never moved to the reference's
median pitch, only by whole octaves when asked.

Upstream resolves its checkpoints relative to the working directory, so loading happens
inside ``seed_vc_home``; the image's build step loads the engine once the same way to bake
the weights in.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

OUTPUT_SAMPLE_RATE = 44_100
HOP_LENGTH = 512
OVERLAP_FRAMES = 16
CONTEXT_SECONDS = 30
REFERENCE_MAX_SECONDS = 25
ALLOWED_SEMITONE_SHIFTS = (-24, -12, 0, 12, 24)
# The model sings at the reference's level, whatever the source's level, and its vocoder
# clamps at full scale. A reference at -19.7 dBFS active RMS came out at -15 dBFS and clipped;
# at -32 dBFS it peaked near -4 dBFS. So the reference goes in at this level and the result
# is then gain-matched to the source.
REFERENCE_RMS_DB = -32.0
OUTPUT_PEAK_LIMIT = 0.98


class EngineError(ValueError):
    """A request the engine cannot honour (bad options or unusable audio)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ConvertOptions:
    diffusion_steps: int = 40
    inference_cfg_rate: float = 0.7
    semitone_shift: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConvertOptions":
        data = dict(data or {})
        unknown = set(data) - {"diffusion_steps", "inference_cfg_rate", "semitone_shift"}
        if unknown:
            raise EngineError("UNKNOWN_OPTION", f"unknown options: {sorted(unknown)}")
        options = cls(
            diffusion_steps=int(data.get("diffusion_steps", cls.diffusion_steps)),
            inference_cfg_rate=float(data.get("inference_cfg_rate", cls.inference_cfg_rate)),
            semitone_shift=int(data.get("semitone_shift", cls.semitone_shift)),
        )
        options.validate()
        return options

    def validate(self) -> None:
        if not 1 <= self.diffusion_steps <= 200:
            raise EngineError("BAD_OPTION", "diffusion_steps must be between 1 and 200")
        if not 0.0 <= self.inference_cfg_rate <= 1.0:
            raise EngineError("BAD_OPTION", "inference_cfg_rate must be between 0 and 1")
        if self.semitone_shift not in ALLOWED_SEMITONE_SHIFTS:
            raise EngineError(
                "BAD_OPTION",
                "semitone_shift must be a whole number of octaves "
                f"{ALLOWED_SEMITONE_SHIFTS} so the vocal stays in the song's key",
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diffusion_steps": self.diffusion_steps,
            "inference_cfg_rate": self.inference_cfg_rate,
            "semitone_shift": self.semitone_shift,
        }


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad or trim to exactly ``length`` samples."""
    if audio.shape[0] >= length:
        return audio[:length]
    return np.pad(audio, (0, length - audio.shape[0]))


def output_length(num_samples: int, sample_rate: int) -> int:
    return int(round(num_samples * OUTPUT_SAMPLE_RATE / sample_rate))


def active_rms_db(audio: np.ndarray, sample_rate: int, window_s: float = 0.05, range_db: float = 30.0) -> Optional[float]:
    """RMS level (dBFS) over the 50 ms windows within ``range_db`` of the loudest: the level while singing."""
    audio = to_mono(audio)
    size = max(1, int(sample_rate * window_s))
    frames = len(audio) // size
    if frames == 0:
        return None
    power = np.mean(audio[: frames * size].reshape(frames, size).astype(np.float64) ** 2, axis=1)
    if not np.any(power > 0):
        return None
    loud = power[power >= power.max() * 10 ** (-range_db / 10)]
    return float(10 * np.log10(np.mean(loud)))


def gain_to(audio: np.ndarray, sample_rate: int, target_db: float) -> float:
    """Linear gain that brings ``audio``'s active RMS to ``target_db``."""
    level = active_rms_db(audio, sample_rate)
    return 1.0 if level is None else float(10 ** ((target_db - level) / 20))


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _crossfade(previous: np.ndarray, current: np.ndarray, overlap: int) -> np.ndarray:
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    span = min(overlap, len(current))
    current[:span] = current[:span] * fade_in[:span] + (previous[-overlap:] * fade_out)[:span]
    return current


class SeedVCEngine:
    def __init__(
        self,
        seed_vc_home: Path | str,
        checkpoint: Optional[Path | str] = None,
        config: Optional[Path | str] = None,
        fp16: Optional[bool] = None,
    ) -> None:
        home = Path(seed_vc_home).resolve()
        if not (home / "inference.py").is_file():
            raise FileNotFoundError(f"no Seed-VC checkout at {home}")
        if (checkpoint is None) != (config is None):
            raise ValueError("a fine-tuned checkpoint needs its config, and the other way round")
        if str(home) not in sys.path:
            sys.path.insert(0, str(home))

        started = time.time()
        with _working_directory(home):
            import inference as upstream  # noqa: PLC0415 - upstream picks device and cache at import

            self._torch = upstream.torch
            self._torchaudio = upstream.torchaudio
            self.device = upstream.device
            self.fp16 = (self.device.type == "cuda") if fp16 is None else fp16
            args = argparse.Namespace(
                f0_condition=True,
                fp16=self.fp16,
                checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()),
                config=None if config is None else str(Path(config).resolve()),
            )
            (
                self._model,
                self._semantic_fn,
                self._f0_fn,
                self._vocoder_fn,
                self._campplus,
                self._mel_fn,
                mel_args,
            ) = upstream.load_models(args)
        if mel_args["sampling_rate"] != OUTPUT_SAMPLE_RATE:
            raise RuntimeError(f"expected a 44.1 kHz model, got {mel_args['sampling_rate']} Hz")
        self.checkpoint = args.checkpoint or "Plachta/Seed-VC:DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
        self.load_seconds = round(time.time() - started, 1)
        self.last_report: Dict[str, Any] = {}

    def _autocast(self):
        if self.device.type == "cuda":
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16 if self.fp16 else self._torch.float32)
        return contextlib.nullcontext()

    def _semantic(self, waves_16k):
        """Whisper features; sources over 30 s go through upstream's overlapping windows."""
        torch = self._torch
        if waves_16k.size(-1) <= 16000 * 30:
            return self._semantic_fn(waves_16k)
        overlap_s = 5
        pieces, buffer, traversed = [], None, 0
        while traversed < waves_16k.size(-1):
            if buffer is None:
                chunk = waves_16k[:, traversed : traversed + 16000 * 30]
            else:
                chunk = torch.cat([buffer, waves_16k[:, traversed : traversed + 16000 * (30 - overlap_s)]], dim=-1)
            features = self._semantic_fn(chunk)
            pieces.append(features if traversed == 0 else features[:, 50 * overlap_s :])
            buffer = chunk[:, -16000 * overlap_s :]
            traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlap_s
        return torch.cat(pieces, dim=1)

    def convert(
        self,
        source: np.ndarray,
        source_sr: int,
        reference: np.ndarray,
        reference_sr: int,
        options: Optional[ConvertOptions] = None,
    ) -> np.ndarray:
        """Sing ``source`` in the voice of ``reference``. Mono float32 at 44.1 kHz, same duration as ``source``."""
        import librosa  # noqa: PLC0415

        options = options or ConvertOptions()
        options.validate()
        torch, torchaudio = self._torch, self._torchaudio
        sr = OUTPUT_SAMPLE_RATE
        want = output_length(len(to_mono(source)), source_sr)

        source_44k = librosa.resample(to_mono(source), orig_sr=source_sr, target_sr=sr) if source_sr != sr else to_mono(source)
        reference_44k = librosa.resample(to_mono(reference), orig_sr=reference_sr, target_sr=sr) if reference_sr != sr else to_mono(reference)
        if len(source_44k) < sr // 2:
            raise EngineError("SOURCE_TOO_SHORT", "source must be at least 0.5 s long")
        if len(reference_44k) < sr:
            raise EngineError("REFERENCE_TOO_SHORT", "reference must be at least 1 s long")
        if not np.any(np.abs(source_44k) > 1e-4):
            self.last_report = {"silent_source": True}
            return np.zeros(want, dtype=np.float32)
        reference_44k = reference_44k[: sr * REFERENCE_MAX_SECONDS]
        reference_gain = gain_to(reference_44k, sr, REFERENCE_RMS_DB)
        reference_44k = (reference_44k * reference_gain).astype(np.float32)

        device = self.device
        with torch.no_grad():
            source_t = torch.tensor(source_44k).unsqueeze(0).float().to(device)
            reference_t = torch.tensor(reference_44k).unsqueeze(0).float().to(device)

            source_16k = torchaudio.functional.resample(source_t, sr, 16000)
            reference_16k = torchaudio.functional.resample(reference_t, sr, 16000)
            s_alt = self._semantic(source_16k)
            s_ori = self._semantic_fn(reference_16k)

            mel = self._mel_fn(source_t)
            mel2 = self._mel_fn(reference_t)
            target_lengths = torch.LongTensor([mel.size(2)]).to(device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)

            feat2 = torchaudio.compliance.kaldi.fbank(reference_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self._campplus(feat2.unsqueeze(0))

            f0_ori = torch.from_numpy(self._f0_fn(reference_16k[0], thred=0.03)).float().to(device)[None]
            f0_alt = torch.from_numpy(self._f0_fn(source_16k[0], thred=0.03)).float().to(device)[None]
            shifted_f0_alt = f0_alt.clone()
            if options.semitone_shift:
                voiced = f0_alt > 1
                shifted_f0_alt[voiced] = f0_alt[voiced] * 2 ** (options.semitone_shift / 12)

            cond, *_ = self._model.length_regulator(s_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
            prompt, *_ = self._model.length_regulator(s_ori, ylens=target2_lengths, n_quantizers=3, f0=f0_ori)

            max_source_window = sr // HOP_LENGTH * CONTEXT_SECONDS - mel2.size(2)
            overlap_wave = OVERLAP_FRAMES * HOP_LENGTH
            processed, pieces, previous = 0, [], None
            while processed < cond.size(1):
                chunk_cond = cond[:, processed : processed + max_source_window]
                last = processed + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt, chunk_cond], dim=1)
                with self._autocast():
                    vc_target = self._model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(device),
                        mel2,
                        style2,
                        None,
                        options.diffusion_steps,
                        inference_cfg_rate=options.inference_cfg_rate,
                    )
                    vc_target = vc_target[:, :, mel2.size(-1) :]
                wave = self._vocoder_fn(vc_target.float()).squeeze().float().cpu().numpy()
                if previous is None and last:
                    pieces.append(wave)
                    break
                if previous is None:
                    pieces.append(wave[:-overlap_wave])
                elif last:
                    pieces.append(_crossfade(previous, wave, overlap_wave))
                    break
                else:
                    pieces.append(_crossfade(previous, wave[:-overlap_wave], overlap_wave))
                previous = wave[-overlap_wave:]
                processed += vc_target.size(2) - OVERLAP_FRAMES

        converted = fit_length(np.concatenate(pieces).astype(np.float32), want)
        source_level = active_rms_db(source_44k, sr)
        raw_level = active_rms_db(converted, sr)
        raw_peak = float(np.max(np.abs(converted)))
        gain = gain_to(converted, sr, source_level) if source_level is not None else 1.0
        peak = raw_peak * gain
        if peak > OUTPUT_PEAK_LIMIT:
            gain *= OUTPUT_PEAK_LIMIT / peak
        self.last_report = {
            "reference_gain_db": round(20 * np.log10(reference_gain), 1),
            "source_active_rms_db": None if source_level is None else round(source_level, 1),
            "model_output_active_rms_db": None if raw_level is None else round(raw_level, 1),
            "model_output_clipped_samples": int(np.sum(np.abs(converted) >= 0.999)),
            "output_gain_db": round(20 * np.log10(gain), 1),
        }
        return (converted * gain).astype(np.float32)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.checkpoint,
            "device": str(self.device),
            "fp16": self.fp16,
            "load_seconds": self.load_seconds,
        }


__all__ = [
    "ALLOWED_SEMITONE_SHIFTS",
    "ConvertOptions",
    "EngineError",
    "OUTPUT_SAMPLE_RATE",
    "SeedVCEngine",
    "active_rms_db",
    "gain_to",
    "fit_length",
    "output_length",
    "to_mono",
]
