# Singing Voice Pipeline

Make a sung vocal sound like a reference voice: same melody, timing and words, different
singer. Conversion runs [Seed-VC](https://github.com/Plachtaa/seed-vc)'s 44.1 kHz
pitch-conditioned singing model on a RunPod serverless GPU worker; the local `svtool` CLI
sends the audio and saves the result. The roadmap is in
[issue #1](https://github.com/zew1me/singing-voice/issues/1).

## Requirements & installation

This project is managed with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
uv pip install -e .[runpod]   # requests, for talking to RunPod
```

## Convert a vocal

```bash
export RUNPOD_API_KEY=...            # RunPod API key
export RUNPOD_ENDPOINT_ID=...        # the Seed-VC serverless endpoint
svtool convert vocals.wav --reference target_voice.wav --start-s 45 --duration-s 30
```

- `SOURCE` is the sung vocal (a Demucs stem is fine; bleed and backing vocals come through
  as artefacts until input cleaning lands).
- `--reference` is a clean clip of the target voice. The first 25 s are used.
- Writes `working/<source>_seedvc_rNN.wav` (24-bit, 44.1 kHz, mono) plus a `.json` sidecar
  with the options, levels and timings. Existing files are never overwritten.
- The result has exactly the source excerpt's duration and lines up with it sample for
  sample, so it can replace the original clip at the same position.
- `--semitone-shift` takes whole octaves only (`-24, -12, 0, 12, 24`) so the vocal stays in
  the song's key. The source's melody is never moved toward the reference's pitch range.
- `--diffusion-steps` (default 40; upstream suggests 30-50 for singing) and
  `--inference-cfg-rate` (default 0.7) pass through to Seed-VC.
- Levels: the model sings at the reference's level whatever the source's level, and its
  vocoder clamps at full scale. The worker brings the reference to -32 dBFS active RMS
  first, then gain-matches the result to the source.

Audio travels inline as base64 FLAC. RunPod caps a `/run` request at 10 MB, so about a
minute of vocal fits; `convert` refuses larger requests. The worker already accepts
`{"url": ...}` inputs and an `output.put_url` (presigned URLs) for full songs; the client
side of that needs a storage bucket.

## RunPod worker

`runpod_worker/` is built by RunPod from this repo (`runpod_worker/Dockerfile`, build
context: repository root):

- `engine.py` loads Seed-VC's f0 44.1 kHz model once per worker and converts whole vocals
  with upstream's own 30 s windowing.
- `handler.py` is the serverless entrypoint. Job input:

  ```json
  {
    "source": {"b64": "<base64 audio file>"},
    "reference": {"url": "https://..."},
    "output": {"put_url": "https://..."},
    "options": {"diffusion_steps": 40, "inference_cfg_rate": 0.7, "semitone_shift": 0}
  }
  ```

  Each audio input is `{"b64": ...}` or `{"url": ...}`; `output` is optional (default:
  inline). It returns `sample_rate`, `num_samples`, `audio_b64` or `output_url`, `levels`,
  `engine` and `timing_s`, or `{"error": "CODE: message"}`.
- The image pins Seed-VC to a commit (`SEED_VC_REF`), installs a curated dependency list
  instead of upstream's `requirements.txt`, and bakes all weights in at build time, so cold
  starts never download from Hugging Face.
- Optional env: `SEEDVC_CHECKPOINT` + `SEEDVC_CONFIG` for a fine-tuned model.

Seed-VC is GPL-3.0.

## Older chunked flow

`svtool preprocess` and `svtool stitch` are the earlier fixed-window chunker and stitcher.
The worker no longer takes their manifests (Seed-VC windows long audio itself); they stay
until phrase-based splitting replaces them.

## Development

```bash
uv run pytest
```

Tests cover the transport, the handler contract (with a fake engine), option validation,
level matching and the client. The model itself is checked by running a real conversion.
