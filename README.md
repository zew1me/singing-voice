# Singing Voice Pipeline

Librosa-based preprocessing + stitching utilities that surround a RunPod hosted Seed-VC inference worker. The repository contains:

- **Step A** – local preprocessor (`svtool preprocess`)
- **Step B** – reference RunPod worker (`runpod_worker/seedvc_worker.py`)
- **Step C** – local stitcher (`svtool stitch`)

All generated artifacts live inside the `./working` directory unless you pass an explicit `--output`/`--manifest-path`.

## Requirements & installation

This project is managed with [`uv`](https://github.com/astral-sh/uv). Install dependencies locally:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
# add HTTP client helpers for RunPod submissions
uv pip install -e .[runpod]
```

## Usage

### Step A – Preprocess locally

```bash
svtool preprocess input.wav \
  --workdir ./working \
  --chunk-seconds 12 \
  --overlap-seconds 1.5 \
  --silence-threshold-db -40
```

The command trims silence, chunks audio, base64-encodes each chunk, and writes a manifest JSON to `working/<stem_timestamp>/manifest.json` (or the path you pass via `--manifest-path`).

### Step B – Seed-VC conversion on RunPod

- Build a RunPod image that contains Seed-VC and copy `runpod_worker/seedvc_worker.py` into it.
- The worker loads the manifest from Step A, performs Seed-VC inference for each chunk, and responds with:

```json
{
  "sample_rate": 16000,
  "converted_chunks": [{"chunk_id": "...", "audio_b64": "...", "start": 0, "end": 18000}]
}
```

- To call the worker from your laptop (HTTP serverless endpoint):

```bash
svtool submit-runpod working/<session>/manifest.json \
  --endpoint https://api.runpod.io/... \
  --api-key $RUNPOD_API_KEY \
  --output working/converted_manifest.json
```

### Step C – Stitch locally

```bash
svtool stitch working/converted_manifest.json --output working/final.wav
```

The stitcher respects the original chunk offsets, crossfades overlaps, and emits a WAV file at the requested sample rate.

## Customisation

- All commands expose knobs for chunk size, overlap, silence threshold, minimum chunk duration, crossfade seconds, etc.
- Override the working directory via `--workdir ./another/path`.
- `submit-runpod` only imports `requests` when executed, so local preprocessing/stitching remain lightweight.

## RunPod worker notes

`runpod_worker/seedvc_worker.py` contains a pure function `convert_manifest` that expects:

- a parsed manifest (`PreprocessManifest`)
- a reference/target voice WAV path
- a Seed-VC checkpoint path

The helper decodes chunk audio, calls `seed_vc.inference.inference_pipeline`, returns a `ConvertedManifest`, and reuses the same JSON format consumed by the stitcher.

## Seed-VC references

- [Seed-VC singing voice conversion overview](https://deepwiki.com/Plachtaa/seed-vc/4.2-singing-voice-conversion) – walkthrough of checkpoints, inference scripts, and expected inputs the worker relies on.
- [Seed-VC training & fine-tuning](https://deepwiki.com/Plachtaa/seed-vc/5-training-and-fine-tuning) – outlines how to customize checkpoints that you later bundle into the RunPod image.

## Next steps

- Generate Dockerfiles for the RunPod worker image and/or a CPU-only local helper.
- Extend the CLI to parallelize chunk submissions across multiple RunPod pods.
- Integrate storage upload/download (e.g., S3) around the manifest JSON if remote orchestration is needed.
