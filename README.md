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

## Score a conversion

```bash
svtool score working/vocals_seedvc_r01.wav
```

Reads the source excerpt, reference and options from the `.json` sidecar `convert` wrote
and writes `working/vocals_seedvc_r01.score.json` (replaced on re-scoring). For a file that
did not come from `convert`, pass `--source` (and `--reference`, `--start-s`,
`--duration-s`, `--semitone-shift`).

| Section | What it measures | Want |
| --- | --- | --- |
| `melody` | pYIN f0 of output vs input on frames both sing, in cents (after the requested octave shift); share within 50 cents; gross (> 600 cents) errors; voiced/unvoiced agreement | ~0 cents, high agreement |
| `alignment` | lag of the output's energy envelope against the input's | 0 ms |
| `levels` | active RMS, peak, clipped samples of input, output, reference | output level ≈ input, no clipping |
| `spectrum` | level-independent band profile (low, low-mid, presence, sibilance, air; centroid) and band distances between the three | output closer to the reference than the input was |
| `timbre` | speaker-embedding cosine similarity (SpeechBrain ECAPA): output↔reference, output↔input, and the input↔reference baseline | up, down, `moved_toward_reference` > 0 |
| `words` | faster-whisper transcripts of input and output, WER of output against input | low WER |

`timbre` and `words` need the `score` extra (`uv pip install -e '.[score]'`, pulls torch)
and download their weights from Hugging Face on first use; without them those sections
say why they were skipped and the rest of the report is still written. Both models were
trained on speech, so read their numbers by comparing runs, not as absolutes. `summary`
holds the headline numbers.

## Test set and benchmark runs

`testsets/m1.json` names three fixed excerpts and the reference voice: a clean vocal, a
Demucs stem with bleed, and a stem with harmonies. Put the audio in `test_files/m1/`
(git-ignored) under the names the file uses, or edit the paths (relative to the JSON file)
and `start_s`/`duration_s` (at most 60 s each).

```bash
svtool bench run testsets/m1.json --label baseline --dry-run   # check files and excerpts
svtool bench run testsets/m1.json --label baseline              # convert + score every clip
svtool bench run testsets/m1.json --label steps50 --diffusion-steps 50
svtool bench compare working/bench/<baseline-run> working/bench/<steps50-run>
svtool bench rescore working/bench/<run>                        # e.g. after installing .[score]
```

A run goes in `working/bench/<timestamp>_<label>/` with `<clip>.wav`, its sidecar,
`<clip>.score.json`, `run.json` and `summary.json`. A clip that fails is recorded and the
run continues. `compare` prints before/after/delta per clip and metric and marks each
change better or worse.

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

### Preflight the RunPod image locally

With Docker and Buildx running, validate a committed worker change before pushing:

```bash
./scripts/runpod-preflight             # HEAD
./scripts/runpod-preflight main        # optional: a branch, tag, or commit
```

This builds the same Dockerfile RunPod uses, from the selected Git commit (uncommitted
changes and local audio are excluded). It targets `linux/amd64` even on Apple Silicon,
installs and checks dependencies, prefetches model weights, and smoke-tests the handler
import with networking disabled. The build needs internet access and several GB of
free Docker disk space; under emulation it can take a while. No RunPod credentials or GPU
are needed. After deployment, run a short real conversion to verify GPU inference.

## Older chunked flow

`svtool preprocess` and `svtool stitch` are the earlier fixed-window chunker and stitcher.
The worker no longer takes their manifests (Seed-VC windows long audio itself); they stay
until phrase-based splitting replaces them.

## Development

```bash
uv sync --locked
uv tool install 'lefthook==2.2.2'
lefthook install
uv run --locked pytest
```

Lefthook checks staged whitespace before commits and runs the locked pytest suite
before pushes. The Docker preflight is opt-in, so routine pushes don't rebuild the
worker image.

Tests cover the transport, the handler contract (with a fake engine), option validation,
level matching, the client, the fixed-window split and reassembly, the scoring metrics (on
synthetic sung lines, with fake embedding and transcription models) and the benchmark
runner. The model itself is checked by running a real conversion and scoring it.
