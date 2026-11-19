# Seed-VC singing conversion worker for RunPod serverless. Build context: repository root.
FROM python:3.10-slim

ARG TORCH_VERSION=2.4.0
ARG TORCHAUDIO_VERSION=2.4.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
# Seed-VC (GPL-3.0), pinned to the newest upstream commit at the time (2025-04-20).
ARG SEED_VC_REF=51383efd921027683c89e5348211d93ff12ac2a8

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCHAUDIO_USE_SOUND_FILE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SEED_VC_HOME=/opt/seed-vc \
    SEED_VC_REF=${SEED_VC_REF} \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Plachtaa/seed-vc.git ${SEED_VC_HOME} \
 && git -C ${SEED_VC_HOME} checkout --quiet ${SEED_VC_REF}

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION}

COPY runpod_worker/requirements.txt /app/runpod_worker/requirements.txt
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir -r runpod_worker/requirements.txt '.[worker]' \
 && pip check

# Bake the weights (DiT f0 44k, whisper-small, BigVGAN 44k, RMVPE, CAMPPlus) into the image.
# Copy only the worker files needed to prefetch, not the handler.
COPY runpod_worker/__init__.py runpod_worker/engine.py runpod_worker/prefetch_weights.py /app/runpod_worker/
RUN python runpod_worker/prefetch_weights.py

# From here on, never reach Hugging Face at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY runpod_worker /app/runpod_worker

CMD ["python", "-u", "-m", "runpod_worker.handler"]
