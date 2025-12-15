FROM python:3.10-slim

ARG TORCH_VERSION=2.4.1
ARG TORCHAUDIO_VERSION=2.4.1
ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/cu124

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCHAUDIO_USE_SOUND_FILE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV SEED_VC_HOME=/opt/seed-vc

COPY pyproject.toml README.md uv.lock /app/
COPY src /app/src
COPY runpod_worker /app/runpod_worker

RUN git clone https://github.com/Plachtaa/seed-vc.git ${SEED_VC_HOME}

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url ${TORCH_WHEEL_URL} \
        torch==${TORCH_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} && \
    pip install --no-cache-dir -r ${SEED_VC_HOME}/requirements.txt && \
    pip install --no-cache-dir -r runpod_worker/requirements.txt && \
    pip install --no-cache-dir .

ENV PYTHONPATH=${SEED_VC_HOME}:${PYTHONPATH}

CMD ["python", "-u", "runpod_worker/handler.py"]
