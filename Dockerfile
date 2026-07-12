FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/stereo2spatial

COPY pyproject.toml README.md LICENSE ./
COPY stereo2spatial ./stereo2spatial
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[dev]" \
    && python - <<'PY'
import torch
import torchaudio
import soundfile
import accelerate
import safetensors

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("torchaudio", torchaudio.__version__)
print("soundfile", soundfile.__version__)
print("accelerate", accelerate.__version__)
print("safetensors", safetensors.__version__)
PY

CMD ["bash"]
