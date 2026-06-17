# ---------------------------------------------------------------------------
# FramePack F1 — RunPod Serverless Dockerfile
#
# Uses RunPod Model Cache: models are pulled from
#   Floppyshy/framepack-runpod-cache
# and land under /runpod-volume/huggingface-cache/hub/ at runtime.
# ---------------------------------------------------------------------------

FROM runpod/pytorch:2.6.0-py3.10-cuda12.6.3-devel-ubuntu22.04

WORKDIR /app

# System deps for opencv and video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (install torch-adjacent deps first, then the rest)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        runpod \
        -r requirements.txt

# Copy application code
COPY diffusers_helper/ diffusers_helper/
COPY handler.py .

# RunPod serverless config
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
