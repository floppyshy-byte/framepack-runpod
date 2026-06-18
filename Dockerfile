# ---------------------------------------------------------------------------
# FramePack F1 — RunPod Serverless Dockerfile
#
# Uses RunPod Model Cache: models are pulled from
#   Floppyshy/framepack-runpod-cache
# and land under /runpod-volume/huggingface-cache/hub/ at runtime.
# ---------------------------------------------------------------------------

FROM runpod/pytorch:1.0.6-cu1281-torch260-ubuntu2204

WORKDIR /workspace

# System deps for opencv and video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (PyTorch already included in base image)
RUN pip install --no-cache-dir \
    "accelerate==1.6.0" \
    "diffusers==0.33.1" \
    "transformers==4.46.2" \
    "sentencepiece==0.2.0" \
    "pillow==11.1.0" \
    "av==12.1.0" \
    "numpy==1.26.2" \
    "scipy==1.12.0" \
    "requests==2.31.0" \
    "torchsde==0.2.6" \
    "einops" \
    "opencv-contrib-python" \
    "safetensors" \
    "runpod"

# Copy application code
COPY diffusers_helper/ diffusers_helper/
COPY handler.py .

# RunPod serverless config
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface-cache

CMD ["python", "-u", "handler.py"]
