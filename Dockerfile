# ---------------------------------------------------------------------------
# FramePack F1 — RunPod Serverless Dockerfile
#
# Uses RunPod Model Cache: models are pulled from
#   Floppyshy/framepack-runpod-cache
# and land under /runpod-volume/huggingface-cache/hub/ at runtime.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /workspace

# System deps for opencv, video encoding, and python
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch + torchvision (CUDA 12.4)
RUN pip install --no-cache-dir \
    "torch>=2.6.0" \
    "torchvision>=0.21.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Python dependencies
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

CMD ["python3", "-u", "handler.py"]
