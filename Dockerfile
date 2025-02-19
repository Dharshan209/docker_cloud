# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the repository
RUN git clone -b totoro4 https://github.com/camenduru/ComfyUI .

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    torchsde \
    einops \
    diffusers \
    accelerate \
    xformers==0.0.28.post2 \
    gradio==3.50.2 \
    python-multipart==0.0.12

# Create directories for models
RUN mkdir -p models/checkpoints models/loras

# Download the model checkpoint
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8-all-in-one.safetensors \
    -d models/checkpoints -o flux1-dev-fp8-all-in-one.safetensors

# Copy the application code
COPY app.py /workspace/

# Expose port
EXPOSE 7860

# Start command
CMD ["python3", "app.py"]