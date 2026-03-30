# mievformer Docker test environment
# Base image: PyTorch with CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install mievformer from PyPI with compatible dependencies
# Note: numpy<2 required for PyTorch compatibility, tensorboard for logging
# igraph required for Leiden clustering in scanpy
# Install mievformer first, then downgrade numpy to fix compatibility
RUN pip install --no-cache-dir mievformer tensorboard igraph leidenalg && \
    pip install --no-cache-dir "numpy<2"

# Copy tutorial files
COPY run_tutorial.py .
COPY tutorial_model.pth .

# Default command: run tutorial
CMD ["python", "run_tutorial.py"]
