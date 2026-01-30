# 1. Use base image that matches server CUDA version (12.8) and includes full development toolchain
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Critical: Tell NVIDIA Container Runtime to mount driver libraries for Video Codec SDK (NVENC/NVDEC)
# Without this, only 'compute' and 'utility' libs are mounted, causing "libnvidia-encode.so not found".
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

ENV DEBIAN_FRONTEND=noninteractive

# 2. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Miniconda
WORKDIR /tmp
RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:$PATH"

# 4. Accept Conda terms and create the target environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n flashvsr python=3.11.13 -y

# 5. Subsequent commands run inside the flashvsr environment
SHELL ["conda", "run", "-n", "flashvsr", "/bin/bash", "-c"]

# 6. Set workdir and copy project (model weights are excluded by .dockerignore)
WORKDIR /workspace/FlashVSR-Pro
COPY . .

# 7. Create necessary directories for mounted models, VAE variants, and inputs
RUN mkdir -p /workspace/FlashVSR-Pro/models/FlashVSR-v1.1 \
    && mkdir -p /workspace/FlashVSR-Pro/models/prompt_tensor \
    && mkdir -p /workspace/FlashVSR-Pro/inputs \
    && mkdir -p /workspace/FlashVSR-Pro/results

# 8. Install specific PyTorch build (important)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 9. Install FlashVSR core dependencies
RUN pip install -e . \
    && pip install -r requirements.txt

# 10. Build and install Block-Sparse-Attention
# This step automates the setup of the sparse attention kernel
WORKDIR /workspace/FlashVSR-Pro/Block-Sparse-Attention
RUN pip install packaging ninja
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Apply optimizations for better compatibility and performance
RUN python setup.py install

# 11. Clean caches to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    conda clean -ya && \
    pip cache purge

# 12. Initialize git-lfs (prepare for runtime model downloads)
WORKDIR /workspace/FlashVSR-Pro
RUN git lfs install

# 13. Ensure entrypoint is executable (defensive)
RUN chmod +x entrypoint.sh

# 14. Set container entrypoint
ENTRYPOINT ["/workspace/FlashVSR-Pro/entrypoint.sh"]

# 15. Default to interactive bash if no command provided
CMD ["/bin/bash"]