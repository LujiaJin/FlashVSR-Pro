# FlashVSR-Pro Installation Guide

This guide provides detailed installation instructions for different platforms and scenarios.

## Table of Contents
- [Platform Requirements](#platform-requirements)
- [Linux Installation](#linux-installation)
- [Windows Installation](#windows-installation)
- [macOS](#macos)
- [Cloud Platforms](#cloud-platforms)
- [Manual Installation (Advanced)](#manual-installation-advanced)
- [Troubleshooting](#troubleshooting)

---

## Platform Requirements

### Supported Operating Systems
- **Linux**: Ubuntu 20.04/22.04, Debian 11+, CentOS 8+, other modern distributions (Primary platform)
- **Windows**: Windows 10 (21H2+) or Windows 11 via WSL 2 or Docker Desktop (Secondary support)
- **macOS**: Not supported (NVIDIA CUDA required)

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 8.0+ (Ampere architecture or newer)
  - ✅ Supported: RTX 3060/3070/3080/3090/4070/4080/4090, A100, H100, H200, A6000, RTX 6000 Ada
  - ❌ Not Supported: GTX series (too old), AMD GPUs, Intel GPUs, Apple Silicon
- **VRAM**: 
  - 8GB minimum (with `--tile-dit --tile-vae`)
  - 16GB recommended for 1080p/1440p
  - 24GB+ for 4K processing
- **CUDA**: Version 11.6 or higher (12.x recommended)
- **RAM**: 16GB+ system memory recommended
- **Storage**: 10GB+ for models and dependencies

---

## Linux Installation

### Method 1: Docker (Recommended)

Docker provides the most reliable installation experience with all dependencies pre-configured.

#### Prerequisites
```bash
# Update package list
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
# Log out and back in for this to take effect

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install git-lfs for model downloads
sudo apt-get install -y git-lfs
git lfs install
```

#### Build and Run
```bash
# Clone the repository
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro

# Build Docker image (takes 15-30 minutes)
docker build -t flashvsr-pro:latest .

# Download model weights
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1

# Run the container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/FlashVSR-Pro \
  flashvsr-pro:latest

# Inside container, verify installation
python infer.py --help
```

### Method 2: Manual Installation (Advanced)

For users who prefer not to use Docker or need custom configurations.

#### Prerequisites
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs curl wget ffmpeg

# Install CUDA Toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads
# Or use existing CUDA installation

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
```

#### Setup Environment
```bash
# Clone repository
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro
git lfs install

# Create conda environment
conda create -n flashvsr python=3.11 -y
conda activate flashvsr

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install FlashVSR dependencies
pip install -e .
pip install -r requirements.txt

# Build and install Block-Sparse-Attention
cd Block-Sparse-Attention
pip install packaging ninja
python setup.py install
cd ..

# Download model weights
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1

# Verify installation
python infer.py --help
```

---

## Windows Installation

### Method 1: Docker Desktop with WSL 2 (Strongly Recommended)

This provides the best compatibility and ease of installation on Windows.

#### Prerequisites

1. **Install WSL 2**
   ```powershell
   # Run in PowerShell as Administrator
   wsl --install
   # Restart computer if prompted
   ```

2. **Install Ubuntu in WSL 2**
   ```powershell
   wsl --install -d Ubuntu-22.04
   # Set up username and password when prompted
   ```

3. **Update Windows GPU Driver**
   - Download latest NVIDIA driver from [NVIDIA website](https://www.nvidia.com/download/index.aspx)
   - Install the Windows driver (NOT Linux driver)
   - The Windows driver provides CUDA support for WSL 2

4. **Verify GPU in WSL**
   ```bash
   # Open Ubuntu in WSL 2
   nvidia-smi
   # Should display your GPU information
   ```

5. **Install Docker Desktop for Windows**
   - Download from [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
   - During installation, ensure "Use WSL 2 instead of Hyper-V" is selected
   - Enable WSL 2 integration for Ubuntu in Docker Desktop settings

6. **Install git-lfs in WSL**
   ```bash
   # In WSL Ubuntu terminal
   sudo apt-get update
   sudo apt-get install -y git-lfs
   git lfs install
   ```

#### Build and Run (in WSL 2 Ubuntu Terminal)

```bash
# All following commands run in WSL 2 Ubuntu terminal, NOT PowerShell/CMD

# Navigate to Windows filesystem (optional)
# cd /mnt/c/Users/YourUsername/

# Clone repository
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro

# Build Docker image
docker build -t flashvsr-pro:latest .

# Download model weights
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1

# Run container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/FlashVSR-Pro \
  flashvsr-pro:latest

# Inside container, verify
python infer.py --help
```

### Method 2: Native WSL 2 Installation (Alternative)

Skip Docker and run directly in WSL 2 Ubuntu.

```bash
# In WSL 2 Ubuntu terminal
# Follow the "Linux Installation - Method 2: Manual Installation" guide above
# All commands are the same as Linux
```

### Method 3: Native Windows Installation (Not Recommended)

⚠️ **Warning**: This is advanced and not officially supported. Many users report compilation issues.

#### Requirements
- Windows 10/11
- Visual Studio 2019 or later with "Desktop development with C++" workload
- CUDA Toolkit for Windows (12.x recommended)
- Git for Windows with Git LFS

#### Steps
```powershell
# Clone repository
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro

# Create conda environment
conda create -n flashvsr python=3.11 -y
conda activate flashvsr

# Install PyTorch with CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Attempt to build Block-Sparse-Attention
cd Block-Sparse-Attention
pip install packaging ninja
$env:DISTUTILS_USE_SDK=1
python setup.py install
# This step often fails on Windows due to compilation issues
```

**Common Issues on Native Windows**:
- CUDA compilation errors
- Missing Visual Studio components
- Incompatible CUDA/PyTorch versions
- Path issues with build tools

**If you encounter issues, strongly consider using Docker Desktop with WSL 2 instead.**

---

## macOS

❌ **FlashVSR-Pro is not compatible with macOS.**

**Reason**: FlashVSR-Pro requires NVIDIA CUDA for GPU acceleration. macOS does not support NVIDIA GPUs, and Apple Silicon (M1/M2/M3) uses a different architecture incompatible with CUDA.

**Alternatives**:
- Use a cloud service with NVIDIA GPUs (Google Colab, AWS, RunPod, Vast.ai)
- Dual-boot Linux on Mac with Intel processors and external NVIDIA eGPU (complex setup)
- Use a remote Linux server or workstation

---

## Cloud Platforms

FlashVSR-Pro works excellently on cloud platforms with NVIDIA GPUs.

### Google Colab

```python
# In a Colab notebook with GPU runtime (T4, A100)
!git clone https://github.com/LujiaJin/FlashVSR-Pro.git
%cd FlashVSR-Pro

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -e .
!pip install -r requirements.txt

# Build Block-Sparse-Attention
%cd Block-Sparse-Attention
!pip install packaging ninja
!python setup.py install
%cd ..

# Download models
!git lfs install
!git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1

# Run inference
!python infer.py -i /content/input.mp4 -o ./results/ --mode tiny
```

### AWS EC2 with GPU

Use Deep Learning AMI or standard Ubuntu AMI with NVIDIA GPU instances (g4dn, g5, p3, p4).

```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-instance-ip

# Follow Linux Installation instructions
# Use Docker method or manual installation
```

### Other Cloud Services (RunPod, Vast.ai, Lambda Labs)

Most cloud GPU services provide Ubuntu-based images. Follow the Linux installation guide.

---

## Manual Installation (Advanced)

For development, custom configurations, or non-Docker deployments.

### Key Dependencies
- Python 3.10 or 3.11
- PyTorch 2.0+ with CUDA support
- CUDA 11.6+ (12.x recommended)
- FFmpeg with hardware encoding support
- Block-Sparse-Attention (custom CUDA kernels)

### Installation Steps

1. **Set up Python environment** (conda or venv)
2. **Install PyTorch** matching your CUDA version
3. **Install FlashVSR-Pro**: `pip install -e .`
4. **Install requirements**: `pip install -r requirements.txt`
5. **Build Block-Sparse-Attention**: 
   ```bash
   cd Block-Sparse-Attention
   pip install packaging ninja
   python setup.py install
   ```
6. **Download models** with git-lfs
7. **Test**: `python infer.py --help`

### Environment Variables

```bash
# Optional: Specify CUDA architectures for Block-Sparse-Attention
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90;100"  # for Ampere, Hopper, Blackwell

# Optional: Control compilation parallelism
export MAX_JOBS=4
export NVCC_THREADS=4

# Optional: Force rebuild instead of using precompiled wheels
export BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE
```

---

## Troubleshooting

### "CUDA out of memory" errors
```bash
# Use tiling to reduce VRAM usage
python infer.py -i input.mp4 -o output/ --tile-dit --tile-vae --tile-size 128
```

### "nvidia-smi: command not found"
- Install NVIDIA GPU drivers
- On WSL 2: Install Windows NVIDIA driver, not Linux driver

### Docker: "could not select device driver"
```bash
# Restart docker daemon
sudo systemctl restart docker

# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Block-Sparse-Attention compilation fails
- Verify CUDA version: `nvcc --version`
- Ensure CUDA_HOME is set: `echo $CUDA_HOME`
- Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- Versions should match (major version)

### Git LFS not downloading models
```bash
# Install git-lfs first
sudo apt-get install git-lfs  # or: brew install git-lfs (macOS)
git lfs install

# Then clone or pull
git lfs pull
```

### WSL 2: GPU not accessible
- Update Windows to latest version
- Install latest NVIDIA driver (Windows version)
- Check: `nvidia-smi` in WSL should work
- Restart WSL: `wsl --shutdown` in PowerShell, then reopen

### FFmpeg audio issues
```bash
# Install FFmpeg with full codec support
sudo apt-get install ffmpeg  # Linux/WSL
brew install ffmpeg           # macOS

# Verify codecs
ffmpeg -codecs | grep h264
```

---

## Getting Help

If you continue to have installation issues:

1. **Check existing GitHub Issues**: [FlashVSR-Pro Issues](https://github.com/LujiaJin/FlashVSR-Pro/issues)
2. **Create a new issue** with:
   - Your OS and version
   - GPU model and CUDA version
   - Full error messages
   - Steps you've already tried
3. **For Windows users**: Please specify if using Docker, WSL 2, or native Windows

---

## Quick Reference

| Platform | Recommended Method | Difficulty | Support Level |
|----------|-------------------|------------|---------------|
| Linux | Docker or Manual | Easy/Medium | ✅ Primary |
| Windows | Docker Desktop + WSL 2 | Medium | ⚠️ Secondary |
| Windows | WSL 2 Native | Medium | ⚠️ Secondary |
| Windows | Native | Hard | ❌ Unsupported |
| macOS | N/A | N/A | ❌ Incompatible |
| Cloud (Linux) | Docker or Manual | Easy | ✅ Fully Supported |

**Bottom Line**: Use Docker on Linux or Docker Desktop with WSL 2 on Windows for the best experience.
