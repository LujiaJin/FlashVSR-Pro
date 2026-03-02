# FlashVSR-Pro: Enhanced Implementation of Real-Time Video Super-Resolution

<sub>💡 **Tip**: If you find this repository's structure or content difficult to understand, visit [deepwiki](https://deepwiki.com/LujiaJin/FlashVSR-Pro) for a comprehensive detailed explanation.</sub>

**FlashVSR-Pro** is an enhanced, production-ready re-implementation of the real-time diffusion-based video super-resolution algorithm introduced in the paper **"FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution"**.

> **Original Paper**: Zhuang, J., Guo, S., Cai, X., Li, X., Liu, Y., Yuan, C., & Xue, T. (2025). *FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution*. arXiv preprint arXiv:2510.12747.  
> **Paper Link**: https://arxiv.org/abs/2510.12747

This project is not the official code release but an independent, refactored implementation focused on improved usability, additional features, and better compatibility for real-world deployment.

---

## ✨ Core Features & Enhancements

### 🚀 **Beyond the Original Implementation**
This project builds upon the core FlashVSR algorithm and introduces several key improvements:

*   **🧩 Unified Inference Script**: A single, parameterized `infer.py` script replaces multiple original scripts (`full`, `tiny`, `tiny-long`), simplifying the user interface.
*   **🎵 Audio Track Preservation**: Automatically detects and preserves the audio track from the input video in the super-resolved output. *(Inspiration for this feature was drawn from the [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) project.)*
*   **💾 Tiled Inference for Reduced VRAM**: Implements a tiling mechanism for the DiT model, significantly lowering GPU memory requirements and enabling processing of higher-resolution videos or operation on GPUs with limited VRAM. *(The concept for tiled DiT inference was inspired by the [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) project.)*
*   **🐳 Optimized Docker Container**: A fully-configured Dockerfile that automatically sets up the complete environment, including Conda environment activation upon container startup.
*   **🔧 Automated Block-Sparse Attention Installation**: Optimizes and automates the installation of the Block-Sparse-Attention backend within the Docker build process. This eliminates the manual compilation complexity encountered in the original implementation, ensuring a seamless setup experience. My specific improvements to Block-Sparse-Attention are documented in this PR: [mit-han-lab/Block-Sparse-Attention#16](https://github.com/mit-han-lab/Block-Sparse-Attention/pull/16#issue-3800081298).
*   **⚡ Enterprise-Grade Performance Optimizations**: Engineered for high-end GPUs (A100/H100/H200, etc), this project features a **Zero-Copy Pipeline**, **Hardware Acceleration** (TF32/cuDNN), **Asynchronous Execution**, and **NVENC Video Encoding**, maximizing throughput and eliminating bottlenecks for production-grade video processing.
*   **� Strict Precision Alignment (Duration & Resolution)**: Implements smart padding strategies to overcome algorithmic block-size constraints (e.g., 8n+1 frames, mod-128 resolution) without data loss.
    *   **Exact Duration**: Ensures output frame count perfectly matches input, solving the "missing seconds" issue and guaranteeing perfect audio sync (e.g., Input 5.0s → Output 5.0s).
    *   **Exact Resolution**: Guarantees output resolution is strictly `Scale × Input` (e.g., 1280x740 → 2560x1480), using reflective padding instead of cropping to prevent pixel loss.
*   **📝 Concise & Professional Logging**: Optimized console output that filters out internal noise and presents key metrics (Real FPS, Resolution Scaling) cleanly, making it suitable for production monitoring.

### ⚡ **Original FlashVSR Advantages (Preserved)**
*   **Real-Time Performance**: Achieves ~17 FPS for 768 × 1408 videos on a single A100 GPU.
*   **One-Step Diffusion**: Efficient streaming framework based on a distilled one-step diffusion model.
*   **State-of-the-Art Quality**: Combines Locality-Constrained Sparse Attention and a Tiny Conditional Decoder for high-fidelity results.
*   **Scalability**: Reliably scales to ultra-high resolutions.

---

## 🎨 Mode Selection

| Mode | VAE Used | Description |
|------|----------------|-------------|
| **full** | `wan2.1` | High Quality, High VRAM. Ideal for quality-critical tasks. |
| **tiny** | `tcd` | Balanced Quality, Lower VRAM. Ideal for efficient real-time processing. |
| **tiny-long** | `tcd` | Balanced Quality, Lower VRAM. Optimized for long videos. |

### Auto-Download

Required VAE models will be anticipated in `./models/FlashVSR-v1.1`.

| VAE | File | Direct Download Link |
| :--- | :--- | :--- |
| **Wan2.1** | `models/FlashVSR-v1.1/Wan2.1_VAE.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/Wan2.1_VAE.pth) |
| **TCDecoder** | `models/FlashVSR-v1.1/TCDecoder.ckpt` | [Download](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/blob/main/TCDecoder.ckpt) |

**Usage Examples:**
```bash
# High quality (full mode)
python infer.py -i inputs/input.mp4 -o results/ --mode full

# Fast inference (tiny mode)
python infer.py -i inputs/input.mp4 -o results/ --mode tiny

# Long video processing (tiny-long mode)
python infer.py -i inputs/long_input.mp4 -o results/ --mode tiny-long
```

## 🖥️ Platform Support

> 📖 **For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

### Supported Platforms
FlashVSR-Pro is **primarily designed for Linux** systems with NVIDIA GPUs. This is due to:

*   **Block-Sparse-Attention Backend**: The core dependency requires CUDA kernel compilation, which is optimized and extensively tested on Linux.
*   **Docker Environment**: The recommended installation uses Docker with Ubuntu base image for consistent environment setup.
*   **CUDA Compilation**: Requires NVIDIA CUDA Toolkit (11.6+) and proper development toolchain, which is more straightforward on Linux.

### Hardware Requirements
*   **GPU**: NVIDIA GPU with CUDA compute capability 8.0+ (Ampere architecture or newer)
  *   Recommended: A100, H100, H200, RTX 3090, RTX 4090
  *   Minimum: RTX 3060, RTX 3070, or similar Ampere/Ada/Hopper GPU
*   **VRAM**: 
  *   Minimum 8GB for low-resolution videos with tiling enabled
  *   16GB+ recommended for high-quality processing
  *   24GB+ for 4K video processing
*   **CUDA**: Version 11.6 or higher (12.x recommended)
*   **System RAM**: 16GB+ recommended

### Windows Users
While FlashVSR-Pro is optimized for Linux, Windows users have the following options:

1. **Docker Desktop (Recommended for Windows)**
   *   Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) with WSL 2 backend
   *   Install [NVIDIA Container Toolkit for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   *   Follow the Docker installation steps below
   *   This provides a Linux environment within Windows with GPU passthrough

2. **Windows Subsystem for Linux (WSL 2)**
   *   Install WSL 2 with Ubuntu 22.04
   *   Install CUDA toolkit in WSL 2
   *   Follow Linux installation instructions within WSL
   *   Note: GPU passthrough must be properly configured

3. **Native Windows Installation (Advanced)**
   *   Requires manual compilation of Block-Sparse-Attention on Windows
   *   Need Visual Studio 2019+ with C++17 support
   *   CUDA Toolkit for Windows
   *   May encounter compatibility issues not tested by maintainers
   *   Not officially supported - use at your own risk

**Important**: If you've been struggling with installation on Windows, we strongly recommend using Docker Desktop with WSL 2, as this provides the most reliable experience.

---

## 🚀 Quick Start with Docker (Recommended)

The easiest way to run FlashVSR-Pro is using the provided Docker container, which includes automated setup for the Block-Sparse-Attention backend. **This is the recommended method for both Linux and Windows (via WSL 2) users.**

### 1. Prerequisites
#### For Linux:
*   Install [Docker](https://docs.docker.com/get-docker/)
*   Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support.
*   Ensure `git-lfs` is installed on your host system to clone model weights.

#### For Windows:
*   Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) with WSL 2 backend enabled
*   Install [NVIDIA CUDA on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
*   Ensure `git-lfs` is installed: `sudo apt-get install git-lfs` (in WSL Ubuntu terminal)
*   Run all subsequent commands in WSL 2 Ubuntu terminal

### 2. Build the Docker Image
```bash
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro
docker build -t flashvsr-pro:latest .
```
**Note**: The Dockerfile automatically handles the compilation and installation of the optimized Block-Sparse-Attention backend, eliminating manual configuration.

### 3. Download Model Weights
Before running the container, download the required model weights.
```bash
# Download the FlashVSR model weights (version 1.1)
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1
```

### 4. Run the Container
The container is configured to automatically activate the `flashvsr` Conda environment upon startup. Make sure that the `models/` directory of the host machine already contains the necessary model weight files, and provide the models when starting the container by mounting.
```bash
# Basic run with interactive shell
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/FlashVSR-Pro \
  flashvsr-pro:latest

# You will be dropped into a shell with the `(flashvsr)` environment active.
# Verify by running: `which python`
```

---

## 🔧 Comprehensive Parameter Reference

### Inference Modes
*   `--mode`: Inference mode selection.
    *   `full`: Uses `WanModel` + `WanVideoVAE`. Highest quality but requires significant VRAM (~14GB+ for 2s 720p without tiling).
    *   `tiny`: Uses `WanModel` + `TCDecoder`. Balanced speed and quality.
    *   `tiny-long`: Optimized streaming inference for long videos.

### Tiling & Memory Management
*   `--tile-dit`: Enable tiling for the DiT model. Drastically reduces VRAM usage by processing the latent space in blocks. Essential for high-resolution inference on consumer GPUs.
    *   `--tile-size`: Tile size in pixels (default: `256`).
    *   `--overlap`: Overlap between tiles (default: `24`).
*   `--tile-vae`: Enable tiling for the VAE decoder. Allows decoding very large frames by splitting the latent representation before decoding.

### Output Control
*   `--fps INTEGER`: Force a specific frame rate for the output video. By default, tries to match input video FPS or uses 30 for image sequences.
*   `--quality INTEGER`: Output video quality (0-10). Default `10` (High). Maps to FFmpeg CRF values.
*   `--color-fix`: Enable AdaIN/Wavelet-based post-processing color correction to match input color tones.
*   `--keep-audio`: Transfer audio track from input to output video.

### Advanced Tuning
*   `--dtype`: Precision format. Options: `bf16` (default, recommended), `fp16`, `fp32`.
*   `--device`: Processing device. Default: `cuda`.
*   `--scale`: Upscaling factor. Default: `2.0`.
*   `--sparse-ratio`: Controls Attention sparsity. Default: `2.0`. Lower (e.g. 1.5) is faster but potentially less stable.
*   `--kv-ratio`: KV cache ratio. Default: `3.0`.
*   `--local-range`: Local attention window size. Default: `11`.
*   `--seed`: Random seed for reproducibility.

### Key Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path to input video or image folder. | **Required** |
| `-o, --output` | Output directory or file path. | `./results` |
| `--mode` | Inference mode: `full` (Wan VAE), `tiny` (TCD), `tiny-long`. | `tiny` |
| `--keep-audio` | Preserve audio from input video (if exists). | `False` |
| `--tile-dit` | Enable memory-efficient tiled DiT inference. | `False` |
| `--tile-vae` | Enable tiled decoding for VAE (Full mode only). | `False` |
| `--tile-size` | Size of each tile when using tiling. | `256` |
| `--overlap` | Overlap between tiles to reduce seams. | `24` |
| `--scale` | Super-resolution scale factor. | `2.0` |
| `--seed` | Random seed for reproducible results. | `0` |

**Note**: The original FlashVSR is primarily designed and tested for 4x super-resolution. While other scales are supported, for optimal quality and stability, using `--scale 4.0` is recommended.

For a full list of arguments, run `python infer.py --help`.

---

## 🎯 Usage

The main interface is the unified `infer.py` script.

### Basic Inference
```bash
# Basic upscaling (Tiny mode - balanced quality/speed)
python infer.py -i ./inputs/example0.mp4 -o ./results/ --mode tiny

# Full mode (Highest quality, requires more VRAM)
python infer.py -i ./inputs/example0.mp4 -o ./results/ --mode full

# Tiny-long mode for long videos
python infer.py -i ./inputs/example4.mp4 -o ./results/ --mode tiny-long
```

### Using Key Enhancements
```bash
# Preserve the audio track from the input video
python infer.py -i inputs/input_with_audio.mp4 -o ./results/ --mode tiny --keep-audio
# Use tiled DiT inference to reduce VRAM usage (enables running on smaller GPUs)
python infer.py -i inputs/large_input.mp4 -o ./results/ --mode tiny --tile-dit --tile-size 256 --overlap 24

# Use tiled VAE for lower memory usage in full mode
python infer.py -i inputs/input.mp4 -o ./results/ --mode full --tile-vae

# Combine multiple enhancements
python infer.py -i inputs/large_input_with_audio.mp4 -o ./results/ --mode full --tile-vae --tile-dit --keep-audio
```

---

## ⚡ Performance Optimizations

FlashVSR-Pro includes comprehensive performance enhancements designed for **high-end GPUs (A100/H100/H200/RTX4090, etc)** and high-throughput production environments.

### 1. Zero-Copy Pipeline
*   **Mechanism**: Implements a fully GPU-resident data pipeline. Input frames are read in batches and immediately transferred to VRAM. Resizing, cropping, and normalization all happen on the GPU.
*   **Benefit**: Eliminates the "CPU -> GPU -> CPU" round-trip bottleneck found in standard implementations. Input tensors stay on the GPU throughout the entire inference lifecycle.

### 2. Hardware Acceleration (Ampere/Hopper)
*   **TF32 Support**: Explicitly enables TensorFloat-32 (TF32) for Matrix Multiplications and Convolutions on Ampere+ architectures.
*   **cuDNN Benchmarking**: Activates `torch.backends.cudnn.benchmark = True`, allowing the driver to auto-tune convolution algorithms for the specific input resolution during the first pass.
*   **Impact**: Significantly boosts raw compute throughput for FP32/BF16 operations without code changes.

### 3. Asynchronous Execution
*   **Mechanism**: Removed unnecessary `torch.cuda.synchronize()` calls from the critical inference path.
*   **Benefit**: Allows the CPU to issue CUDA kernels ahead of execution, fully filling the GPU's instruction pipeline and maximizing utilization rate.

### 4. Efficient I/O
*   **Mechanism**: Bypasses the standard PIL (Python Imaging Library) conversion stack. The pipeline returns raw Numpy arrays directly to the video encoder (FFmpeg).
*   **Benefit**: Saves thousands of CPU cycles per second by avoiding "Tensor -> Numpy -> PIL Image -> Numpy -> FFmpeg" conversion chains.

### 5. Hardware-Accelerated Video Encoding (NVENC)
*   **Mechanism**: Automatically detects and utilizes NVIDIA's NVENC hardware encoder when available. It includes a robust fallback mechanism that performs a functional test of the encoder before usage, ensuring stability even in varied container environments.
*   **Benefit**: Offloads video compression from the CPU to the GPU's dedicated media engine. This drastically reduces CPU load and I/O latency during the final video saving stage, ensuring that high-throughput inference isn't bottlenecked by disk write speeds.

---

## 🔧 Troubleshooting

### 1. Platform-Specific Issues

#### Windows Installation Failures
**Symptom**: Compilation errors, missing CUDA libraries, or general installation failures on Windows.

**Solution**:
- **Recommended**: Use Docker Desktop with WSL 2 backend (see [Platform Support](#-platform-support) section above)
- Ensure you're running commands in WSL 2 Ubuntu terminal, not PowerShell or CMD
- Verify GPU is accessible in WSL: `nvidia-smi` should work in WSL terminal
- If using native Windows: Ensure Visual Studio 2019+ with C++17 support is installed
- Check that CUDA Toolkit matches PyTorch CUDA version

#### Linux Permission Errors
**Symptom**: Docker commands fail with permission errors.

**Solution**:
- Add your user to docker group: `sudo usermod -aG docker $USER`
- Log out and back in for changes to take effect
- Or use `sudo` before docker commands

#### WSL 2 GPU Not Detected
**Symptom**: `nvidia-smi` fails in WSL or Docker can't access GPU.

**Solution**:
- Update Windows to latest version (GPU support requires Windows 11 or Windows 10 21H2+)
- Install latest NVIDIA GPU driver for Windows (not Linux driver)
- Verify in WSL: `nvidia-smi` should show your GPU
- Ensure Docker Desktop has WSL 2 integration enabled
- Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

### 2. CUDA Out of Memory
- Use `--tile-dit` and `--tile-vae` to enable tiled inference.
- Decrease `--tile-size` (e.g., from 256 to 128).
- Use `--dtype fp16` to reduce memory usage.
- Process shorter video segments or lower resolution inputs

### 3. Audio Not Preserved
- Ensure ffmpeg is installed on the host system (or in WSL for Windows users).
- Check whether the input video contains an audio stream: `ffprobe -i input.mp4`
- Verify ffmpeg codec support: `ffmpeg -codecs | grep h264`

### 4. Model Loading Fails
- Make sure model weights are downloaded with git-lfs: `git lfs pull`
- If download is incomplete, files may be text pointers instead of actual weights
- Verify model file integrity (should be several GB in size)
- Check that weight files are in the correct directory: `./models/FlashVSR-v1.1/`
- Ensure git-lfs was installed BEFORE cloning: `git lfs install`

### 5. Block-Sparse-Attention Compilation Errors
- The Docker build process should handle this automatically. If building manually, ensure you have CUDA 12.1+ and the correct PyTorch version installed.
- Reference the fix I released in: [mit-han-lab/Block-Sparse-Attention#16](https://github.com/mit-han-lab/Block-Sparse-Attention/pull/16#issue-3800081298)
- On Windows: Compilation may fail due to missing Visual Studio C++ build tools
- Check available CUDA architectures match your GPU compute capability

### 6. Performance Issues
- Ensure you're using CUDA 12.x for best performance
- Enable TF32: automatically enabled on Ampere+ GPUs
- Check GPU utilization: `nvidia-smi` should show near 100% during processing
- Verify you're not CPU-bottlenecked by checking CPU usage
- Consider using faster storage (SSD) for large video files

---

## ❓ Frequently Asked Questions (FAQ)

### Q: Can I use FlashVSR-Pro on Windows?
**A**: Yes, but with specific requirements. Windows is not the primary development platform, but you can use FlashVSR-Pro on Windows through:
1. **Docker Desktop with WSL 2** (Recommended) - Provides Linux environment with GPU access
2. **WSL 2 directly** - Run Ubuntu in Windows and follow Linux instructions
3. **Native Windows** (Advanced, not officially supported) - Requires manual compilation

See the [Platform Support](#-platform-support) section for detailed instructions.

### Q: Why does installation fail on Windows?
**A**: The Block-Sparse-Attention dependency requires CUDA kernel compilation, which is primarily tested on Linux. The Docker/WSL 2 approach avoids these compilation issues by using a pre-configured Linux environment.

### Q: What GPU do I need?
**A**: You need an NVIDIA GPU with compute capability 8.0 or higher (Ampere generation or newer). Recommended GPUs include:
- Consumer: RTX 3060, RTX 3070, RTX 3090, RTX 4070, RTX 4090
- Professional: A100, H100, H200, A6000, RTX 6000 Ada

### Q: Can I use AMD or Intel GPUs?
**A**: No. FlashVSR-Pro requires NVIDIA GPUs with CUDA support due to the Block-Sparse-Attention kernels being written specifically for NVIDIA CUDA.

### Q: How much VRAM do I need?
**A**: 
- Minimum: 8GB with tiling enabled (`--tile-dit --tile-vae`)
- Recommended: 16GB for 1080p/1440p video processing
- Optimal: 24GB+ for 4K videos without tiling

### Q: Will this work on macOS?
**A**: No. FlashVSR-Pro requires NVIDIA CUDA, which is not available on macOS. Apple Silicon (M1/M2/M3) uses different GPU architecture and is not compatible.

### Q: I've been trying to install for weeks, what should I do?
**A**: We recommend the Docker approach as it handles all dependencies automatically:
1. Install Docker Desktop (with WSL 2 if on Windows)
2. Follow the [Quick Start with Docker](#-quick-start-with-docker-recommended) guide
3. This eliminates most installation issues related to CUDA, compilation, and dependencies

### Q: Can I use this on Google Colab or cloud services?
**A**: Yes! Cloud platforms with NVIDIA GPUs (Google Colab with T4/A100, AWS with GPU instances, etc.) work well. Use the Docker method or follow Linux installation instructions.

---

## 🛠️ Project Structure
```
FlashVSR-Pro/
├── .gitmodules                         # Git submodule configuration
├── Block-Sparse-Attention/             # Git submodule: Sparse attention backend (with automated build)
├── models/                             # Model weights directory
│   ├── FlashVSR-v1.1/                  # Model weights V1.1
│   ├── prompt_tensor/                  # Pre-computed text prompt embeddings
├── diffsynth/                          # Core library (ModelManager, Pipelines)
├── inputs/                             # Default directory for input videos/images
├── results/                            # Default directory for output videos
├── utils/                              # Enhanced utilities module
│   ├── __init__.py
│   ├── utils.py                        # Core utilities (Causal_LQ4x_Proj, etc.)
│   ├── TCDecoder.py                    # Tiny Conditional Decoder for 'tiny' mode
│   ├── audio_utils.py                  # Audio preservation functions
│   ├── tile_utils.py                   # Tiled inference for low VRAM
│   └── vae_manager.py                  # VAE Manager for multiple VAE support
├── infer.py                            # Main unified inference script
├── Dockerfile                          # Container definition with auto-activation
├── entrypoint.sh                       # Container entry script
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup for the `diffsynth` module
├── LICENSE                             # Project license file
└── README.md                           # This file
```

---

## 📄 License & Citation

### License
This project is released under the same license (Apache Software license) as the original FlashVSR implementation. Please see the `LICENSE` file in the original repository for details.

### Citation
If you use the FlashVSR algorithm in your research, please cite the original FlashVSR paper:
```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```

**If you use this implementation (FlashVSR-Pro) in your work, please cite:**
- This repository: https://github.com/LujiaJin/FlashVSR-Pro
- The optimized Block-Sparse-Attention backend: https://github.com/LujiaJin/Block-Sparse-Attention

### Acknowledgements
*   The core algorithm is from the original **FlashVSR** authors.
*   Inspiration for the **audio preservation** and **tiled DiT inference** features came from the community project [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus).
*   The automated build and optimization of the **Block-Sparse-Attention** backend is a contribution of this project, with improvements documented in [mit-han-lab/Block-Sparse-Attention#16](https://github.com/mit-han-lab/Block-Sparse-Attention/pull/16#issue-3800081298).

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/LujiaJin/FlashVSR-Pro/issues) if you want to contribute.

---

**Happy Super-Resolution!** 🚀