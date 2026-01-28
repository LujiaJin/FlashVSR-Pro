# utils/__init__.py
"""
FlashVSR utility modules
"""

# Utilities
from .utils import (
    RMS_norm,
    CausalConv3d,
    PixelShuffle3d,
    Buffer_LQ4x_Proj,
    Causal_LQ4x_Proj,
)

from .TCDecoder import build_tcdecoder

# VAE manager
from . import vae_manager

# Audio and tiling utilities will be imported dynamically when used
# Avoid forcing dependencies
