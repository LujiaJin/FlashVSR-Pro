# utils/vae/vae_system.py
"""
VAE System for FlashVSR-Pro
Unified VAE management and loading system with quality/VRAM trade-offs.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, Any
import warnings
import os

from utils.TCDecoder import build_tcdecoder, TAEW2_1DiffusersWrapper, TAEHV
from utils.tile_utils import vae_decode_tiled


class VAESystem:
    """
    Unified VAE system supporting multiple decoder variants.
    Provides consistent interface with varying quality/VRAM trade-offs.
    """

    VAE_CONFIGS = {
        "wan2.1": {
            "class": "WanVAE",
            "default_path": "models/FlashVSR-v1.1/Wan2.1_VAE.pth",
            "is_tcdecoder": False,
            "description": "High Quality, High VRAM. Ideal for quality-critical tasks"
        },
        "tcd": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": "models/FlashVSR-v1.1/TCDecoder.ckpt",
            "channels": [512, 256, 128, 128],
            "is_tcdecoder": True,
            "description": "Balanced Quality, Lower VRAM. Ideal for efficient real-time processing"
        }
    }

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Initialize VAE System."""
        self.device = device
        self.dtype = dtype
        self.current_vae = None
        self.current_type = None

    def load_vae(self,
                 vae_type: str = "wan2.1",
                 weight_path: Optional[str] = None,
                 mode: str = "full",
                 tile_vae: bool = False,
                 tile_size: int = 512,
                 overlap: int = 32,
                 model_dir: str = "./models/FlashVSR") -> nn.Module:
        """Load specified VAE model."""
        vae_type = vae_type.lower()
        if vae_type not in self.VAE_CONFIGS:
             # Fallback logic if needed or raise error
             if mode == 'full':
                 vae_type = 'wan2.1'
             else:
                 vae_type = 'tcd'
             
             if vae_type not in self.VAE_CONFIGS: # Should not happen
                raise ValueError(
                    f"Unsupported VAE type: '{vae_type}'. "
                    f"Available: {list(self.VAE_CONFIGS.keys())}"
                )

        config = self.VAE_CONFIGS[vae_type]
        self.current_type = vae_type

        # Determine weight path
        if weight_path is None:
             if config["default_path"]:
                weight_path = config["default_path"]
             else:
                weight_path = None # tcd has no weight path

        if weight_path and not os.path.exists(weight_path):
            warnings.warn(
                f"VAE weights not found: {weight_path}. "
                f"Model '{vae_type}' may not load correctly."
            )

        # print(f"Loading VAE: {vae_type}")

        # Load model based on type
        if config["is_tcdecoder"]:
            vae_model = self._load_tcdecoder_vae(vae_type, config, weight_path, mode)
        else:
            vae_model = self._load_wan_vae(vae_type, config, weight_path, mode)

        self.current_vae = vae_model

        # Enable tiling if requested
        if tile_vae and hasattr(vae_model, 'decode'):
            self._wrap_decode_for_tiling(tile_size, overlap)

        # print(f"VAE ready: {config['description']}")

        return vae_model

    def _load_tcdecoder_vae(self,
                           vae_type: str,
                           config: Dict,
                           weight_path: str,
                           mode: str) -> nn.Module:
        """Load TCDecoder type VAE using specialized loaders."""
        channels = config.get("channels", [512, 256, 128, 128])

        # Use specialized loader
        vae_model = VAELoaderFactory.load_vae(vae_type, weight_path, channels=channels)

        # Move to device and dtype
        vae_model = vae_model.to(device=self.device, dtype=self.dtype)

        return vae_model

    def _load_wan_vae(self,
                    vae_type: str,
                    config: Dict,
                    weight_path: str,
                    mode: str) -> nn.Module:
        """Load Wan VAE type for full mode using specialized loader."""
        # Use specialized loader
        vae_model = VAELoaderFactory.load_vae(vae_type, weight_path)

        # Move to device and dtype in single call, then remove encoder
        vae_model = vae_model.to(device=self.device, dtype=self.dtype)

        # Remove encoder to save memory (if present)
        if hasattr(vae_model, 'encoder'):
            vae_model.encoder = None
        if hasattr(vae_model, 'conv1'):
            vae_model.conv1 = None

        return vae_model

    def _wrap_decode_for_tiling(self, tile_size: int, overlap: int):
        """Enable tiled VAE decoding."""
        if not self.current_vae or not hasattr(self.current_vae, 'decode'):
            return

        original_decode = self.current_vae.decode

        def tiled_decode(latents, **kwargs):
            # Extract desc/decoding_msg if present to pass as 'desc' to vae_decode_tiled
            # The pipeline passes 'decoding_msg' in kwargs
            desc = kwargs.get("decoding_msg", "VAE Decoding")
            
            # Prioritize runtime parameters from pipeline (Latent Space) if present
            # Pipeline passes 'tile_size' (tuple) and 'tile_stride' (tuple)
            rt_tile_size = kwargs.pop('tile_size', None)
            rt_tile_stride = kwargs.pop('tile_stride', None)
            rt_overlap = kwargs.pop('overlap', None)
            
            # Determine effective tile_size (int calling for tile_utils)
            if rt_tile_size is not None:
                final_tile_size = rt_tile_size[0] if isinstance(rt_tile_size, (tuple, list)) else rt_tile_size
            else:
                final_tile_size = tile_size # Fallback to closure (Configured Pixel Size?)

            # Determine effective overlap
            if rt_tile_stride is not None:
                stride_val = rt_tile_stride[0] if isinstance(rt_tile_stride, (tuple, list)) else rt_tile_stride
                final_overlap = final_tile_size - stride_val
            elif rt_overlap is not None:
                final_overlap = rt_overlap
            else:
                final_overlap = overlap

            # Disable nested tiling for the inner decoder calls
            kwargs['tiled'] = False
            
            return vae_decode_tiled(
                self.current_vae,
                latents,
                tile_size=final_tile_size,
                overlap=final_overlap,
                desc=desc,
                decode_fn=original_decode, # Pass original decode to avoid recursion
                **kwargs # Pass remaining arguments (like device) to decoder
            )

        self.current_vae.decode = tiled_decode
        print(f"VAE tiled decoding enabled (tile_size={tile_size}, overlap={overlap})")

    def get_current_vae_info(self) -> Dict:
        """Get information about currently loaded VAE."""
        if not self.current_type:
            return {}

        config = self.VAE_CONFIGS.get(self.current_type, {})
        return {
            'type': self.current_type,
            'description': config.get('description', ''),
            'is_tcdecoder': config.get('is_tcdecoder', False),
            'weight_path': config.get('default_path', '')
        }

    def clean_memory(self):
        """Clean up memory and reset state."""
        if self.current_vae:
            if hasattr(self.current_vae, 'clean_mem'):
                self.current_vae.clean_mem()
            elif hasattr(self.current_vae, 'tcd_model') and hasattr(self.current_vae.tcd_model, 'clean_mem'):
                self.current_vae.tcd_model.clean_mem()

        self.current_vae = None
        self.current_type = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------
# Specialized VAE Loaders
# -------------------------

class WanVAELoader:
    """Specialized loader for Wan VAE models (wan2.1)"""

    @staticmethod
    def load_vae(weight_path: str, vae_type: str) -> nn.Module:
        """Load Wan VAE with proper architecture detection"""
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError(f"VAE weights not found: {weight_path}")

        # Load checkpoint
        if weight_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(weight_path, device='cpu')
        else:
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                raise ValueError("Cannot extract state_dict from checkpoint")

        # Detect architecture from state_dict keys
        keys = list(state_dict.keys())

        # Wan VAEs have specific key patterns
        has_encoder = any(k.startswith('encoder') for k in keys)
        has_decoder = any(k.startswith('decoder') for k in keys)

        if not has_decoder:
            raise ValueError(f"No decoder found in {vae_type} weights")

        # Create WanVideoVAE instance
        from diffsynth.models.wan_video_vae import WanVideoVAE

        # Standard Wan VAE
        vae_model = WanVideoVAE(z_dim=16, dim=96)

        # Add 'model.' prefix to match WanVideoVAE state_dict keys
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        # Load state dict
        missing, unexpected = vae_model.load_state_dict(state_dict, strict=False)

        if missing:
            warnings.warn(f"Missing keys in {vae_type}: {missing[:3]}...")
        if unexpected:
            warnings.warn(f"Unexpected keys in {vae_type}: {unexpected[:3]}...")

        return vae_model


class TCDVAELoader:
    """Specialized loader for TCD VAE models"""

    @staticmethod
    def load_vae(weight_path: str, vae_type: str, channels: list = None) -> nn.Module:
        """Load TCD VAE with proper TCDecoder structure"""
        if channels is None:
            channels = [512, 256, 128, 128]  # Default for tcd

        # Build TCDecoder with appropriate channels
        latent_channels = 16 + 768  # TCD specific
        tcdecoder = build_tcdecoder(
            new_channels=channels,
            new_latent_channels=latent_channels
        )

        # Load weights if available
        if weight_path and os.path.exists(weight_path):
            try:
                if weight_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_path, device='cpu')
                else:
                    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif isinstance(checkpoint, dict):
                        state_dict = checkpoint
                    else:
                        raise ValueError("Cannot extract state_dict from checkpoint")

                missing, unexpected = tcdecoder.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    warnings.warn(f"TCD weight loading: {len(missing)} missing, {len(unexpected)} unexpected keys")

            except Exception as e:
                warnings.warn(f"Failed to load weights for {vae_type}: {e}")

        # Wrap for pipeline compatibility
        class TCDecoderWrapper(nn.Module):
            def __init__(self, tcd_model, device, dtype):
                super().__init__()
                self.tcd_model = tcd_model
                self.device = device
                self.dtype = dtype
                self.config = type('Config', (), {
                    'scaling_factor': 1.0,
                    'latents_mean': torch.zeros(16),
                    'latents_std': torch.ones(16)
                })()

            def decode(self, latents):
                n, c, t, h, w = latents.shape
                latents_ntc = latents.transpose(1, 2)

                decoded = self.tcd_model.decode_video(
                    latents_ntc, parallel=False, show_progress_bar=False
                )

                decoded = decoded.transpose(1, 2)
                decoded = decoded * 2.0 - 1.0

                return decoded

            def decode_video(self, *args, **kwargs):
                return self.tcd_model.decode_video(*args, **kwargs)

            def clear_cache(self):
                if hasattr(self.tcd_model, 'clean_mem'):
                    self.tcd_model.clean_mem()

            def clean_mem(self):
                if hasattr(self.tcd_model, 'clean_mem'):
                    self.tcd_model.clean_mem()

            def to(self, device=None, dtype=None):
                if device:
                    self.device = device
                    self.tcd_model = self.tcd_model.to(device)
                if dtype:
                    self.dtype = dtype
                    self.tcd_model = self.tcd_model.to(dtype)
                return self

        return TCDecoderWrapper(tcdecoder, "cuda", torch.bfloat16)


class VAELoaderFactory:
    """Factory for creating appropriate VAE loaders"""

    LOADERS = {
        # Wan series
        "wan2.1": WanVAELoader,

        # TCD series
        "tcd": TCDVAELoader,
    }

    @staticmethod
    def get_loader(vae_type: str):
        """Get the appropriate loader for a VAE type"""
        if vae_type not in VAELoaderFactory.LOADERS:
            raise ValueError(f"No loader available for VAE type: {vae_type}")

        return VAELoaderFactory.LOADERS[vae_type]

    @staticmethod
    def load_vae(vae_type: str, weight_path: str, **kwargs) -> nn.Module:
        """Load VAE using the appropriate specialized loader"""
        loader_class = VAELoaderFactory.get_loader(vae_type)
        return loader_class.load_vae(weight_path, vae_type, **kwargs)


# Backward compatibility alias
VAEManager = VAESystem
