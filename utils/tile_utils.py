"""
Tile utilities for FlashVSR
Provides tiled inference for DiT and VAE to reduce VRAM usage
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Generator
import warnings
from tqdm import tqdm

def calculate_tile_coords(height: int, width: int, tile_size: int, overlap: int, multiple: int = 32) -> List[Tuple[int, int, int, int]]:
    """
    Calculate tile coordinates (x1, y1, x2, y2)
    """
    if overlap >= tile_size // 2:
        raise ValueError(f"Overlap ({overlap}) must be less than half of tile_size ({tile_size})")
    
    if tile_size < multiple:
        tile_size = ((tile_size + multiple - 1) // multiple) * multiple
        warnings.warn(f"Tile size increased to {tile_size} to be multiple of {multiple}")
    
    coords = []
    stride = tile_size - overlap
    
    # Calculate number of rows and columns
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            # Initial end coordinates
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            # Ensure consistent tile size (adjust end coordinates)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            
            # Adjust to multiple
            tile_h = y2 - y1
            tile_w = x2 - x1
            
            if tile_h % multiple != 0:
                y2 = y1 + (tile_h // multiple) * multiple
                if y2 <= y1:
                    y2 = y1 + multiple
            
            if tile_w % multiple != 0:
                x2 = x1 + (tile_w // multiple) * multiple
                if x2 <= x1:
                    x2 = x1 + multiple
            
            # Ensure not exceeding boundaries
            y2 = min(y2, height)
            x2 = min(x2, width)
            
            # Final adjustment, ensure at least one pixel
            if y2 <= y1 or x2 <= x1:
                continue
                
            coords.append((x1, y1, x2, y2))
    
    # If no tiles generated, return entire image as one tile
    if not coords:
        coords = [(0, 0, width, height)]
    
    return coords

def create_feather_mask(tile_h: int, tile_w: int, overlap: int, device='cpu') -> torch.Tensor:
    """
    Create feather mask (tensor version)
    """
    mask = torch.ones(1, 1, tile_h, tile_w, device=device, dtype=torch.float32)
    
    if overlap > 0:
        # Create gradient
        ramp = torch.linspace(0, 1, overlap, device=device)
        
        # Left edge
        mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
        # Right edge
        mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
        # Top edge
        mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
        # Bottom edge
        mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def split_video_into_tiles(
    video: torch.Tensor, 
    tile_size: int, 
    overlap: int,
    multiple: int = 32
) -> Generator[Tuple[torch.Tensor, Tuple[int, int, int, int]], None, None]:
    """
    Split video tensor into spatial tiles
    """
    # Pipeline returns 4D tensor (C, T, H, W), but we need 5D for splitting
    if video.dim() == 4:
        # Add batch dimension
        video = video.unsqueeze(0)
    
    if video.dim() != 5:
        raise ValueError(f"Expected video tensor with 4 or 5 dimensions (B, C, T, H, W) or (C, T, H, W), got {video.dim()}")
    
    B, C, T, H, W = video.shape
    
    # Calculate tile coordinates
    coords = calculate_tile_coords(H, W, tile_size, overlap, multiple)
    
    for x1, y1, x2, y2 in coords:
        # Extract tile
        tile = video[:, :, :, y1:y2, x1:x2]
        yield tile, (x1, y1, x2, y2)

def stitch_video_tiles_back(
    tiles: List[torch.Tensor],
    coords: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, int],  # (H, W)
    overlap: int,
    scale: int = 1
) -> torch.Tensor:
    """
    Stitch video tiles back to original image with feather blending
    """
    if not tiles or not coords:
        raise ValueError("No tiles or coordinates provided")
    
    if len(tiles) != len(coords):
        raise ValueError(f"Number of tiles ({len(tiles)}) doesn't match number of coordinates ({len(coords)})")
    
    # Pipeline returns 4D tensors (C, T, H, W)
    # Convert all tiles to 4D if they are 5D (should not happen, but just in case)
    for i, tile in enumerate(tiles):
        if tile.dim() == 5:
            tiles[i] = tile.squeeze(0)  # Remove batch dimension
        elif tile.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C, T, H, W), got {tile.dim()}D")
    
    # Get shape information from first tile
    C, T, tile_h, tile_w = tiles[0].shape
    H, W = original_shape
    H_scaled, W_scaled = H * scale, W * scale
    
    # Create canvas (4D: C, T, H_scaled, W_scaled)
    canvas = torch.zeros((C, T, H_scaled, W_scaled), dtype=tiles[0].dtype, device=tiles[0].device)
    weight_canvas = torch.zeros((1, T, H_scaled, W_scaled), dtype=tiles[0].dtype, device=tiles[0].device)
    
    for tile, (x1, y1, x2, y2) in zip(tiles, coords):
        # Check tile shape
        if tile.shape != (C, T, tile_h, tile_w):
            warnings.warn(f"Tile shape {tile.shape} doesn't match expected ({C}, {T}, {tile_h}, {tile_w}), resizing")
            # Resize spatial dimensions only
            tile = F.interpolate(tile, size=(tile_h, tile_w), mode='bilinear', align_corners=False)
        
        # Create feather mask for spatial dimensions
        mask = create_feather_mask(tile_h, tile_w, overlap * scale, device=tile.device)
        # Expand mask to match tile dimensions: (1, 1, tile_h, tile_w) -> (1, T, tile_h, tile_w)
        mask_expanded = mask.expand(1, T, tile_h, tile_w)
        
        # Calculate scaled coordinates
        out_x1, out_y1 = x1 * scale, y1 * scale
        out_x2, out_y2 = out_x1 + tile_w, out_y1 + tile_h
        
        # Ensure not exceeding canvas boundaries
        out_y2 = min(out_y2, H_scaled)
        out_x2 = min(out_x2, W_scaled)
        
        # Adjust tile size to fit canvas (if needed)
        if out_y2 - out_y1 != tile_h or out_x2 - out_x1 != tile_w:
            # Resize spatial dimensions
            tile = F.interpolate(
                tile, 
                size=(out_y2 - out_y1, out_x2 - out_x1), 
                mode='bilinear', 
                align_corners=False
            )
            mask_expanded = F.interpolate(
                mask_expanded, 
                size=(out_y2 - out_y1, out_x2 - out_x1), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Accumulate to canvas
        canvas[:, :, out_y1:out_y2, out_x1:out_x2] += tile * mask_expanded
        weight_canvas[:, :, out_y1:out_y2, out_x1:out_x2] += mask_expanded
    
    # Normalize (avoid division by zero)
    weight_canvas[weight_canvas == 0] = 1.0
    result = canvas / weight_canvas
    
    return result

def apply_tiled_inference_simple(
    pipeline,  # FlashVSR pipeline
    LQ_video: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 24,
    tile_size_vae: tuple = None, # Explicitly accept VAE parameter to avoid kwargs collision
    **pipeline_kwargs
) -> torch.Tensor:
    """
    Simplified tiled inference function for FlashVSR
    """
    if tile_size <= 0:
        # No tiling
        if tile_size_vae is not None: pipeline_kwargs['tile_size'] = tile_size_vae
        return pipeline(**pipeline_kwargs)
    
    # Check input shape
    if LQ_video.dim() not in [4, 5]:
        raise ValueError(f"Expected LQ_video with 4 or 5 dimensions (C, T, H, W) or (B, C, T, H, W), got {LQ_video.dim()}")
    
    # Ensure LQ_video is 5D for consistency (add batch dimension if 4D)
    if LQ_video.dim() == 4:
        LQ_video = LQ_video.unsqueeze(0)
    
    B, C, T, H, W = LQ_video.shape
    
    # Calculate tile coordinates
    coords = calculate_tile_coords(H, W, tile_size, overlap, multiple=32)
    
    print(f"Tiled Inference: {H}x{W} -> {len(coords)} tiles")
    
    # Store all tile results
    output_tiles = []
    
    for idx, (x1, y1, x2, y2) in enumerate(coords):
        # Print tile inference progress
        is_streaming = "TinyLong" in type(pipeline).__name__
        msg = f"Infering tile {idx+1}/{len(coords)}"
        if is_streaming:
            msg += " (Streaming"
            if tile_size_vae is not None:
                msg += " & VAE-tiled"
            msg += ")..."
            # For streaming mode, we rely on the pipeline's internal progress bar,
            # which we want to be clean (without 'DiT Inference' prefix if user requested)
            # But here we just set the header message.
        else:
            msg += "..."
            
        print(msg)

        # Extract tile
        tile = LQ_video[:, :, :, y1:y2, x1:x2]
        
        # Update pipeline parameters
        tile_kwargs = pipeline_kwargs.copy()
        if tile_size_vae is not None:
            tile_kwargs['tile_size'] = tile_size_vae
        tile_kwargs['LQ_video'] = tile
        tile_kwargs['height'] = y2 - y1
        tile_kwargs['width'] = x2 - x1
        
        # Set decoding message, indicating if VAE tiling is also active
        if tile_size_vae is not None:
             tile_kwargs['decoding_msg'] = "Decoding video (Tiled)"
        else:
             tile_kwargs['decoding_msg'] = "Decoding video"
        
        # Run inference (quiet mode for performance)
        tile_output = pipeline(**tile_kwargs)
        output_tiles.append(tile_output)
    
    # Stitch all tiles
    print(f"Stitching {len(output_tiles)} tiles...")
    final_output = stitch_video_tiles_back(
        output_tiles, coords, (H, W), overlap, scale=1
    )
    
    return final_output

def vae_decode_tiled(
    vae_model, 
    latents: torch.Tensor, 
    tile_size: int = 512, 
    overlap: int = 32, 
    desc: str = "Decoding video",
    decode_fn = None,
    **kwargs
):
    """
    Tiled processing for VAE decoding
    
    Args:
        vae_model: VAE model with .decode() method (used if decode_fn is None)
        latents: Input latent tensor
        tile_size: Tile size for spatial dimension
        overlap: Overlap size
        desc: Description for progress bar
        decode_fn: Optional callable for decoding. If provided, used instead of vae_model.decode.
                  Crucial for avoiding recursion when monkey-patching.
        **kwargs: Additional arguments to pass to the decode function (e.g. device)
    """
    if latents.dim() == 4:
        # Image case: add time dimension
        latents = latents.unsqueeze(2)  # (B, C, 1, H, W)
        is_image = True
    elif latents.dim() == 5:
        is_image = False
    else:
        raise ValueError(f"Expected latents with 4 or 5 dimensions, got {latents.dim()}")
    
    # Determine decoder function
    decoder = decode_fn if decode_fn is not None else vae_model.decode
    
    B, C, T, H, W = latents.shape
    
    # If dimensions smaller than tile_size, decode directly
    if H <= tile_size and W <= tile_size:
        result = decoder(latents, **kwargs)
        return result.squeeze(2) if is_image else result
    
    # Tiled decoding
    coords = calculate_tile_coords(H, W, tile_size, overlap, multiple=32)
    
    # Store tile results
    output_tiles = []
    
    # Update description to include (Tiled)
    if not desc.endswith("(Tiled)"):
        desc += " (Tiled)"
    
    # Always use tqdm so we can show the description (e.g. "Decoding video (Tiled): 100%|...| 16/16")
    pbar = tqdm(coords, desc=desc)
        
    for idx, (x1, y1, x2, y2) in enumerate(pbar):
        # Extract tile
        tile = latents[:, :, :, y1:y2, x1:x2]
        
        # Decode using the selected decoder function
        # Silence inner tqdm if kwargs has 'decoding_msg', remove it for inner calls
        inner_kwargs = kwargs.copy()
        if 'decoding_msg' in inner_kwargs:
            inner_kwargs.pop('decoding_msg')
            
        tile_decoded = decoder(tile, **inner_kwargs)
        output_tiles.append(tile_decoded)
    
    # Calculate scale factor from the first tile
    # tile input H: (y2-y1), decoded H: output_tiles[0].shape[-2]
    # Assuming output is (B, C, T, H, W) or (C, T, H, W)
    if output_tiles:
        first_tile_in_h = coords[0][3] - coords[0][1]
        first_tile_out_h = output_tiles[0].shape[-2]
        # Round scale to nearest integer to avoid float precision issues with very small tiles
        scale = int(round(first_tile_out_h / first_tile_in_h))
    else:
        scale = 1

    # Stitch
    final_output = stitch_video_tiles_back(
        output_tiles, coords, (H, W), overlap, scale=scale
    )
    
    # Check if final_output lost its channel dimension or is squeezed inappropriately
    # stitch_video_tiles_back returns (C, T, H, W).
    # Input was (B, C, T, H, W) or (C, T, H, W) if B=1 is handled inside.
    
    # If original input had batch dimension, we might need to restore it?
    # But wait, vae_decode_tiled is expected to return (C, T, H, W) or (B, C, T, H, W)?
    # FlashVSR Pipeline expects (1, 3, F, H, W) usually.
    # stitch_video_tiles_back returns (C, T, H_scaled, W_scaled).
    
    # The error says "Received 3-dim tensor: [85, 768, 768]". This looks like (T, H, W).
    # Missing C dimension!
    
    # Let's check why C dimension is lost.
    # stitch_video_tiles_back: "canvas = torch.zeros((C, T, H_scaled, W_scaled)..."
    # It returns canvas / weight_canvas.
    
    # However, `WanVideoVAE.decode` returns (C, T, H, W)? Or (3, T, H, W)?
    # Wait, `WanVideoVAE.single_decode` -> `model.decode` -> `video.clamp_(-1, 1)`
    # Let's look at `WanVideoVAE.decode` again.
    # It stacks videos.
    
    if is_image:
         return final_output.squeeze(2)
    else:
         # Restoring batch dimension if input had it
         if B == 1:
             return final_output.unsqueeze(0)
         return final_output
