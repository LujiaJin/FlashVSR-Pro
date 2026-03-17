from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline


# -----------------------------
# Basic utilities: Statistics for ADAIN (Reserved if needed; pipeline defaults to wavelet)
# -----------------------------
def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat.dim() == 4, 'feat must be (N, C, H, W)'
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    assert content_feat.shape[:2] == style_feat.shape[:2], "ADAIN: N, C must match"
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


# -----------------------------
# Wavelet-based blur and decomposition/reconstruction (Used by ColorCorrector)
# -----------------------------
def _make_gaussian3x3_kernel(dtype, device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125 ],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    assert x.dim() == 4, 'x must be (N, C, H, W)'
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    out = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C)
    return out


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 4, 'x must be (N, C, H, W)'
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2 ** i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(content: torch.Tensor, style: torch.Tensor, levels: int = 5) -> torch.Tensor:
    c_high, _ = _wavelet_decompose(content, levels=levels)
    _, s_low = _wavelet_decompose(style, levels=levels)
    return c_high + s_low


# -----------------------------
# Stateless Color Correction Module (Video-friendly, defaults to wavelet)
# -----------------------------
class TorchColorCorrectorWavelet(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        assert x.dim() == 5, 'Input must be (B, C, f, H, W)'
        B, C, f, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * f, C, H, W)
        return y, B, f

    @staticmethod
    def _unflatten_time(y: torch.Tensor, B: int, f: int) -> torch.Tensor:
        BF, C, H, W = y.shape
        assert BF == B * f
        return y.reshape(B, f, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,  # (B, C, f, H, W)
        lq_image: torch.Tensor,  # (B, C, f, H, W)
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal['wavelet', 'adain'] = 'wavelet',
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # Check basic dimensions
        assert hq_image.dim() == 5 and hq_image.shape[1] == 3, "Input must be (B, 3, f, H, W)"
        
        # Auto-resize lq_image if spatial dimensions mismatch (Robustness Fix)
        if hq_image.shape[-2:] != lq_image.shape[-2:]:
            lq_flat, B_lq, f_lq = self._flatten_time(lq_image)
            lq_flat = F.interpolate(
                lq_flat, 
                size=(hq_image.shape[-2], hq_image.shape[-1]), 
                mode='bilinear', 
                align_corners=False
            )
            lq_image = self._unflatten_time(lq_flat, B_lq, f_lq)

        # Ensure shapes match now
        if hq_image.shape != lq_image.shape:
             raise ValueError(f"ColorCorrector Shape Mismatch: HQ {hq_image.shape} vs LQ {lq_image.shape}")

        B, C, f, H, W = hq_image.shape
        if chunk_size is None or chunk_size >= f:
            hq4, B, f = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)
            if method == 'wavelet':
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == 'adain':
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"Unknown method: {method}")
            out4 = torch.clamp(out4, *clip_range)
            out = self._unflatten_time(out4, B, f)
            return out

        outs = []
        for start in range(0, f, chunk_size):
            end = min(start + chunk_size, f)
            hq_chunk = hq_image[:, :, start:end]
            lq_chunk = lq_image[:, :, start:end]
            hq4, B_, f_ = self._flatten_time(hq_chunk)
            lq4, _, _ = self._flatten_time(lq_chunk)
            if method == 'wavelet':
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == 'adain':
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"Unknown method: {method}")
            out4 = torch.clamp(out4, *clip_range)
            out_chunk = self._unflatten_time(out4, B_, f_)
            outs.append(out_chunk)
        out = torch.cat(outs, dim=2)
        return out


# -----------------------------
# Simplified Pipeline (DiT + VAE only)
# -----------------------------
class FlashVSRTinyPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        # Only manage dit / vae
        dtype = next(iter(self.dit.parameters())).dtype
        from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRTinyPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        # Optional: Unified Sequence Parallelism (Default off here)
        pipe.use_unified_sequence_parallel = False
        return pipe

    def denoising_model(self):
        return self.dit

    # -------------------------
    # Added: Explicit KV Pre-initialization Function
    # -------------------------
    def init_cross_kv(
        self,
        context_tensor: Optional[torch.Tensor] = None,
    ):
        self.load_models_to_device(["dit"])
        """
        Generate text context using fixed prompt and initialize all CrossAttention KV caches in WanModel.
        Must be explicitly called once before __call__.
        """
        prompt_path = "models/prompt_tensor/posi_prompt.pth"

        if self.dit is None:
            raise RuntimeError("Please initialize self.dit first via fetch_models / from_model_manager")

        if context_tensor is None:
            if prompt_path is None:
                raise ValueError("init_cross_kv: Either prompt_path or context_tensor must be provided")
            ctx = torch.load(prompt_path, map_location=self.device)
        else:
            ctx = context_tensor

        ctx = ctx.to(dtype=self.torch_dtype, device=self.device)

        if self.prompt_emb_posi is None:
            self.prompt_emb_posi = {}
        self.prompt_emb_posi['context'] = ctx

        if hasattr(self.dit, "reinit_cross_kv"):
            self.dit.reinit_cross_kv(ctx)
        else:
            raise AttributeError("WanModel is missing reinit_cross_kv(ctx) method, please add this capability in the model implementation.")
        self.timestep = torch.tensor([1000.], device=self.device, dtype=self.torch_dtype)
        self.t = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))
        # Scheduler
        self.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
        self.load_models_to_device([])

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), decoding_msg=None):
        if not decoding_msg:
             decoding_msg = "Decoding video"
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, decoding_msg=decoding_msg)
        return frames

    def _build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def _build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self._build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self._build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = h.view(H, 1).expand(H, W)
        w = w.view(1, W).expand(H, W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = mask.view(1, 1, 1, H, W)
        return mask

    def _tiled_decode(self, latents, cond, tile_size, tile_stride, decoding_msg):
        # latents: (B, C, F, H, W)
        # cond: (B, C, F, H_c, W_c)
        
        device = self.device
        dtype = latents.dtype
        B, C, F, H, W = latents.shape
        _, _, _, H_c, W_c = cond.shape
        
        # Determine upscale factor for VAE (Latent -> Output)
        vae_upscale = 8 # WanVideoVAE default
        out_H, out_W = H * vae_upscale, W * vae_upscale
        
        # Determine scale factor for Cond (Latent -> Cond)
        # Assuming cond is 8x latent size (standard for WanVideoVAE 8x downsample)
        # PixelShuffle3d(4, 8, 8) reduces Cond by 8x spatially to match Latent
        cond_scale = 8
        
        # Prepare Output Accumulator
        # Output is [-1, 1] range from TCDecoder, we accumulate directly?
        # We need float32 accumulator for precision
        # TCDecoder temporal upsampling: F_latent -> F_latent * 4 - 3 (typically)
        # We use dimensions from cond (LQ_video) if available, as they should match target output
        out_F = cond.shape[2] 
        # Fallback if cond not consistent: out_F = F * 4 - 3
        
        value = torch.zeros((B, 3, out_F, out_H, out_W), device="cpu", dtype=torch.float32)
        count = torch.zeros((B, 1, out_F, out_H, out_W), device="cpu", dtype=torch.float32)
        
        # Define Tiles
        # tile_size and tile_stride are in Latent Space
        ts_h, ts_w = tile_size
        st_h, st_w = tile_stride
        
        tasks = []
        for h in range(0, H, st_h):
            if (h-st_h >= 0 and h-st_h+ts_h >= H): continue
            for w in range(0, W, st_w):
                if (w-st_w >= 0 and w-st_w+ts_w >= W): continue
                tasks.append((h, w))
        
        if not decoding_msg:
             decoding_msg = "Decoding video (Tiled)"
             
        for (h, w) in tqdm(tasks, desc=decoding_msg):
            self.TCDecoder.clean_mem()
            
            # Tile coordinates in Latent Space
            h_end = min(h + ts_h, H)
            w_end = min(w + ts_w, W)
            
            # 1. Crop Latents
            lat_tile = latents[:, :, :, h:h_end, w:w_end].to(device)
            
            # 2. Crop Cond
            # Calculate cond coordinates
            hc, wc = h * cond_scale, w * cond_scale
            hc_end, wc_end = h_end * cond_scale, w_end * cond_scale
            
            # Ensure cond crop is within bounds (though it should be if ratio is correct)
            hc_end = min(hc_end, H_c)
            wc_end = min(wc_end, W_c)
            
            cond_tile = cond[:, :, :, hc:hc_end, wc:wc_end].to(device)
            
            # 3. Decode
            # TCDecoder.decode_video expects latents (B, F, C, H, W)
            frames_tile = self.TCDecoder.decode_video(
                lat_tile.transpose(1, 2),
                parallel=False,
                show_progress_bar=False,
                cond=cond_tile,
                decoding_msg=None
            ) # returns (B, F, C, H, W) in [0, 1] usually? 
            # Wait, standard TCDecoder returns [0, 1]? 
            # TCDecoder `decode_video` output:
            # "returns NTCHW RGB in ~[0, 1]"
            # BUT in `__call__` originally: `... .transpose(1, 2).mul_(2).sub_(1)`
            # So `__call__` expects [0, 1] output from decode_video and converts to [-1, 1] for later processing (ColorCorrector)?
            # ColorCorrector expects inputs.
            # `frames` returned by `__call__` are usually [-1, 1]?
            # Let's check `__call__` return logic.
            # `return frames[0]`.
            # Typically DiffSynth pipelines return [-1, 1] tensors?
            # Or [0, 1]?
            # ColorCorrector input `lq_image`?
            
            # Let's match existing `__call__` logic.
            # Existing: `frames = self.TCDecoder.decode_video(...).transpose(1, 2).mul_(2).sub_(1)`
            # So `decode_video` returns [0, 1]. `frames` becomes [-1, 1].
            # Then ColorCorrector is applied.
            
            # So here: frames_tile is [0, 1], (B, F, 3, H, W).
            # Convert to [-1, 1] and (B, 3, F, H, W).
            frames_tile = frames_tile.transpose(1, 2).mul_(2.0).sub_(1.0)
            
            # Ensure time dimension matches accumulator (trim if necessary)
            if frames_tile.shape[2] > out_F:
                frames_tile = frames_tile[:, :, :out_F, :, :]
            
            # Move to CPU for accumulation
            frames_tile = frames_tile.to("cpu")
            
            # 4. Mask
            # Using self.vae.build_mask logic
            # build_mask expects (..., H, W)
            # Border width in output pixels
            # Latent border was (ts_h - st_h). Output border is * vae_upscale.
            border_h = (ts_h - st_h) * vae_upscale
            border_w = (ts_w - st_w) * vae_upscale
            
            mask = self._build_mask(
                frames_tile,
                is_bound=(h==0, h+ts_h>=H, w==0, w+ts_w>=W),
                border_width=(border_h, border_w)
            ).to(dtype=frames_tile.dtype, device="cpu")
            
            # 5. Accumulate
            oh, ow = h * vae_upscale, w * vae_upscale
            oh_end = oh + frames_tile.shape[3]
            ow_end = ow + frames_tile.shape[4]
            
            value[:, :, :, oh:oh_end, ow:ow_end] += frames_tile * mask
            count[:, :, :, oh:oh_end, ow:ow_end] += mask
            
        return value / count

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt="",
        denoising_strength=1.0,
        seed=None,
        rand_device="gpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=False,
        tile_size=(60, 104),
        tile_stride=(30, 52),
        # Removed unused TeaCache arguments
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        LQ_video=None,
        is_full_block=False,
        if_buffer=False,
        topk_ratio=2.0,
        kv_ratio=3.0,
        local_range = 9,
        color_fix = True,
        decoding_msg = None,
        lq_bootstrap_windows: int = 7,
    ):
        # Only accept cfg=1.0 (Consistent with original code)
        assert cfg_scale == 1.0, "cfg_scale must be 1.0"

        # Requirement: init_cross_kv() must be called first
        if self.prompt_emb_posi is None or 'context' not in self.prompt_emb_posi:
            raise RuntimeError(
                "Cross-Attn KV not initialized. Please execute before calling __call__:\n"
                "    pipe.init_cross_kv()\n"
                "or pass custom context:\n"
                "    pipe.init_cross_kv(context_tensor=your_context_tensor)"
            )

        # Dimension Correction
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Tiler Parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Initialize Noise
        if if_buffer:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        else:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        # noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        # Streaming path needs at least 6 latent frames for the first step:
        # WanVideoDiT.SelfAttention enforces f==6 when starting a new stream block (no KV cache yet).
        # For short clips with if_buffer=True, (num_frames-1)//4 can be < 6 (e.g. 17 frames -> 4),
        # so we pad the latent time dimension by repeating the last frame.
        if latents.shape[2] < 6:
            pad = 6 - latents.shape[2]
            latents = torch.cat([latents, latents[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)

        # For short clips / realtime batches (e.g., 9/17 frames), the original formula becomes <= 0,
        # which would skip inference entirely. We guarantee at least one process step.
        process_total_num = max(1, (num_frames - 1) // 8 - 2)
        is_stream = True

        # Clear potential LQ_proj_in cache
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()

        latents_total = []
        self.TCDecoder.clean_mem()
        LQ_pre_idx = 0
        LQ_cur_idx = 0

        with torch.no_grad():
            for cur_process_idx in tqdm(range(process_total_num), desc="DiT Inference"):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    # Original code used 7 windows, which assumes 25 LQ frames are available:
                    # end index sequence: 1, 5, 9, 13, 17, 21, 25
                    # For realtime, allow fewer windows to reduce first-batch spike and to support
                    # batch-size < 25. We also cap by what current num_frames can cover.
                    max_windows_by_frames = (max(num_frames, 1) + 2) // 4 + 1
                    inner_loop_num = max(1, min(int(lq_bootstrap_windows), int(max_windows_by_frames)))
                    for inner_idx in range(inner_loop_num):
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            LQ_video[:, :, max(0, inner_idx*4-3):(inner_idx+1)*4-3, :, :]
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    # Keep original behavior for the default long-clip case (25 frames, 7 windows),
                    # but for reduced-window / short batches, advance LQ_cur_idx to the covered end.
                    if inner_loop_num == 7 and num_frames >= 25 and int(lq_bootstrap_windows) >= 7:
                        LQ_cur_idx = (inner_loop_num - 1) * 4 - 3  # legacy: 21
                    else:
                        LQ_cur_idx = min(num_frames, inner_loop_num * 4 - 3)
                    cur_latents = latents[:, :, :6, :, :]
                else:
                    LQ_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            LQ_video[:, :, cur_process_idx*8+17+inner_idx*4:cur_process_idx*8+21+inner_idx*4, :, :]
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_cur_idx = cur_process_idx*8+21+(inner_loop_num-2)*4
                    cur_latents = latents[:, :, 4+cur_process_idx*2:6+cur_process_idx*2, :, :]

                # Inference (No motion_controller / vace)
                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range = local_range,
                )

                # Update latent
                cur_latents = cur_latents - noise_pred_posi
                latents_total.append(cur_latents)
                LQ_pre_idx = LQ_cur_idx

            latents = torch.cat(latents_total, dim=2)

            # Decode: for very short / reduced-bootstrap streams, disable cond to avoid
            # temporal/channel mismatches inside TCDecoder when latent time has been padded.
            use_cond_for_decode = (LQ_cur_idx >= 25 and num_frames >= 25 and int(lq_bootstrap_windows) >= 7)
            cond_for_decode = LQ_video[:, :, :LQ_cur_idx, :, :] if (LQ_video is not None and use_cond_for_decode) else None

            if tiled:
                frames = self._tiled_decode(
                    latents,
                    cond=cond_for_decode,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    decoding_msg=decoding_msg if decoding_msg else "Decoding video (Tiled)"
                )
            else:
                frames = self.TCDecoder.decode_video(
                    latents.transpose(1, 2),
                    parallel=False, 
                    show_progress_bar=True, 
                    cond=cond_for_decode,
                    decoding_msg=decoding_msg if decoding_msg else "Decoding video" # If no msg, default to "Decoding video"
                ).transpose(1, 2).mul_(2).sub_(1)

            # Color correction (wavelet)
            try:
                if color_fix:
                    frames = self.ColorCorrector(
                        frames.to(device=LQ_video.device),
                        LQ_video[:, :, :frames.shape[2], :, :],
                        clip_range=(-1, 1),
                        chunk_size=16,
                        method='adain'
                    )
            except Exception as e:
                print(f"[ColorFix Error] {e}")
                pass

        return frames[0]




# -----------------------------
# Simplified Model Forward Wrapper (No vace / No motion_controller)
# -----------------------------
def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    tea_cache: Optional[object] = None, # Reserved for compatibility
    use_unified_sequence_parallel: bool = False,
    LQ_latents: Optional[torch.Tensor] = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: Optional[list[torch.Tensor]] = None,
    pre_cache_v: Optional[list[torch.Tensor]] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod : torch.Tensor = None,
    t : torch.Tensor = None,
    local_range: int = 9,
    **kwargs,
):
    # patchify
    x, (f, h, w) = dit.patchify(x)

    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)

    # RoPE Position (Segmented)
    if cur_process_idx == 0:
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    else:
        freqs = torch.cat([
            dit.freqs[0][4 + cur_process_idx*2:4 + cur_process_idx*2 + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Unified Sequence Parallel (Default OFF)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group)
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # Block Stacking
    for block_id, block in enumerate(dit.blocks):
        if LQ_latents is not None and block_id < len(LQ_latents):
            lq = LQ_latents[block_id]
            # Robustness for short / padded streams:
            # LQ_proj_in.stream_forward emits one latent-time slice per call (first call returns None),
            # so with reduced bootstrap windows, lq token length can be smaller than current x.
            # We pad by repeating the last time-slice worth of tokens so shapes match.
            if lq is not None and lq.shape[1] != x.shape[1]:
                if lq.shape[1] > x.shape[1]:
                    lq = lq[:, :x.shape[1], :]
                else:
                    # tokens per latent frame at current resolution
                    tokens_per_f = x.shape[1] // f
                    if tokens_per_f > 0 and lq.shape[1] >= tokens_per_f:
                        need = x.shape[1] - lq.shape[1]
                        reps = (need + tokens_per_f - 1) // tokens_per_f
                        pad_chunk = lq[:, -tokens_per_f:, :].repeat(1, reps, 1)[:, :need, :]
                        lq = torch.cat([lq, pad_chunk], dim=1)
                    else:
                        # Fallback: repeat last token if shape is unexpectedly small
                        need = x.shape[1] - lq.shape[1]
                        lq = torch.cat([lq, lq[:, -1:, :].repeat(1, need, 1)], dim=1)
                LQ_latents[block_id] = lq
            x = x + LQ_latents[block_id]
        x, last_pre_cache_k, last_pre_cache_v = block(
            x, context, t_mod, freqs, f, h, w,
            local_num, topk,
            block_id=block_id,
            kv_len=kv_len,
            is_full_block=is_full_block,
            is_stream=is_stream,
            pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
            pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
            local_range = local_range,
        )
        if pre_cache_k is not None: pre_cache_k[block_id] = last_pre_cache_k
        if pre_cache_v is not None: pre_cache_v[block_id] = last_pre_cache_v

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sp_group
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x, pre_cache_k, pre_cache_v
