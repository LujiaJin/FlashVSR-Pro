#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 FlashVSR-Pro 的实时直播视频超分脚本

整体链路（推荐示例）：

1. 主播端（OBS）：
   - 推流地址：rtmp://你的服务器IP:1935/live/src
   - 分辨率建议：1280x720 或 1920x1080，编码 H.264 + AAC

2. 服务器端（已启动 SRS）：
   - 本脚本从 SRS 拉取低清流：  --input-rtmp  rtmp://服务器IP:1935/live/src
   - 使用 FlashVSR-Pro 进行超分：--mode tiny-long --tile-dit --tile-vae
   - 将高清结果推回 SRS：        --output-rtmp rtmp://服务器IP:1935/live/sr
   - 当指定 --keep-audio 时，会从 input-rtmp 流中复制音频，保证声音不变。

3. 观众端：
   - 直接播放 SRS 的高清流：rtmp://服务器IP:1935/live/sr（或对应的 HLS 地址）

用法示例（2 倍超分，保持音频）：

python livestream_infer.py \
  --input-rtmp  rtmp://127.0.0.1:1935/live/src \
  --output-rtmp rtmp://127.0.0.1:1935/live/sr  \
  --input-width 1280 --input-height 720        \
  --mode tiny-long --tile-dit --tile-vae      \
  --scale 2.0 --keep-audio
"""

import os
import sys
import time
import argparse
import threading
import queue
from collections import deque
from typing import List, Optional

import numpy as np
from PIL import Image

import subprocess

import torch

# 确保可以导入同目录下的 infer.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from infer import (  # type: ignore
    init_pipeline,
    compute_scaled_and_target_dims,
    process_batch_gpu,
    tensor2video,
    NVENC_AVAILABLE,
    TILE_AVAILABLE,
    apply_tiled_inference_simple,
)


class StreamBuffer:
    """简单的线程安全帧缓冲队列"""

    def __init__(self, max_size: int = 300):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_count = 0

    def put(self, frame: Image.Image) -> None:
        with self.lock:
            self.buffer.append(frame)
            self.frame_count += 1

    def get_batch(self, batch_size: int) -> Optional[List[Image.Image]]:
        with self.lock:
            if len(self.buffer) >= batch_size:
                batch = [self.buffer.popleft() for _ in range(batch_size)]
                return batch
        return None

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


class FlashVSRRealtime:
    """
    基于 infer.py 的实时 FlashVSR 推理器

    这里不重新实现模型加载逻辑，而是直接复用 infer.py 中的 init_pipeline、
    compute_scaled_and_target_dims、process_batch_gpu 等函数，以保证行为一致。
    """

    def __init__(
        self,
        mode: str = "tiny-long",
        tile_dit: bool = False,
        tile_vae: bool = False,
        tile_size: int = 256,
        overlap: int = 24,
        device: str = "cuda",
        dtype: str = "bf16",
        scale: float = 2.0,
        input_width: int = 1280,
        input_height: int = 720,
        sparse_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 11,
        seed: int = 0,
        warmup: bool = True,
        warmup_frames: int = 9,
        lq_bootstrap_windows: int = 7,
    ):
        self.mode = mode
        self.tile_dit = tile_dit
        self.tile_vae = tile_vae
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.dtype_str = dtype
        self.scale = scale
        self.in_w = input_width
        self.in_h = input_height
        self.sparse_ratio = sparse_ratio
        self.kv_ratio = kv_ratio
        self.local_range = local_range
        self.seed = seed
        self.warmup_enabled = warmup
        self.warmup_frames = warmup_frames
        self._warmup_done = False
        self.lq_bootstrap_windows = lq_bootstrap_windows

        # dtype 映射
        if self.dtype_str == "fp16":
            self.dtype_torch = torch.float16
        elif self.dtype_str == "bf16":
            self.dtype_torch = torch.bfloat16
        else:
            self.dtype_torch = torch.float32

        # 启用 TF32 & cudnn benchmark，和 infer.py 保持一致
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # 计算放大后的尺寸以及对齐到 128 倍数的推理尺寸
        self.sW, self.sH, self.tW, self.tH = compute_scaled_and_target_dims(
            self.in_w, self.in_h, scale=self.scale, multiple=128
        )
        self.exact_w = self.sW
        self.exact_h = self.sH

        print(
            f"[FlashVSRRealtime] Input {self.in_w}x{self.in_h}, "
            f"scale {self.scale}x -> target {self.exact_w}x{self.exact_h}, "
            f"padded to {self.tW}x{self.tH}"
        )

        # 构造一个简单的 Namespace 交给 infer.init_pipeline 复用加载逻辑
        args = argparse.Namespace()
        args.mode = self.mode
        args.device = self.device
        args.dtype = self.dtype_str
        args.tile_vae = self.tile_vae
        args.tile_size = self.tile_size
        args.overlap = self.overlap

        # init_pipeline 内部会根据 mode 选择 VAE / TCDecoder 并加载 DiT
        self.pipe, self.vae_instance = init_pipeline(args)

        if self.device == "cuda":
            # Reduce first-run jitter by keeping allocator behavior stable
            try:
                torch.cuda.set_per_process_memory_fraction(1.0)
            except Exception:
                pass

        # Warmup once so the first real batch doesn't pay compile/alloc costs
        if self.warmup_enabled:
            try:
                self.warmup(self.warmup_frames)
            except Exception as e:
                print(f"[FlashVSRRealtime] Warmup failed (will continue): {e}")

    def warmup(self, num_frames: int) -> None:
        """
        One-time warmup to reduce cold-start latency:
        - GPU preprocess kernels (resize/pad/normalize)
        - DiT forward path + VAE/TCDecoder decode path
        """
        if self._warmup_done:
            return
        if num_frames < 5:
            num_frames = 5

        print(f"[FlashVSRRealtime] Warmup started (frames={num_frames}, tile_dit={self.tile_dit}, tile_vae={self.tile_vae})")
        dummy = [Image.new("RGB", (self.in_w, self.in_h), (0, 0, 0)) for _ in range(num_frames)]

        # Run a minimal end-to-end pass. This intentionally reuses process_batch() logic
        # to hit the same code path as real streaming batches.
        start = time.time()
        _ = self.process_batch(dummy)
        if self.device == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed = time.time() - start
        self._warmup_done = True
        print(f"[FlashVSRRealtime] Warmup done in {elapsed:.2f}s")

    def process_batch(self, frames: List[Image.Image]) -> List[Image.Image]:
        """
        处理一批帧，返回超分后的帧列表（保持帧数一致）。
        """
        if not frames:
            return frames

        num_frames = len(frames)
        if num_frames < 5:
            # 帧数太少时直接透传，避免模型对极短序列不稳定
            print(f"[FlashVSRRealtime] Batch too small ({num_frames}), bypass SR")
            return frames

        # 转为 numpy 数组列表，形状 (H, W, C)，uint8
        batch_arr = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]

        # 利用 infer.py 中的 GPU 预处理逻辑：resize + pad + 归一化
        # 输出形状 (B, C, H, W)，位于 GPU 上
        lq_batch = process_batch_gpu(
            batch_arr,
            sH=self.sH,
            sW=self.sW,
            tH=self.tH,
            tW=self.tW,
            dtype=self.dtype_torch,
            device=self.device,
        )

        # 变换为 FlashVSR 需要的形状：1, C, T, H, W
        LQ = lq_batch.permute(1, 0, 2, 3).unsqueeze(0)

        # 组装与 infer.py 一致的 pipeline 参数
        pipeline_kwargs = {
            "prompt": "",
            "negative_prompt": "",
            "cfg_scale": 1.0,
            "num_inference_steps": 1,
            "seed": self.seed,
            "LQ_video": LQ,
            "num_frames": num_frames,
            "height": self.tH,
            "width": self.tW,
            "is_full_block": False,
            "if_buffer": True,
            "topk_ratio": self.sparse_ratio * 768 * 1280 / (self.tH * self.tW),
            "kv_ratio": self.kv_ratio,
            "local_range": self.local_range,
            "color_fix": True,
            "lq_bootstrap_windows": self.lq_bootstrap_windows,
        }

        # VAE 分块参数（仅在 tile-vae 时生效）
        if self.tile_vae:
            vae_tile_size_latent = max(32, self.tile_size // 8)
            vae_overlap_latent = max(4, self.overlap // 8)
            pipeline_kwargs["tiled"] = True
            pipeline_kwargs["tile_size"] = (vae_tile_size_latent, vae_tile_size_latent)
            pipeline_kwargs["tile_stride"] = (
                vae_tile_size_latent - vae_overlap_latent,
                vae_tile_size_latent - vae_overlap_latent,
            )
            print(
                f"[FlashVSRRealtime] VAE tiling: size={pipeline_kwargs['tile_size']}, "
                f"stride={pipeline_kwargs['tile_stride']}"
            )

        # 执行推理
        start = time.time()
        with torch.inference_mode():
            if self.tile_dit and TILE_AVAILABLE:
                print(
                    f"[FlashVSRRealtime] DiT tiled inference, tile_size={self.tile_size}, overlap={self.overlap}"
                )
                tile_kwargs = dict(pipeline_kwargs)
                tile_kwargs.pop("LQ_video", None)
                vae_tile_size_tuple = tile_kwargs.pop("tile_size", None)

                video = apply_tiled_inference_simple(
                    self.pipe,
                    LQ,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    tile_size_vae=vae_tile_size_tuple,
                    **tile_kwargs,
                )
            else:
                video = self.pipe(**pipeline_kwargs)

        elapsed = time.time() - start
        print(
            f"[FlashVSRRealtime] Processed {num_frames} frames in {elapsed:.2f}s "
            f"({num_frames / max(elapsed, 1e-6):.1f} FPS)"
        )

        # 输出形状：1, C, T, H, W，需要裁剪回精确分辨率 self.exact_h, self.exact_w
        if video.shape[-2] != self.exact_h or video.shape[-1] != self.exact_w:
            curr_h, curr_w = video.shape[-2], video.shape[-1]
            pad_h = curr_h - self.exact_h
            pad_w = curr_w - self.exact_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            video = video[
                ...,
                pad_top : pad_top + self.exact_h,
                pad_left : pad_left + self.exact_w,
            ]

        # 转回 numpy 帧（列表）
        frames_np = tensor2video(video)

        # 保证输出帧数与输入一致
        if len(frames_np) > num_frames:
            frames_np = frames_np[:num_frames]
        elif len(frames_np) < num_frames:
            frames_np.extend([frames_np[-1]] * (num_frames - len(frames_np)))

        out_frames = [Image.fromarray(f) for f in frames_np]
        return out_frames


class RTMPCapture:
    """通过 ffmpeg+pipe 从 SRS/RTMP 拉取原始帧"""

    def __init__(self, rtmp_url: str, fps: int, width: int, height: int):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        print(f"[RTMPCapture] Connecting to {self.rtmp_url} ...")
        cmd = [
            "ffmpeg",
            "-i",
            self.rtmp_url,
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-f",
            "rawvideo",
            "-",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * self.width * self.height * 3,
        )
        print("[RTMPCapture] Started.")

    def read_frame(self) -> Optional[Image.Image]:
        if self.process is None or self.process.stdout is None:
            return None
        frame_size = self.width * self.height * 3
        data = self.process.stdout.read(frame_size)
        if len(data) != frame_size:
            return None
        frame = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)
        return Image.fromarray(frame, "RGB")

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            self.process.wait()


class RTMPPush:
    """
    将处理后帧通过 ffmpeg 推回 SRS。

    当 keep_audio=True 且提供 audio_source_url 时，ffmpeg 会从该 RTMP 流复制音频，
    从而实现“保留原始音频”的效果，等价于 infer.py 的 --keep-audio 语义。
    """

    def __init__(
        self,
        rtmp_url: str,
        fps: int,
        width: int,
        height: int,
        keep_audio: bool = False,
        audio_source_url: Optional[str] = None,
    ):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.keep_audio = keep_audio
        self.audio_source_url = audio_source_url
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        print(f"[RTMPPush] Pushing to {self.rtmp_url} ...")

        if self.keep_audio and self.audio_source_url:
            # 双输入：0 为超分后原始帧，1 为原始 RTMP（只取音频）
            cmd = [
                "ffmpeg",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:",
                "-i",
                self.audio_source_url,
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-b:v",
                "6000k",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-flvflags",
                "no_duration_filesize",
                "-f",
                "flv",
                self.rtmp_url,
            ]
        else:
            # 仅视频
            cmd = [
                "ffmpeg",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-b:v",
                "6000k",
                "-flvflags",
                "no_duration_filesize",
                "-f",
                "flv",
                self.rtmp_url,
            ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * self.width * self.height * 3,
        )

        # 打印 ffmpeg 日志，便于排查问题
        def _log_ffmpeg() -> None:
            assert self.process is not None
            try:
                for line in iter(self.process.stderr.readline, b""):
                    if not line:
                        break
                    print(
                        f"[RTMPPush ffmpeg] {line.decode(errors='ignore').strip()}"
                    )
            except Exception:
                pass

        t = threading.Thread(target=_log_ffmpeg, daemon=True)
        t.start()
        print("[RTMPPush] Started.")

    def write_frame(self, frame: Image.Image) -> bool:
        if not self.process or not self.process.stdin:
            return False
        try:
            arr = np.array(frame.convert("RGB"), dtype=np.uint8)
            self.process.stdin.write(arr.tobytes())
            return True
        except Exception as e:
            print(f"[RTMPPush] write_frame error: {e}")
            return False

    def stop(self) -> None:
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except Exception:
                pass
            self.process.terminate()
            self.process.wait()


def capture_thread(capture: RTMPCapture, buffer: StreamBuffer) -> None:
    try:
        while True:
            frame = capture.read_frame()
            if frame is None:
                print("[CaptureThread] Stream ended.")
                break
            buffer.put(frame)
            if buffer.frame_count % 30 == 0:
                print(
                    f"[CaptureThread] Captured {buffer.frame_count} frames, "
                    f"buffer size={buffer.size()}"
                )
    except Exception as e:
        print(f"[CaptureThread] Error: {e}")


def process_thread(
    flashvsr: FlashVSRRealtime,
    buffer: StreamBuffer,
    output_queue: "queue.Queue[Image.Image]",
    batch_size: int,
    bootstrap_batch_size: Optional[int] = None,
) -> None:
    try:
        first = True
        bs0 = int(bootstrap_batch_size) if bootstrap_batch_size is not None else int(batch_size)
        if bs0 < 5:
            bs0 = 5
        while True:
            cur_bs = bs0 if first else int(batch_size)
            batch = buffer.get_batch(cur_bs)
            if batch:
                out_frames = flashvsr.process_batch(batch)
                for f in out_frames:
                    output_queue.put(f)
                first = False
            else:
                time.sleep(0.05)
    except Exception as e:
        print(f"[ProcessThread] Error: {e}")


def push_thread(
    pusher: RTMPPush,
    output_queue: "queue.Queue[Image.Image]",
) -> None:
    """
    推送线程：当队列短暂为空时，重复上一帧，避免 ffmpeg 因断流输出噪点。
    """
    last_frame: Optional[Image.Image] = None
    pushed = 0
    try:
        while True:
            try:
                frame = output_queue.get(timeout=1.0)
                last_frame = frame
            except queue.Empty:
                if last_frame is None:
                    last_frame = Image.new(
                        "RGB", (pusher.width, pusher.height), (0, 0, 0)
                    )
                frame = last_frame
                print("[PushThread] Queue empty, repeat last frame.")

            ok = pusher.write_frame(frame)
            if not ok:
                print("[PushThread] write_frame failed.")
                break
            pushed += 1
            if pushed % 30 == 0:
                print(f"[PushThread] Pushed {pushed} frames")
    except Exception as e:
        print(f"[PushThread] Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlashVSR-Pro Livestream Super-Resolution (RTMP → RTMP)"
    )
    parser.add_argument(
        "--input-rtmp",
        type=str,
        required=True,
        help="输入 RTMP 地址（来自 OBS -> SRS），如 rtmp://127.0.0.1:1935/live/src",
    )
    parser.add_argument(
        "--output-rtmp",
        type=str,
        required=True,
        help="输出 RTMP 地址（推回 SRS），如 rtmp://127.0.0.1:1935/live/sr",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=1280,
        help="输入流分辨率宽度（需与 OBS 设置一致）",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=720,
        help="输入流分辨率高度（需与 OBS 设置一致）",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="帧率（需与 OBS 设置大致一致）"
    )

    # FlashVSR 推理参数（对齐 infer.py）
    parser.add_argument(
        "--mode",
        type=str,
        default="tiny-long",
        choices=["full", "tiny", "tiny-long"],
        help="FlashVSR 模式，推荐 tiny-long",
    )
    parser.add_argument(
        "--tile-dit",
        action="store_true",
        help="对 DiT 启用分块推理，降低显存（推荐开启）",
    )
    parser.add_argument(
        "--tile-vae",
        action="store_true",
        help="对 VAE 解码启用分块（长视频 / 大分辨率时推荐开启）",
    )
    parser.add_argument(
        "--tile-size", type=int, default=256, help="DiT 分块大小（像素）"
    )
    parser.add_argument(
        "--overlap", type=int, default=24, help="分块重叠大小（像素）"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="超分倍率（理论上 4.0 质量最佳，2.0 更易实时）",
    )
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=2.0,
        help="稀疏注意力比例，越小越快但略降质",
    )
    parser.add_argument(
        "--kv-ratio",
        type=float,
        default=3.0,
        help="KV cache 比例",
    )
    parser.add_argument(
        "--local-range",
        type=int,
        default=11,
        help="局部注意力范围",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="随机种子（可固定风格）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu，推荐 cuda",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="计算精度，推荐 bf16",
    )

    # 直播相关参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=9,
        help="每次送入 FlashVSR 的帧数（8n+1 更稳定，如 9, 17）",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=0,
        help="首段用于尽快出画面的 batch-size（0 表示与 batch-size 相同；推荐 9/13/17）",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300,
        help="输入帧缓冲最大长度",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="保留原始 RTMP 音频（由 ffmpeg 从 input-rtmp 复制）",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="禁用启动预热（会增加首帧时延，但启动更快）",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="预热用的帧数（默认=0 表示使用 batch-size；建议 9/17/25）",
    )
    parser.add_argument(
        "--lq-bootstrap-windows",
        type=int,
        default=7,
        help="首段 LQ 特征预取窗口数（7 对应 25 帧；实时可用 2~4 降低首帧尖峰并支持 batch<25）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FlashVSR-Pro Livestream Super-Resolution")
    print("=" * 80)
    print(f"Input RTMP : {args.input_rtmp}")
    print(f"Output RTMP: {args.output_rtmp}")
    print(f"Input Size : {args.input_width}x{args.input_height} @ {args.fps}fps")
    print(f"Mode       : {args.mode}")
    print(
        f"Scale      : {args.scale}x, tile-dit={args.tile_dit}, "
        f"tile-vae={args.tile_vae}, keep-audio={args.keep_audio}"
    )
    print("=" * 80)

    # 初始化缓冲与队列
    buffer = StreamBuffer(max_size=args.buffer_size)
    output_queue: "queue.Queue[Image.Image]" = queue.Queue(maxsize=100)

    # 初始化 RTMP 拉流 / 推流
    capture = RTMPCapture(
        args.input_rtmp,
        fps=args.fps,
        width=args.input_width,
        height=args.input_height,
    )

    # 根据 scale 计算输出尺寸（对齐 infer.py 的对齐策略）
    sW, sH, tW, tH = compute_scaled_and_target_dims(
        args.input_width, args.input_height, scale=args.scale, multiple=128
    )
    out_w, out_h = sW, sH

    pusher = RTMPPush(
        args.output_rtmp,
        fps=args.fps,
        width=out_w,
        height=out_h,
        keep_audio=args.keep_audio,
        audio_source_url=args.input_rtmp if args.keep_audio else None,
    )

    # 初始化 FlashVSR 实时推理器
    try:
        flashvsr = FlashVSRRealtime(
            mode=args.mode,
            tile_dit=args.tile_dit,
            tile_vae=args.tile_vae,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=args.device,
            dtype=args.dtype,
            scale=args.scale,
            input_width=args.input_width,
            input_height=args.input_height,
            sparse_ratio=args.sparse_ratio,
            kv_ratio=args.kv_ratio,
            local_range=args.local_range,
            seed=args.seed,
            warmup=not args.no_warmup,
            warmup_frames=args.warmup_frames if args.warmup_frames > 0 else args.batch_size,
            lq_bootstrap_windows=args.lq_bootstrap_windows,
        )
    except Exception as e:
        print(f"[Main] Failed to init FlashVSR: {e}")
        return

    # 启动 RTMP 拉流 / 推流
    try:
        capture.start()
        pusher.start()
    except Exception as e:
        print(f"[Main] Failed to start RTMP IO: {e}")
        return

    # 启动 3 个工作线程
    t_cap = threading.Thread(
        target=capture_thread, args=(capture, buffer), daemon=True
    )
    t_proc = threading.Thread(
        target=process_thread,
        args=(
            flashvsr,
            buffer,
            output_queue,
            args.batch_size,
            args.bootstrap_batch_size if args.bootstrap_batch_size > 0 else None,
        ),
        daemon=True,
    )
    t_push = threading.Thread(
        target=push_thread, args=(pusher, output_queue), daemon=True
    )

    t_cap.start()
    t_proc.start()
    t_push.start()

    # 主线程仅负责阻塞与 Ctrl-C 退出
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt, shutting down ...")
    finally:
        try:
            capture.stop()
        except Exception:
            pass
        try:
            pusher.stop()
        except Exception:
            pass
        try:
            flashvsr.vae_instance.clean_memory()
        except Exception:
            pass
        torch.cuda.empty_cache()
        print("[Main] Exit.")


if __name__ == "__main__":
    main()

