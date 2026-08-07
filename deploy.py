"""Modal deploy entry for fastwan (single file).

Text-to-video with FastWan-QAD-FP8-1.3B: a quantization-aware-distilled
Wan2.1-T2V-1.3B that denoises in 3 steps with FP8 linear layers and the
SageAttention backend, decoded by the full Wan VAE. Runs on a Hopper GPU
(FP8-capable); ~5s of 480p video in a few seconds.

Deploy:
  modal deploy deploy.py

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import modal
from tongflow import deploy
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset
from tongflow.slots import node_slot


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "FastVideo/FastWan-QAD-FP8-1.3B")
MODEL_DIR = f"/models/{REPO_ID}"

# Diffusion sampling defaults — plugin-internal, not part of the ABI contract.
# QAD distillation collapses denoising to 3 steps at guidance 1.0 (no CFG).
DEFAULT_INFER_STEPS = 3
DEFAULT_GUIDANCE = 1.0
DEFAULT_FPS = 16
# Native training resolution (832x480, 480p) and clip length (81 frames = 5s).
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


# Hard ceiling on clip length. The model is natively a ~5s (81-frame) model and
# generates the whole clip in a single pass, so DiT attention activation memory
# grows with frame count squared: 30s (~477 frames) OOMs even on an 80 GB H100
# (one activation alone needs ~34 GB). 15s (241 frames) is the safe single-pass
# ceiling here. Raise cautiously only if the startup memory logs show headroom.
MAX_FRAMES = 241


def _frames_from_duration(duration: float, fps: int = DEFAULT_FPS) -> int:
    """Seconds -> Wan frame count. The Wan VAE compresses time by 4, so the
    pipeline requires ``num_frames = 4k + 1``. Clamp to ~0.25s..15s."""
    n = max(1, round(float(duration) * fps))
    n = ((n - 1) // 4) * 4 + 1
    return max(5, min(n, MAX_FRAMES))


# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "build-essential")
    # FP8 quantization (FP8Config) only landed on FastVideo's main branch; the
    # PyPI 0.2.0 release predates it. Pin the GitHub commit that ships FP8.
    # fastvideo pins its own torch==2.11.0 + fastvideo-kernel; let it resolve.
    .pip_install(
        "git+https://github.com/hao-ai-lab/FastVideo.git"
        "@4f3ad3f6df9327257f08f4c45d24154b52c06616"
    )
    # SAGE_ATTN backend imports `from sageattention import sageattn`. The PyPI
    # build (Triton) works out of the box; build SageAttention2++ from source
    # for max FP8 throughput if needed.
    .pip_install("sageattention")
    .pip_install("tongflow==0.2.21", "fastapi[standard]")
    .env(
        {
            "FASTVIDEO_ATTENTION_BACKEND": "SAGE_ATTN",
            "HF_HOME": "/models/hf",
            # Reduce allocator fragmentation across compile + video activations.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

with image.imports():
    import tempfile

    import imageio
    import torch


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="H100",
    volumes={"/models": volume},
    timeout=1800,
)
class Inference:
    @modal.enter()
    def load(self):
        import fastvideo
        from fastvideo import VideoGenerator
        from fastvideo.layers.quantization.fp8_config import FP8Config

        # Confirm the FP8-capable build is the one actually installed. The PyPI
        # release predates FP8Config; this import fails loudly on the old build.
        print(f"fastvideo {getattr(fastvideo, '__version__', '?')} @ {fastvideo.__file__}")

        os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")

        # FP8 e4m3 weight quantization of the DiT linears; activations quantized
        # dynamically at runtime. Decode uses the full Wan VAE (output_type="pil")
        # rather than the tiny TAEHV autoencoder: noticeably sharper / fewer
        # artifacts at a small speed cost. The Wan VAE streams frames with a
        # feature cache, so decode memory stays bounded even for long clips.
        self.generator = VideoGenerator.from_pretrained(
            MODEL_DIR,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            # VAE lives on CPU during the heavy DiT denoise, moves to GPU only to
            # decode — keeps the denoise-phase memory peak low.
            vae_cpu_offload=True,
            # Offload the ~11 GB UMT5-XXL text encoder to CPU between the (single)
            # prompt encode and the heavy DiT denoise, freeing peak GPU headroom.
            text_encoder_cpu_offload=True,
            pin_cpu_memory=False,
            enable_torch_compile=True,
            enable_torch_compile_vae=False,
            output_type="pil",
            transformer_quant=FP8Config(granularity="tensor"),
        )

        # Pay the torch.compile + cuDNN warmup cost once on cold start so the
        # first user request runs at steady-state latency.
        self.generator.generate(
            request={
                "prompt": "warmup",
                "sampling": {
                    "num_inference_steps": DEFAULT_INFER_STEPS,
                    "guidance_scale": DEFAULT_GUIDANCE,
                    "num_frames": 81,
                    "height": DEFAULT_HEIGHT,
                    "width": DEFAULT_WIDTH,
                    "fps": DEFAULT_FPS,
                },
                "output": {"save_video": False, "return_frames": True},
            }
        )

    def _generate_core(
        self,
        *,
        prompt: str,
        duration: float,
        seed: int,
        height: int,
        width: int,
    ) -> bytes:
        num_frames = _frames_from_duration(duration)
        result = self.generator.generate(
            request={
                "prompt": prompt,
                "sampling": {
                    "num_inference_steps": DEFAULT_INFER_STEPS,
                    "guidance_scale": DEFAULT_GUIDANCE,
                    "seed": int(seed),
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "fps": DEFAULT_FPS,
                },
                "output": {"save_video": False, "return_frames": True},
            }
        )
        # result.frames: list[np.ndarray] (uint8 HWC), decoded by the Wan VAE.
        frames = result.frames
        # Release the latents/activations before encoding so a warm container's
        # next request starts from a clean allocator (does not lower the single
        # forward pass's peak, only inter-request residue).
        del result
        gc.collect()
        torch.cuda.empty_cache()

        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            imageio.mimsave(path, frames, fps=DEFAULT_FPS, format="mp4")
            with open(path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_VIDEO)
    def text_gen_video(self, input: TextGenVideoInput) -> TextGenVideoOutput:
        text = (input.text or "").strip()
        if not text:
            return TextGenVideoOutput(success=False, error="Missing text prompt")
        try:
            raw = self._generate_core(
                prompt=text,
                duration=input.duration,
                seed=int(input.seed) if input.seed is not None else 42,
                height=input.height if input.height is not None else DEFAULT_HEIGHT,
                width=input.width if input.width is not None else DEFAULT_WIDTH,
            )
        except Exception as e:  # pragma: no cover
            return TextGenVideoOutput(success=False, error=str(e))
        return TextGenVideoOutput(success=True, video=asset(raw, mime="video/mp4"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

