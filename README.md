# tongflow-modal-fastwan

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text-to-video generation with **FastWan-QAD-FP8-1.3B** (`FastVideo/FastWan-QAD-FP8-1.3B`), running on a GPU via [Modal](https://modal.com).

A quantization-aware-distilled Wan2.1-T2V-1.3B: 3-step denoising, FP8 linear layers, [SageAttention](https://github.com/thu-ml/SageAttention) backend, decoded by the full Wan VAE for quality. Generates ~5s of 480p (832×480, 16 fps) video in a few seconds on Hopper GPUs.

## Capabilities

- **Text → video** (`text-gen-video`) — generate a short video clip from a text prompt.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The FastWan weights are public — no Hugging Face token required.

## Notes

- Runs on an **H100** GPU (FP8-capable, 80 GB). The model is small, but the Wan UMT5-XXL text encoder, torch.compile workspaces, and 81-frame video activations together exceed a 48 GB card. The cold-start build compiles the DiT, so the first request is slow; subsequent requests on a warm container are fast.
- `duration` is rounded to the Wan frame grid (`4k + 1` frames at 16 fps) and capped at **15s** (241 frames). The model is natively a ~5s clip model and generates the whole clip in one pass, so attention activation memory grows with the square of the frame count — 30s OOMs even on an 80 GB H100. Longer requests are clamped to 15s. `height`/`width` default to the native 480×832; `seed` is honored. The model uses no classifier-free guidance.
- The image installs the PyPI `sageattention` (Triton) build, which works out of the box. For maximum FP8 throughput, build SageAttention2++ from source instead.
