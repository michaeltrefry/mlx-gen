"""Dump the Qwen-Image **ControlNet (strict pose)** end-to-end golden for the Rust port
(epic 3401 / sc-3574).

Reference = the **diffusers** `QwenImageControlNetPipeline` + `QwenImageControlNetModel` (InstantX
`Qwen-Image-ControlNet-Union`, DWPose-trained, Apache-2.0) — the exact torch path SceneWorks runs
today for the Qwen strict-pose tier. Runs one render on a DWPose skeleton control image and dumps the
control image (RGB8), the per-block controlnet residuals, the final latents, and the decoded image so
`mlx-gen-qwen-image/tests/control_real_weights.rs::e2e_matches_diffusers_golden` can compare.

Unlike the other goldens (dumped from the frozen mflux fork), the fork has **no Qwen ControlNet**, so
this reference is raw diffusers/torch (MPS or CPU). It is therefore a *cross-backend* reference: the
Rust↔torch comparison carries the established Qwen mixed-precision-vs-bf16 floor (the test gates on
the same %px>8 tolerance as the base e2e), not bit-exactness.

Requires a torch venv with diffusers (>= 0.35) + the InstantX checkpoint and base Qwen-Image in the
HF cache, and a DWPose skeleton PNG:
    python -m venv /tmp/diffusers-venv && /tmp/diffusers-venv/bin/pip install \
        "diffusers>=0.35" transformers accelerate safetensors torch pillow
    CONTROL_IMAGE=~/dwpose_skeleton.png \
        /tmp/diffusers-venv/bin/python tools/dump_qwen_control_golden.py

Env:
    CONTROL_IMAGE  path to the DWPose skeleton PNG (required for a faithful golden; a synthetic
                   gradient placeholder is used if unset — exercises the path but is not a real pose).
    PROMPT, SEED, STEPS, GUIDANCE, CONTROL_SCALE, SIZE  generation knobs (mirror the Rust test).
Output (gitignored): tools/golden/qwen_control_golden.safetensors
"""

import os

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageControlNetModel, QwenImageControlNetPipeline
from safetensors.numpy import save_file

BASE = os.environ.get("QWEN_IMAGE_REPO", "Qwen/Qwen-Image")
CONTROL = os.environ.get("QWEN_CONTROL_REPO", "InstantX/Qwen-Image-ControlNet-Union")
PROMPT = os.environ.get("PROMPT", "a person standing, photorealistic, studio lighting")
NEGATIVE = os.environ.get("NEGATIVE", " ")
SEED = int(os.environ.get("SEED", "42"))
STEPS = int(os.environ.get("STEPS", "20"))
GUIDANCE = float(os.environ.get("GUIDANCE", "4.0"))
CONTROL_SCALE = float(os.environ.get("CONTROL_SCALE", "1.0"))
SIZE = int(os.environ.get("SIZE", "512"))  # square; must be a multiple of 16

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16

# --- control image: a real DWPose skeleton (CONTROL_IMAGE) or a synthetic placeholder ---
if os.environ.get("CONTROL_IMAGE"):
    control_image = Image.open(os.environ["CONTROL_IMAGE"]).convert("RGB").resize((SIZE, SIZE))
else:
    print("WARNING: CONTROL_IMAGE unset — using a synthetic gradient (NOT a real pose skeleton)")
    grad = np.tile(np.linspace(0, 255, SIZE, dtype=np.uint8)[None, :, None], (SIZE, 1, 3))
    control_image = Image.fromarray(grad)

controlnet = QwenImageControlNetModel.from_pretrained(CONTROL, torch_dtype=dtype)
pipe = QwenImageControlNetPipeline.from_pretrained(BASE, controlnet=controlnet, torch_dtype=dtype)
pipe.to(device)

# Capture the per-step controlnet residuals (first step) by wrapping the controlnet forward.
captured = {}
_orig = controlnet.forward


def _capture(*args, **kwargs):
    out = _orig(*args, **kwargs)
    if "residuals" not in captured:
        samples = out[0] if isinstance(out, tuple) else out.controlnet_block_samples
        captured["residuals"] = [s.float().cpu().numpy() for s in samples]
    return out


controlnet.forward = _capture

generator = torch.Generator(device=device).manual_seed(SEED)
image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    control_image=control_image,
    controlnet_conditioning_scale=CONTROL_SCALE,
    width=SIZE,
    height=SIZE,
    num_inference_steps=STEPS,
    true_cfg_scale=GUIDANCE,
    generator=generator,
).images[0]

out = {
    "control_image_rgb8": np.asarray(control_image, dtype=np.uint8),
    "image_rgb8": np.asarray(image, dtype=np.uint8),
}
for i, r in enumerate(captured.get("residuals", [])):
    out[f"residual_{i}"] = r.astype(np.float32)

golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_control_golden.safetensors")
save_file(
    out,
    path_out,
    metadata={
        "seed": str(SEED),
        "steps": str(STEPS),
        "guidance": str(GUIDANCE),
        "control_scale": str(CONTROL_SCALE),
        "width": str(SIZE),
        "height": str(SIZE),
        "prompt": PROMPT,
        "base_repo": BASE,
        "control_repo": CONTROL,
    },
)
print(f"residuals={len(captured.get('residuals', []))} image={np.asarray(image).shape}")
print(f"wrote {path_out} ({len(out)} tensors)")
