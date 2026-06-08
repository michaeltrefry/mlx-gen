"""sc-3624 torch A/B reference: diffusers FluxPipeline + XLabs IP-Adapter render.

Reproduces the SceneWorks torch `FluxDiffusersAdapter` path (`pipe.load_ip_adapter(
"XLabs-AI/flux-ip-adapter", image_encoder=openai/clip-vit-large-patch14)` + `true_cfg_scale`) so
the Rust/MLX IP-Adapter render can be compared against it. Generates a reference image (plain
FLUX-dev) then a FLUX+XLabs-IP render conditioned on it. Run with the torch venv:

    HF_HUB_OFFLINE=1 ~/mlx-flux-venv/bin/python tools/flux_ip_torch_ab.py /tmp/flux_ab

Writes <out>/reference.png + <out>/torch_ip.png.
"""
import os
import sys

import torch
from diffusers import FluxPipeline

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flux_ab"
os.makedirs(OUT, exist_ok=True)

DEVICE = "mps"
DTYPE = torch.bfloat16
SIZE = 512
STEPS = 16
SCALE = 0.7
TRUE_CFG = 4.0
REF_PROMPT = "a studio portrait photo of a fluffy orange tabby cat, sharp focus, plain background"
IP_PROMPT = "an oil painting in the bold swirling brushstroke style of Van Gogh"
NEG = os.environ.get("IP_NEG", "")  # match the MLX run for a fair A/B

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE)
pipe.to(DEVICE)
print("[torch] FLUX.1-dev loaded", flush=True)

# 1) Reference image (plain FLUX, no IP). Reuse an existing reference.png (seed-fixed → identical).
from PIL import Image as _PILImage  # noqa: E402

ref_path = os.path.join(OUT, "reference.png")
if os.path.exists(ref_path):
    ref = _PILImage.open(ref_path).convert("RGB")
    print("[torch] reusing reference.png", flush=True)
else:
    g = torch.Generator(device="cpu").manual_seed(1)
    ref = pipe(
        REF_PROMPT, num_inference_steps=STEPS, guidance_scale=3.5,
        height=SIZE, width=SIZE, generator=g,
    ).images[0]
    ref.save(ref_path)
    print("[torch] wrote reference.png", flush=True)

# 2) FLUX + XLabs IP-Adapter, conditioned on the reference (the parity target).
pipe.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
)
pipe.set_ip_adapter_scale(SCALE)
g = torch.Generator(device="cpu").manual_seed(2)
ip = pipe(
    IP_PROMPT, ip_adapter_image=ref, num_inference_steps=STEPS, guidance_scale=3.5,
    true_cfg_scale=TRUE_CFG, negative_prompt=NEG, height=SIZE, width=SIZE, generator=g,
).images[0]
ip.save(os.path.join(OUT, "torch_ip.png"))
print("[torch] wrote torch_ip.png", flush=True)
print(f"[torch] done: prompt={IP_PROMPT!r} scale={SCALE} true_cfg={TRUE_CFG}", flush=True)
