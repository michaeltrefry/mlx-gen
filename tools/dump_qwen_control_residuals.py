"""Dump a **residual-isolation** parity golden for the Qwen-Image ControlNet port (epic 3401 / sc-3574).

Loads ONLY the diffusers `QwenImageControlNetModel` (InstantX `Qwen-Image-ControlNet-Union`, ~1.6 GB
— NOT the full 40 GB pipeline), feeds it deterministic seeded inputs (packed noise latents, a packed
control latent, prompt embeds, a timestep), and dumps both the inputs and the per-block
`controlnet_block_samples` (pre-scale, conditioning_scale=1). The Rust test
`control_real_weights.rs::residuals_match_diffusers` reads the SAME inputs and compares its
`QwenControlNet::forward` residuals — isolating the control branch from all pipeline plumbing
(noise/scheduler/prompt-template/CFG), which were matched against the mflux fork, not raw diffusers.

This is a **cross-backend** reference (mlx mixed-precision f32-latents vs torch bf16), so the gate is
peak-relative error at the established Qwen floor, not bit-exactness.

Run (diffusers >= 0.35 + the InstantX checkpoint in the HF cache):
    ~/Repos/mflux/.venv/bin/python tools/dump_qwen_control_residuals.py
Output (gitignored): tools/golden/qwen_control_residuals.safetensors
"""

import os

import numpy as np
import torch
from diffusers import QwenImageControlNetModel
from safetensors.numpy import save_file

CONTROL = os.environ.get("QWEN_CONTROL_REPO", "InstantX/Qwen-Image-ControlNet-Union")
SIZE = int(os.environ.get("SIZE", "512"))  # square; multiple of 16
TXT = int(os.environ.get("TXT", "64"))
SIGMA = float(os.environ.get("SIGMA", "0.7"))
SEED = int(os.environ.get("SEED", "1234"))

LH = LW = SIZE // 16
SEQ = LH * LW
JOINT_DIM = 3584

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16

rng = np.random.default_rng(SEED)
# f32 reference inputs (shared verbatim with Rust): packed noise latents, packed control latent,
# prompt embeds. The real inference dtype flow is mixed (Rust keeps latents f32, torch runs bf16),
# so we feed bf16 to torch here and dump the f32 originals — the residual delta is exactly that floor.
hidden_f32 = rng.standard_normal((1, SEQ, 64), dtype=np.float32)
control_f32 = rng.standard_normal((1, SEQ, 64), dtype=np.float32)
# Embeds: both sides use bf16 — dump the bf16-rounded values (as f32) so Rust casts to the same bf16.
embeds_bf16_as_f32 = (
    torch.from_numpy(rng.standard_normal((1, TXT, JOINT_DIM), dtype=np.float32))
    .to(torch.bfloat16)
    .to(torch.float32)
    .numpy()
)

model = QwenImageControlNetModel.from_pretrained(CONTROL, torch_dtype=dtype).to(device).eval()

hidden = torch.from_numpy(hidden_f32).to(device, dtype)
control = torch.from_numpy(control_f32).to(device, dtype)
embeds = torch.from_numpy(embeds_bf16_as_f32).to(device, dtype)
timestep = torch.tensor([SIGMA], device=device, dtype=dtype)
img_shapes = [(1, LH, LW)]
txt_seq_lens = [TXT]

with torch.no_grad():
    # return_dict=False → the list of per-block `controlnet_block_samples` directly (one [1, seq, dim]
    # tensor per control layer); do NOT index [0] (that would take only the first residual).
    samples = model(
        hidden_states=hidden,
        controlnet_cond=control,
        conditioning_scale=1.0,
        encoder_hidden_states=embeds,
        encoder_hidden_states_mask=None,
        timestep=timestep,
        img_shapes=img_shapes,
        return_dict=False,
    )

out = {
    "hidden_states": hidden_f32,
    "controlnet_cond": control_f32,
    "encoder_hidden_states": embeds_bf16_as_f32,
}
for i, s in enumerate(samples):
    out[f"residual_{i}"] = s.float().cpu().numpy().astype(np.float32)

golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_control_residuals.safetensors")
save_file(
    out,
    path_out,
    metadata={
        "seed": str(SEED),
        "size": str(SIZE),
        "txt": str(TXT),
        "sigma": str(SIGMA),
        "lh": str(LH),
        "lw": str(LW),
        "num_residuals": str(len(samples)),
        "control_repo": CONTROL,
    },
)
print(f"residuals={len(samples)} each {tuple(samples[0].shape)} | seq={SEQ} txt={TXT}")
print(f"wrote {path_out} ({len(out)} tensors)")
