#!/usr/bin/env python
"""Golden dump for the SDXL ControlNet branch (sc-3058).

Runs the diffusers `ControlNetModel` (xinsir tile-CN) in **float32** on a fixed input and saves the
9 down-block residuals + the mid residual (NCHW→NHWC), so the Rust port can be validated bit-exact,
isolated from the schedule (a fixed timestep is fed to both sides; mlx-gen's sinusoidal timestep
embedding equals diffusers `get_timestep_embedding`).

Run from the mflux venv:
    ~/repos/mflux/.venv/bin/python ~/Repos/mlx-gen/tools/dump_sdxl_controlnet_golden.py
"""
from pathlib import Path

import torch
from diffusers import ControlNetModel
from safetensors.torch import save_file

CN_DIR = next(
    (Path.home() / ".cache/huggingface/hub/models--xinsir--controlnet-tile-sdxl-1.0/snapshots").iterdir()
)
OUT = Path(__file__).resolve().parent / "golden" / "sdxl_controlnet_golden.safetensors"

# Fixed inputs (seeded). Latent 64×64 ⇒ a 512×512 control image.
TIMESTEP = 500.0
CONDITIONING_SCALE = 1.0


def main():
    torch.manual_seed(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    cn = ControlNetModel.from_pretrained(CN_DIR, torch_dtype=torch.float32)
    cn.eval()

    sample = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    control = torch.rand(1, 3, 512, 512, dtype=torch.float32)  # [0,1], like the CN image processor
    encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float32)
    text_embeds = torch.randn(1, 1280, dtype=torch.float32)  # pooled
    time_ids = torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], dtype=torch.float32)

    with torch.no_grad():
        down, mid = cn(
            sample,
            TIMESTEP,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control,
            conditioning_scale=CONDITIONING_SCALE,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )

    def nhwc(t):  # [1, C, h, w] -> [1, h, w, C]
        return t.permute(0, 2, 3, 1).contiguous()

    out = {
        "sample": sample.contiguous(),  # NCHW (the Rust test transposes)
        "control": control.contiguous(),  # NCHW
        "encoder_hidden_states": encoder_hidden_states.contiguous(),
        "text_embeds": text_embeds.contiguous(),
        "time_ids": time_ids.contiguous(),
        "timestep": torch.tensor([TIMESTEP]),
        "mid": nhwc(mid),
    }
    for i, d in enumerate(down):
        out[f"down_{i}"] = nhwc(d)

    save_file(out, str(OUT))
    print(f"wrote {OUT}")
    print(f"  {len(down)} down residuals, shapes {[tuple(d.shape) for d in down]}")
    print(f"  mid {tuple(mid.shape)}  mean={mid.mean():.5f} std={mid.std():.5f}")


if __name__ == "__main__":
    main()
