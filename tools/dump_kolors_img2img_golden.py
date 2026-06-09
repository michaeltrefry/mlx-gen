"""Kolors img2img end-to-end golden — reference for the img2img pipeline parity (sc-3095).

Drives the diffusers Kolors components **directly** (not `pipe(...)`) so the init encode and the
add_noise noise are fully controlled and reproducible by the Rust side:

 - the init image is a fixed, deterministic 512² RGB pattern (no resize on either side);
 - the init latents are the VAE encoder **mean** (`sample_mode="argmax"` ⇒ mode == mean), matching the
   Rust `encode_init_latents` (and the production fork convention) rather than diffusers' default
   `sample` (which draws non-reproducible RNG);
 - the add_noise `noise` is a FIXED unit-normal tensor (dumped, fed to the Rust side verbatim);
 - the schedule is sliced at the strength-derived start exactly as `get_timesteps` + `set_begin_index`,
   and `add_noise` is the raw `x₀ + noise·σ_start` (EulerDiscrete at `begin_index`).

This mirrors `KolorsEulerSampler::kolors_img2img` + `Kolors::denoise_img2img_latents` op-for-op. Like
the T2I golden, the full-trajectory pixel delta vs this torch reference is cross-backend-chaos-limited
(see the t2i parity note), so the gate is the early-step latent integration + render coherence.

Dumps: `init_image` (f32 [H,W,3] in [0,1], exact uint8 values), `init_latents` (scaled VAE mean,
NHWC), `noise` (NHWC), `pos/neg_context`+`pos/neg_pooled`, `step0/step1_latents`, `final_latents`,
`image`.

Loads the full pipeline f32 (~35 GB). Run backgrounded:
    ~/repos/mflux/.venv-0312/bin/python tools/dump_kolors_img2img_golden.py
Output (gitignored): tools/golden/kolors_img2img_golden.safetensors
"""

import glob
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image as PILImage

from _paths import fixture, hf_hub_cache

from diffusers import KolorsImg2ImgPipeline
from diffusers.pipelines.kolors.pipeline_kolors_img2img import retrieve_latents

PROMPT = "A cat playing a grand piano on a city rooftop at sunset."
NEGATIVE = "blurry, low quality"
STEPS = 8
CFG = 5.0
STRENGTH = 0.6  # > 1/STEPS so the slice keeps several effective steps (int(8·0.6)=4)
H = W = 512


def snapshot() -> Path:
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-diffusers" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-diffusers snapshot not found in HF cache")
    return Path(snaps[-1])


def nhwc(t):  # [B,C,H,W] → [B,H,W,C]
    return mx.array(t.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32))


def arr(t):
    return mx.array(t.detach().cpu().numpy().astype(np.float32))


def make_init_pixels() -> np.ndarray:
    """A deterministic 512² RGB pattern (uint8): smooth diagonal gradients + a colour block."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = (xx * 255 // (W - 1)).astype(np.uint8)
    g = (yy * 255 // (H - 1)).astype(np.uint8)
    b = (((xx + yy) * 255) // (W + H - 2)).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img[128:384, 128:384] = np.array([200, 60, 120], dtype=np.uint8)  # a flat block
    return img  # [H,W,3] uint8


@torch.no_grad()
def main():
    snap = snapshot()
    pipe = KolorsImg2ImgPipeline.from_pretrained(snap, variant="fp16", torch_dtype=torch.float32)
    pipe.to("cpu")
    device = torch.device("cpu")

    pixels = make_init_pixels()
    pil = PILImage.fromarray(pixels, mode="RGB")

    # Conditioning (same encode_prompt as T2I).
    pos_embeds, neg_embeds, pos_pooled, neg_pooled = pipe.encode_prompt(
        prompt=PROMPT,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=NEGATIVE,
    )

    # --- init latents: VAE encoder MEAN (mode), scaled. Matches Rust encode_init_latents. ---
    image = pipe.image_processor.preprocess(pil, height=H, width=W)  # [1,3,H,W] in [-1,1]
    image = image.to(device=device, dtype=torch.float32)
    init_latents = retrieve_latents(pipe.vae.encode(image), sample_mode="argmax")
    init_latents = pipe.vae.config.scaling_factor * init_latents  # [1,4,64,64]

    # --- schedule slice (get_timesteps + set_begin_index) ---
    pipe.scheduler.set_timesteps(STEPS, device=device)
    timesteps, eff_steps = pipe.get_timesteps(STEPS, STRENGTH, device)
    print("timesteps:", [float(t) for t in timesteps], "eff_steps:", int(eff_steps))
    latent_timestep = timesteps[:1]

    # --- fixed add_noise noise (NCHW), raw add_noise at begin_index ---
    g = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(init_latents.shape, generator=g, dtype=torch.float32)
    latents = pipe.scheduler.add_noise(init_latents, noise, latent_timestep)
    print("start_sigma:", float(pipe.scheduler.sigmas[pipe.scheduler.begin_index]))

    # --- micro-conditioning (SDXL _get_add_time_ids), CFG batch [pos, neg] (Rust convention) ---
    add_time_ids = pipe._get_add_time_ids(
        (H, W), (0, 0), (H, W), dtype=pos_embeds.dtype,
        text_encoder_projection_dim=pos_pooled.shape[-1],
    )
    cond = torch.cat([pos_embeds, neg_embeds], dim=0)
    pooled = torch.cat([pos_pooled, neg_pooled], dim=0)
    time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    step_latents = []
    for i, t in enumerate(timesteps):
        x_in = pipe.scheduler.scale_model_input(latents, t)
        x_unet = torch.cat([x_in, x_in], dim=0)
        eps = pipe.unet(
            x_unet, t,
            encoder_hidden_states=cond,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            return_dict=False,
        )[0]
        eps_text, eps_neg = eps.chunk(2)
        eps = eps_neg + CFG * (eps_text - eps_neg)
        latents = pipe.scheduler.step(eps, t, latents, return_dict=False)[0]
        step_latents.append(latents.detach().clone())

    final_latents = latents
    img = pipe.vae.decode(final_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)

    tensors = {
        "init_image": mx.array((pixels.astype(np.float32) / 255.0)),  # [H,W,3] in [0,1]
        "init_latents": nhwc(init_latents),
        "noise": nhwc(noise),
        "pos_context": arr(pos_embeds),
        "pos_pooled": arr(pos_pooled),
        "neg_context": arr(neg_embeds),
        "neg_pooled": arr(neg_pooled),
        "step0_latents": nhwc(step_latents[0]),
        "step1_latents": nhwc(step_latents[1]),
        "final_latents": nhwc(final_latents),
        "image": nhwc(img),
    }
    mx.eval(list(tensors.values()))
    meta = {
        "prompt": PROMPT,
        "negative": NEGATIVE,
        "steps": str(STEPS),
        "strength": str(STRENGTH),
        "cfg": str(CFG),
        "h": str(H),
        "w": str(W),
        "eff_steps": str(int(eff_steps)),
    }
    out_path = fixture("tools/golden/kolors_img2img_golden.safetensors")
    mx.save_safetensors(out_path, tensors, metadata=meta)
    print(f"wrote {out_path}")
    print(f"  init_latents {tuple(tensors['init_latents'].shape)} final_latents "
          f"{tuple(tensors['final_latents'].shape)} image {tuple(tensors['image'].shape)}")


if __name__ == "__main__":
    main()
