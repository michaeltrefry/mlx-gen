"""SDXL acceleration-sampler goldens — references for mlx-gen-sdxl sc-2769 (LCM / SDXL-Lightning /
Hyper-SD). Unlike the other SDXL goldens (vendored Apple `mlx_sd`, which has only the ancestral
Euler sampler), these few-step schedulers exist only in **diffusers**, so diffusers is the reference.

Two tiers, selected by argv:

  scheduler-isolation (default, fast, no model) → `sdxl_accel_sched_golden.safetensors`
    For each scheduler (LCMScheduler / EulerDiscreteScheduler(trailing) / TCDScheduler) built on the
    SDXL `scaled_linear` betas, feed FIXED synthetic (model_output, sample) tensors through
    `set_timesteps` + `step` and dump the timesteps, sigmas/init_noise_sigma, and per-step
    DETERMINISTIC output (LCM `denoised` / Euler `prev_sample` / TCD `pred_noised_sample`). The Rust
    `mlx_gen::sampler` port is then bit-exact-ish (~1e-5, torch-f32 vs MLX-f32) to this — validating
    the new scheduler math free of the U-Net-backend confound (the analog of the SDXL per-step gate).
    The between-step re-noise is NOT compared (torch RNG ≠ MLX RNG); only the deterministic core is.

  render <accel_lora_dir> (heavy, loads the full fp16 SDXL pipeline) → `sdxl_accel_render_*.safetensors`
    Deterministic end-to-end renders: dump the (post-`prepare_latents`) initial latent + the final
    RGB8 image for Lightning (4-step, Euler-trailing, CFG 1) and Hyper (4-step, TCD eta=0, CFG 1) —
    both fully deterministic given the prior. The Rust e2e test injects the dumped initial latent and
    compares px>8 (a torch↔MLX backend gap, NOT 0.00%; interpret against the ancestral baseline).

Run from a torch+diffusers venv (NOT the mlx venv):
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_accel_golden.py
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_accel_golden.py render
"""

import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

# SDXL noise-schedule config (scheduler/scheduler_config.json): scaled_linear 0.00085→0.012 / 1000.
SDXL_BETAS = dict(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
)
SHAPE = (1, 4, 16, 16)  # synthetic latent-ish; the scheduler math is shape-agnostic.


def _synthetic(seed: int) -> torch.Tensor:
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.standard_normal(SHAPE, dtype=np.float32))


def dump_scheduler_isolation():
    from diffusers import EulerDiscreteScheduler, LCMScheduler, TCDScheduler

    tensors: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {}

    # alphas_cumprod table (validate the Rust AlphaSchedule once).
    lcm = LCMScheduler(**SDXL_BETAS)
    tensors["alphas_cumprod"] = lcm.alphas_cumprod.numpy().astype(np.float32)

    configs = []

    def run(key: str, sched, num_steps: int, *, is_euler: bool, det_index: int, step_kwargs=None):
        """Drive `num_steps` of `sched` on fresh synthetic tensors; dump the deterministic output.

        det_index: which element of step(return_dict=False) is the deterministic part
                   (1 = LCM `denoised` / TCD `pred_noised`; 0 = Euler `prev_sample`).
        """
        sched.set_timesteps(num_steps)
        ts = sched.timesteps
        tensors[f"{key}.timesteps"] = ts.numpy().astype(np.float32)
        if hasattr(sched, "sigmas"):
            tensors[f"{key}.sigmas"] = sched.sigmas.numpy().astype(np.float32)
        try:
            tensors[f"{key}.init_noise_sigma"] = np.array(
                [float(sched.init_noise_sigma)], dtype=np.float32
            )
        except Exception:
            pass
        for i in range(num_steps):
            t = ts[i]
            eps = _synthetic(1000 + i)
            x = _synthetic(2000 + i)
            tensors[f"{key}.eps{i}"] = eps.numpy().astype(np.float32)
            tensors[f"{key}.x{i}"] = x.numpy().astype(np.float32)
            if is_euler:
                scaled = sched.scale_model_input(x, t)
                tensors[f"{key}.scaled{i}"] = scaled.numpy().astype(np.float32)
            out = sched.step(eps, t, x, return_dict=False, **(step_kwargs or {}))
            tensors[f"{key}.det{i}"] = out[det_index].numpy().astype(np.float32)
        configs.append(key)

    for n in (4, 8):
        run(f"lcm_{n}", LCMScheduler(**SDXL_BETAS), n, is_euler=False, det_index=1)
        run(
            f"lightning_{n}",
            EulerDiscreteScheduler(**SDXL_BETAS, timestep_spacing="trailing"),
            n,
            is_euler=True,
            det_index=0,
        )
        run(
            f"tcd_eta0_{n}",
            TCDScheduler(**SDXL_BETAS),
            n,
            is_euler=False,
            det_index=1,
            step_kwargs={"eta": 0.0},
        )
        run(
            f"tcd_eta03_{n}",
            TCDScheduler(**SDXL_BETAS),
            n,
            is_euler=False,
            det_index=1,
            step_kwargs={"eta": 0.3},
        )

    meta["configs"] = ",".join(configs)
    meta["shape"] = "x".join(str(s) for s in SHAPE)
    out = os.path.join(_GOLDEN_DIR, "sdxl_accel_sched_golden.safetensors")
    save_file(tensors, out, metadata=meta)
    print(f"wrote {out}")
    print(f"  configs: {meta['configs']}")


def _repo_root_for(name: str) -> str:
    repo = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"), f"models--{name.replace('/', '--')}"
    )
    snaps = os.path.join(repo, "snapshots")
    snap = next(os.path.join(snaps, d) for d in os.listdir(snaps))
    return snap


def dump_renders():
    """Deterministic end-to-end renders (Lightning + Hyper eta=0), fp16."""
    import diffusers
    from diffusers import (
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        TCDScheduler,
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prompt = os.environ.get("SDXL_PROMPT", "a red fox in a forest, highly detailed")
    seed = int(os.environ.get("SDXL_SEED", "42"))
    w = int(os.environ.get("SDXL_W", "1024"))
    h = int(os.environ.get("SDXL_H", "1024"))

    lcm_lora = _repo_root_for("latent-consistency/lcm-lora-sdxl")
    light_lora = os.path.join(
        _repo_root_for("ByteDance/SDXL-Lightning"), "sdxl_lightning_4step_lora.safetensors"
    )
    hyper_lora = os.path.join(
        _repo_root_for("ByteDance/Hyper-SD"), "Hyper-SDXL-4steps-lora.safetensors"
    )

    def _is_conv(module_key: str) -> bool:
        # kohya conv module stems in these SDXL LoRAs: resnet conv1/conv2/conv_shortcut, down/up
        # samplers' conv, and conv_in/conv_out. Everything else is a Linear (attention/proj/ff/time).
        return any(
            s in module_key
            for s in ("_conv1", "_conv2", "_conv_shortcut", "_conv.", "_conv_in", "_conv_out")
        ) or module_key.endswith("_conv")

    def render(tag, lora_path, lora_name, scheduler, steps, cfg, *, step_kwargs=None, strip_conv=False):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
        pipe.scheduler = scheduler
        if lora_path is not None:
            if strip_conv:
                # Linear-only fusion: drop the conv-layer LoRA modules (what the Rust SDXL merge does,
                # sc-2639 — Linear-only). Proves the conv drop is the sole accel gap vs full fusion.
                from safetensors.torch import load_file as _load_t

                sd = _load_t(lora_path)
                sd = {k: v for k, v in sd.items() if not _is_conv(k)}
                pipe.load_lora_weights(sd)
            else:
                pipe.load_lora_weights(
                    os.path.dirname(lora_path) if lora_path.endswith(".safetensors") else lora_path,
                    weight_name=os.path.basename(lora_path)
                    if lora_path.endswith(".safetensors")
                    else lora_name,
                )
            pipe.fuse_lora()

        # Capture the post-prepare_latents initial latent (so the Rust port can teacher-force it).
        captured = {}
        orig = pipe.prepare_latents

        def spy(*a, **k):
            lat = orig(*a, **k)
            captured["init"] = lat.detach().to(torch.float32).cpu().numpy()
            return lat

        pipe.prepare_latents = spy

        # Capture the exact CLIP conditioning torch fed the U-Net (CFG off → positive only), so the
        # Rust port can teacher-force it and isolate the CLIP-backend gap from the U-Net-backend gap.
        with torch.no_grad():
            pe, _, ppe, _ = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        captured["prompt_embeds"] = pe.detach().to(torch.float32).cpu().numpy()
        captured["pooled"] = ppe.detach().to(torch.float32).cpu().numpy()

        g = torch.Generator(device=device).manual_seed(seed)
        kw = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=w,
            height=h,
            generator=g,
            output_type="np",
            # Match the vendored Apple `mlx_sd` micro-conditioning convention the Rust SDXL path
            # reproduces verbatim: time_ids = [512,512,0,0,512,512] regardless of the render size
            # (pipeline.rs::text_time_ids). Without this, diffusers passes the real 1024 sizes and the
            # teacher-forced parity gap is dominated by that convention difference, not the sampler.
            original_size=(512, 512),
            target_size=(512, 512),
            crops_coords_top_left=(0, 0),
        )
        kw.update(step_kwargs or {})
        img = pipe(**kw).images[0]
        img_u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
        tensors = {
            "init_latent": captured["init"],  # NCHW f32 (diffusers latent layout)
            "image_u8": img_u8,  # HWC uint8
            "prompt_embeds": captured["prompt_embeds"],  # [1,77,2048] f32 (torch CLIP conditioning)
            "pooled": captured["pooled"],  # [1,1280] f32
        }
        meta = {
            "tag": tag,
            "prompt": prompt,
            "seed": str(seed),
            "steps": str(steps),
            "cfg": str(cfg),
            "w": str(w),
            "h": str(h),
        }
        out = os.path.join(_GOLDEN_DIR, f"sdxl_accel_render_{tag}.safetensors")
        save_file(tensors, out, metadata=meta)
        print(f"wrote {out}  ({tag}: {steps} steps, cfg {cfg})")
        del pipe
        if device == "mps":
            torch.mps.empty_cache()

    print(f"diffusers {diffusers.__version__} on {device}")
    # Deterministic NO-LoRA backend baseline: base SDXL + Euler-trailing, 30 steps, CFG 1 (single
    # forward, matches the captured positive-only conditioning). Establishes the torch↔MLX SDXL U-Net
    # backend floor with NO acceleration LoRA — so the accel variants' teacher-forced gap can be read
    # against it (if accel ≈ base, the few-step samplers add no divergence beyond the backend gap).
    render(
        "base",
        None,
        None,
        EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler", timestep_spacing="trailing"),
        steps=30,
        cfg=1.0,
    )
    # Ancestral baseline (the torch↔MLX gap reference). Note: ancestral is stochastic per-step, so its
    # init-latent teacher-forcing only fixes the prior; the per-step noise still differs torch↔MLX.
    render(
        "ancestral",
        None,
        None,
        EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler"),
        steps=8,
        cfg=7.0,
    )
    render(
        "lightning",
        light_lora,
        None,
        EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler", timestep_spacing="trailing"),
        steps=4,
        cfg=1.0,
    )
    render(
        "hyper",
        hyper_lora,
        None,
        TCDScheduler.from_pretrained(base, subfolder="scheduler"),
        steps=4,
        cfg=1.0,
        step_kwargs={"eta": 0.0},
    )
    render("lcm", lcm_lora, None, _lcm_sched(base), steps=4, cfg=1.0)
    # Linear-only Lightning (conv LoRA stripped) — matches the Rust Linear-only merge, to confirm the
    # accel gap vs full fusion is exactly the dropped conv-layer LoRA (sc-2639 boundary).
    render(
        "lightning_linonly",
        light_lora,
        None,
        EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler", timestep_spacing="trailing"),
        steps=4,
        cfg=1.0,
        strip_conv=True,
    )


def _lcm_sched(base):
    from diffusers import LCMScheduler

    return LCMScheduler.from_pretrained(base, subfolder="scheduler")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        dump_renders()
    else:
        dump_scheduler_isolation()
