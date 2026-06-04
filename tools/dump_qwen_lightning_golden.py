"""Qwen-Image Lightning goldens — references for mlx-gen-qwen-image sc-2909.

Qwen-Image is flow-match, so the few-step Lightning recipe is its own schedule under the
`DiffusionSampler` trait (NOT the DDPM `alphas_cumprod` accel samplers of sc-2769). The frozen mflux
fork has no dedicated Qwen Lightning sampler (it reuses its base `LinearScheduler` + the LoRA), so —
as with the SDXL accel work — the reference is the **official lightx2v recipe as realized in
diffusers**: a static flow-match shift of 3.0 (the model card's `base_shift = max_shift = ln 3`),
`shift_terminal = None`, and `true_cfg_scale = 1.0` (CFG-off, single forward).

Two tiers, selected by argv:

  scheduler-isolation (default, fast, no model) -> `qwen_lightning_sched_golden.safetensors`
    Build the diffusers `FlowMatchEulerDiscreteScheduler` from the official Qwen-Image-Lightning
    config, `set_timesteps` for 4 and 8 steps, and dump the resulting sigmas + a per-step
    DETERMINISTIC Euler `step` on fixed synthetic (velocity, sample) tensors. The Rust
    `FlowMatchSampler::lightning(n)` is then bit-exact-ish (~1e-6, torch-f32 vs MLX-f32) — validating
    the Lightning schedule against diffusers free of the transformer-backend confound.

  render <out_tag> (heavy, loads the full ~40 GB Qwen-Image pipeline in torch) ->
                   `qwen_lightning_render_<tag>.safetensors`
    Deterministic end-to-end Lightning render: official scheduler + the lightx2v Lightning LoRA +
    CFG-off. Dumps the seeded noise, the (CFG-off) prompt embeds, the final packed latents and the
    decoded RGB8 — so the Rust e2e test can teacher-force the noise+embeds and compare px>8 (a
    torch<->MLX backend gap, NOT 0.00%; interpret against the production base render).

Run from the torch+diffusers venv (NOT the mlx venv):
  /Users/michael/Repos/mflux/.venv/bin/python tools/dump_qwen_lightning_golden.py
  /Users/michael/Repos/mflux/.venv/bin/python tools/dump_qwen_lightning_golden.py render 8step
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

# Official lightx2v Qwen-Image-Lightning scheduler config (model card). base_shift == max_shift ==
# ln 3 collapses dynamic shifting to a constant exp(mu) = 3.0; shift_terminal None = no rescale.
LIGHTNING_SCHED = dict(
    base_image_seq_len=256,
    base_shift=math.log(3),
    max_image_seq_len=8192,
    max_shift=math.log(3),
    num_train_timesteps=1000,
    shift=1.0,
    shift_terminal=None,
    use_dynamic_shifting=True,
    time_shift_type="exponential",
)
SHAPE = (1, 1024, 64)  # synthetic packed-latent-ish; the Euler step is shape-agnostic.


def _synthetic(seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    return g.standard_normal(SHAPE, dtype=np.float32)


def dump_scheduler_isolation():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from safetensors.numpy import save_file

    tensors: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {}
    configs = []
    # base==max ⇒ mu is constant ln 3 regardless of image_seq_len (diffusers `calculate_shift`).
    mu = math.log(3)

    for n in (4, 8):
        sched = FlowMatchEulerDiscreteScheduler(**LIGHTNING_SCHED)
        sched.set_timesteps(num_inference_steps=n, mu=mu)
        key = f"lightning_{n}"
        tensors[f"{key}.sigmas"] = sched.sigmas.numpy().astype(np.float32)
        tensors[f"{key}.timesteps"] = sched.timesteps.numpy().astype(np.float32)
        # Per-step deterministic Euler update on fixed synthetic tensors.
        for i in range(n):
            t = sched.timesteps[i]
            v = _synthetic(1000 + i)
            x = _synthetic(2000 + i)
            tensors[f"{key}.v{i}"] = v
            tensors[f"{key}.x{i}"] = x
            out = sched.step(torch.from_numpy(v), t, torch.from_numpy(x), return_dict=False)[0]
            tensors[f"{key}.det{i}"] = out.numpy().astype(np.float32)
        configs.append(key)

    meta["configs"] = ",".join(configs)
    meta["shift"] = "3.0"
    meta["shape"] = "x".join(str(s) for s in SHAPE)
    out = os.path.join(_GOLDEN_DIR, "qwen_lightning_sched_golden.safetensors")
    save_file(tensors, out, metadata=meta)
    print(f"wrote {out}")
    print(f"  configs: {meta['configs']}")
    for n in (4, 8):
        print(f"  lightning_{n} sigmas: {tensors[f'lightning_{n}.sigmas'].tolist()}")


def _lightning_lora() -> str:
    """The cached lightx2v Qwen-Image-Lightning 8-step LoRA file (diffusers/bf16)."""
    repo = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        "models--lightx2v--Qwen-Image-Lightning",
    )
    snaps = os.path.join(repo, "snapshots")
    for d in os.listdir(snaps):
        p = os.path.join(snaps, d, "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("cached lightx2v/Qwen-Image-Lightning 8-step LoRA not found")


def dump_render(tag: str):
    import diffusers
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
    from safetensors.numpy import save_file

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prompt = os.environ.get("QWEN_PROMPT", "a fox sitting in a forest, photorealistic")
    seed = int(os.environ.get("QWEN_SEED", "42"))
    steps = int(os.environ.get("QWEN_STEPS", "8"))
    w = int(os.environ.get("QWEN_W", "512"))
    h = int(os.environ.get("QWEN_H", "512"))

    print(f"diffusers {diffusers.__version__} on {device}: loading Qwen-Image (bf16)...")
    pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to(device)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {**pipe.scheduler.config, **LIGHTNING_SCHED}
    )
    pipe.load_lora_weights(_lightning_lora())
    pipe.fuse_lora()

    captured = {}
    orig_prepare = pipe.prepare_latents

    def spy_prepare(*a, **k):
        out = orig_prepare(*a, **k)
        lat = out[0] if isinstance(out, tuple) else out
        captured["init"] = lat.detach().to(torch.float32).cpu().numpy()
        return out

    pipe.prepare_latents = spy_prepare

    g = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=" ",
        true_cfg_scale=1.0,  # CFG-off (single forward), the Lightning fast path.
        num_inference_steps=steps,
        width=w,
        height=h,
        generator=g,
        output_type="np",
    )
    img = result.images[0]
    img_u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)

    tensors = {
        "init_latent": captured.get("init", np.zeros((1, 1), np.float32)),
        "image_u8": img_u8,
    }
    meta = dict(tag=tag, prompt=prompt, seed=str(seed), steps=str(steps), w=str(w), h=str(h),
                cfg="1.0", shift="3.0")
    out = os.path.join(_GOLDEN_DIR, f"qwen_lightning_render_{tag}.safetensors")
    save_file(tensors, out, metadata=meta)
    print(f"wrote {out}  ({tag}: {steps} steps, CFG-off, {w}x{h})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        dump_render(sys.argv[2] if len(sys.argv) > 2 else "8step")
    else:
        dump_scheduler_isolation()
