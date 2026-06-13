#!/usr/bin/env python
"""Dump an end-to-end golden for the full Lens-Turbo T2I pipeline (mlx-gen sc-3173).

Constructs the authoritative vendor ``LensPipeline`` (SceneWorks ``_vendor/lens``) from the cached
``microsoft/Lens-Turbo`` snapshot and runs one **4-step turbo** generation (guidance 1.0) with an
**injected** initial latent — so the Rust port can feed byte-identical starting noise and the only
divergence is bf16 MLX-vs-torch op-order (the e2e is cross-build, gated on structural similarity, per
the FLUX-hyper / cross-backend precedent).

Production dtypes: encoder + transformer **bf16** (the MXFP4 experts dequantize to dense bf16 on CPU),
VAE **f32** (the shared Flux.2 decoder). Resolution **512×512** keeps the CPU forward tractable
(latent 32×32 = 1024 image tokens) while exercising the whole wiring + the turbo schedule.

Golden contents:
  - ``input_ids``      [1, L] int32 — the positive harmony-rendered ids (the Rust e2e re-tokenizes
                       with ``current_date`` and asserts it reproduces these, validating the tokenizer
                       inside the e2e while guaranteeing the encoder sees identical input);
  - ``init_latents``   [1, h·w, 128] f32 — the injected starting noise;
  - ``final_latents``  [1, h·w, 128] f32 — the torch denoise output (the tightest e2e signal:
                       encoder + DiT + scheduler + norm-rescaled CFG, pre-VAE);
  - ``image``          [1, H, W, 3] f32 in [0,1] — the decoded image (full e2e incl. the VAE shim);
  - metadata: prompt, negative_prompt, current_date, height, width, latent_h, latent_w, num_steps,
              guidance.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_e2e_golden.py
Writes ``tools/golden/lens_e2e_golden.safetensors`` (gitignored real-weights golden).
Needs ~64 GB+ free and the vendor lens package importable (PYTHONPATH=_vendor parent).
"""

from __future__ import annotations

import datetime
import glob
import importlib.util
import os
import sys

import types

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Mxfp4Config
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*"
VENDOR_DIR = os.path.expanduser(
    "~/Repos/SceneWorks/apps/worker/scene_worker/_vendor"
)
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_e2e_golden.safetensors")

# Fixed generation knobs — turbo defaults, a modest square resolution for a tractable CPU forward.
PROMPT = "a red fox sitting in a snowy forest at sunrise, photorealistic"
NEGATIVE = ""
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 4
GUIDANCE = 1.0
SEED = 0


@torch.no_grad()
def _patched_encoder_forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
    """The vendor ``LensGptOssEncoder.forward`` feature path, with **transformers 5.0** mask kwargs
    (``input_embeds`` / ``cache_position``). The vendor file targets 5.8 (``inputs_embeds``); the
    reference venv is 5.0, so calling the vendor forward unchanged raises the kwarg skew. Identical
    math otherwise (capture each selected layer's output, early-exit at the max)."""
    is_lens = (
        input_ids is not None
        and attention_mask is not None
        and hasattr(self, "_lens_selected_layers")
        and not args
        and not kwargs
    )
    if not is_lens:
        return super(type(self), self).forward(input_ids, attention_mask, *args, **kwargs)

    m = self.model
    inputs_embeds = m.embed_tokens(input_ids)
    seq_len = inputs_embeds.shape[1]
    cache_position = torch.arange(seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0).expand_as(input_ids)
    mask_kwargs = {
        "config": m.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    hidden_states = inputs_embeds
    position_embeddings = m.rotary_emb(hidden_states, position_ids)
    index_lookup = {idx: pos for pos, idx in enumerate(self._lens_selected_layers)}
    captured = [None] * len(self._lens_selected_layers)
    for i, decoder_layer in enumerate(m.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=mask_mapping[m.config.layer_types[i]],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if i in index_lookup:
            captured[index_lookup[i]] = hidden_states
        if i == self._lens_max_layer:
            break
    return captured


@torch.no_grad()
def main() -> None:
    snaps = sorted(p for p in glob.glob(SNAP_GLOB) if os.path.isdir(p))
    if not snaps:
        raise SystemExit(f"no Lens-Turbo snapshot at {SNAP_GLOB}")
    snap = snaps[-1]
    print(f"snapshot: {snap}", flush=True)

    # Import the vendor classes (the package, now that einops is installed).
    sys.path.insert(0, VENDOR_DIR)
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from lens import LensPipeline, LensTransformer2DModel
    from lens.text_encoder import LensGptOssEncoder

    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))

    te_cfg = AutoConfig.from_pretrained(os.path.join(snap, "text_encoder"))
    te_cfg._attn_implementation = "eager"
    te_cfg._experts_implementation = "eager"
    print("loading text_encoder (MXFP4 → bf16, CPU)…", flush=True)
    text_encoder = LensGptOssEncoder.from_pretrained(
        os.path.join(snap, "text_encoder"),
        config=te_cfg,
        quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()
    # Patch the feature-path forward for the 5.0 mask-kwarg convention (see _patched_encoder_forward).
    text_encoder.forward = types.MethodType(_patched_encoder_forward, text_encoder)

    print("loading transformer (bf16)…", flush=True)
    transformer = (
        LensTransformer2DModel.from_pretrained(
            os.path.join(snap, "transformer"), torch_dtype=torch.bfloat16
        )
        .to("cpu")
        .eval()
    )

    print("loading vae (f32)…", flush=True)
    vae = (
        AutoencoderKLFlux2.from_pretrained(
            os.path.join(snap, "vae"), torch_dtype=torch.float32
        )
        .to("cpu")
        .eval()
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(snap, "scheduler")
    )

    pipe = LensPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tok,
        transformer=transformer,
    )

    # Injected initial latents [1, h·w, 128] (seeded). prepare_latents returns them as-is (cast to
    # the transformer dtype), so the denoise starts from exactly this noise.
    latent_h, latent_w = HEIGHT // pipe.vae_scale_factor, WIDTH // pipe.vae_scale_factor
    seq_len = latent_h * latent_w
    g = torch.Generator().manual_seed(SEED)
    init = torch.randn((1, seq_len, 128), generator=g, dtype=torch.float32)

    # The exact positive ids the pipeline tokenizes (re-rendered via the same path), for the Rust
    # tokenizer cross-check.
    input_ids, _ = pipe._build_chat_inputs([PROMPT], 512, torch.device("cpu"))
    current_date = datetime.date.today().isoformat()
    print(f"input_ids L={input_ids.shape[1]}  date={current_date}", flush=True)

    print(f"denoising {NUM_STEPS} steps @ {WIDTH}x{HEIGHT}…", flush=True)
    final_latents = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        latents=init.to(transformer.dtype),
        output_type="latent",
    ).images  # [1, seq, 128]

    print("decoding…", flush=True)
    decoded = pipe._decode(final_latents, latent_h, latent_w)  # [1, 3, H, W] in [-1,1]
    image = decoded.clamp(-1.0, 1.0)
    image = (image + 1.0) * 0.5  # → [0,1]
    image = image.permute(0, 2, 3, 1).to(torch.float32).contiguous()  # [1, H, W, 3]

    tensors = {
        "input_ids": input_ids.to(torch.int32).cpu(),
        "init_latents": init.to(torch.float32).cpu(),
        "final_latents": final_latents.to(torch.float32).cpu(),
        "image": image.cpu(),
    }
    meta = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "current_date": current_date,
        "height": str(HEIGHT),
        "width": str(WIDTH),
        "latent_h": str(latent_h),
        "latent_w": str(latent_w),
        "num_steps": str(NUM_STEPS),
        "guidance": str(GUIDANCE),
        "seed": str(SEED),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(
        f"wrote {OUT}\n  final_latents {tuple(final_latents.shape)}  "
        f"image {tuple(image.shape)}  date={current_date}"
    )


if __name__ == "__main__":
    main()
