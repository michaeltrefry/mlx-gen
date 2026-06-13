#!/usr/bin/env python
"""Dump an end-to-end + per-block golden for the Lens DiT (mlx-gen sc-3168).

Runs the authoritative vendor `LensTransformer2DModel` (SceneWorks `_vendor/lens/transformer.py`) on
the cached `microsoft/Lens-Turbo` transformer weights, in **float32** (a tight, decisive correctness
gate for a 48-block DiT — bf16 cross-backend accumulation over 48 residual blocks would obscure
subtle bugs), over synthetic inputs. Records the full-forward output plus the block-0 inputs and
output so the Rust port can be checked both per-block and end-to-end.

Text features are synthetic (seeded) random tensors — the story stands alone, independent of the
gpt-oss encoder slices. The Rust side loads the same real transformer weights (cast to f32) directly
from the snapshot via `Weights::from_dir`, so only the activations live in the golden.

Golden contents:
  - `hidden_states` [1, img_len, 128], `feat_{0..3}` [1, txt_len, 2880], `timestep` [1];
  - `img_in_out` [1, img_len, 1536], `txt_in_out` [1, txt_len, 1536], `temb` [1, 1536]
    (block-0 inputs, captured by replaying the model's sub-modules);
  - `block0_enc` / `block0_hidden` [1, *, 1536] (block-0 outputs);
  - `out` [1, img_len, 128] (full forward);
  - metadata: frame, h_lat, w_lat, txt_len, img_len.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_dit_golden.py
Writes `tools/golden/lens_dit_golden.safetensors` (gitignored real-weights golden).
"""

from __future__ import annotations

import glob
import importlib.util
import os

import torch
from safetensors.torch import save_file

HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/transformer"
VENDOR_T = os.path.expanduser(
    "~/Repos/SceneWorks/apps/worker/scene_worker/_vendor/lens/transformer.py"
)
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_dit_golden.safetensors")

FRAME, H_LAT, W_LAT = 1, 16, 16
TXT_LEN = 120


def load_model_cls():
    spec = importlib.util.spec_from_file_location("lens_transformer", VENDOR_T)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensTransformer2DModel


def main() -> None:
    matches = sorted(glob.glob(SNAP_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo transformer snapshot at {SNAP_GLOB}")
    tdir = matches[-1]

    LensTransformer2DModel = load_model_cls()
    print("loading transformer (f32, CPU)…", flush=True)
    model = (
        LensTransformer2DModel.from_pretrained(tdir, torch_dtype=torch.float32)
        .to("cpu")
        .eval()
    )

    img_len = FRAME * H_LAT * W_LAT
    n_text = len(model.config.selected_layer_index)
    enc_dim = model.config.enc_hidden_dim

    torch.manual_seed(0)
    hidden_states = torch.randn(1, img_len, model.config.in_channels, dtype=torch.float32)
    feats = [torch.randn(1, TXT_LEN, enc_dim, dtype=torch.float32) for _ in range(n_text)]
    timestep = torch.rand(1, dtype=torch.float32)  # in [0, 1]
    text_mask = torch.ones(1, TXT_LEN, dtype=torch.bool)
    img_shapes = [(FRAME, H_LAT, W_LAT)]

    with torch.no_grad():
        # --- replay the model sub-modules to capture block-0 inputs ---
        img_in_out = model.img_in(hidden_states)
        normed = [model.txt_norm[i](feats[i]) for i in range(n_text)]
        txt_in_out = model.txt_in(torch.cat(normed, dim=-1))
        temb = model.time_text_embed(timestep, img_in_out)
        rope = model.pos_embed(img_shapes, [TXT_LEN], device=torch.device("cpu"))
        mask = model._build_joint_attention_mask(text_mask, img_len)
        block0_enc, block0_hidden = model.transformer_blocks[0](
            img_in_out, txt_in_out, temb, rope, mask
        )

        # --- full forward ---
        out = model(hidden_states, feats, text_mask, timestep, img_shapes)

    tensors = {
        "hidden_states": hidden_states.contiguous(),
        "timestep": timestep.contiguous(),
        "img_in_out": img_in_out.contiguous(),
        "txt_in_out": txt_in_out.contiguous(),
        "temb": temb.contiguous(),
        "block0_enc": block0_enc.contiguous(),
        "block0_hidden": block0_hidden.contiguous(),
        "out": out.contiguous(),
    }
    for i, f in enumerate(feats):
        tensors[f"feat_{i}"] = f.contiguous()

    meta = {
        "frame": str(FRAME),
        "h_lat": str(H_LAT),
        "w_lat": str(W_LAT),
        "txt_len": str(TXT_LEN),
        "img_len": str(img_len),
        "n_text": str(n_text),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(
        f"wrote {OUT}  (img_len={img_len}, txt_len={TXT_LEN}, "
        f"out={tuple(out.shape)}, block0_hidden={tuple(block0_hidden.shape)})"
    )


if __name__ == "__main__":
    main()
