"""Bisection capture for the Wan-VACE structural parity (epic 3040 / sc-3388, S1).

Rebuilds the SAME randomly-initialized small-config diffusers `WanVACETransformer3DModel` as
`dump_wanvace_transformer_golden.py` (seed 3388, identical cfg) and captures intermediate activations
via forward hooks so the Rust port can be byte-bisected stage by stage (patch-embed → control
patch-embed → condition embed → vace block 0 → main block 0). Writes a NON-committed fixture used only
during development of the S1 parity gate.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_wanvace_bisect.py
Writes `mlx-gen-wan/tests/fixtures/wanvace_bisect.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel

from _paths import fixture

torch.manual_seed(3388)

NUM_HEADS = 4
HEAD_DIM = 16
cfg = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=NUM_HEADS,
    attention_head_dim=HEAD_DIM,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=64,
    ffn_dim=128,
    num_layers=4,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    image_dim=None,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
    pos_embed_seq_len=None,
    vace_layers=[0, 2],
    vace_in_channels=96,
)
model = WanVACETransformer3DModel(**cfg).to(torch.float32).eval()

T, H, W = 4, 8, 8
hidden_states = torch.randn(1, 16, T, H, W)
control_hidden_states = torch.randn(1, 96, T, H, W)
timestep = torch.tensor([3.0])
encoder_hidden_states = torch.randn(1, 12, cfg["text_dim"])
control_scale = torch.tensor([1.0, 0.5])

cap: dict[str, torch.Tensor] = {}


def patch_hook(mod, inp, out):  # Conv3d → [1, dim, T', H', W'] → flatten(2).transpose(1,2)
    cap["x_tokens"] = out.flatten(2).transpose(1, 2).contiguous()


def vace_patch_hook(mod, inp, out):
    cap["control_emb"] = out.flatten(2).transpose(1, 2).contiguous()


def cond_hook(mod, inp, out):
    temb, timestep_proj, ehs, ehs_img = out
    cap["temb"] = temb.contiguous()
    cap["timestep_proj"] = timestep_proj.contiguous()
    cap["text_emb"] = ehs.contiguous()


def vace0_hook(mod, inp, out):
    cond, control = out
    cap["vace0_hint"] = cond.contiguous()
    cap["vace0_control"] = control.contiguous()


def block0_hook(mod, inp, out):
    cap["block0_out"] = out.contiguous()


model.patch_embedding.register_forward_hook(patch_hook)
model.vace_patch_embedding.register_forward_hook(vace_patch_hook)
model.condition_embedder.register_forward_hook(cond_hook)
model.vace_blocks[0].register_forward_hook(vace0_hook)
model.blocks[0].register_forward_hook(block0_hook)

with torch.no_grad():
    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        control_hidden_states=control_hidden_states,
        control_hidden_states_scale=control_scale,
        return_dict=False,
    )[0]

cap["output"] = out.contiguous()
out_path = fixture("mlx-gen-wan/tests/fixtures/wanvace_bisect.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(cap, out_path)
print(f"wrote {out_path}")
for k, v in cap.items():
    print(f"  {k:16s} {tuple(v.shape)}  mean={float(v.mean()):+.5f} std={float(v.std()):.5f}")
