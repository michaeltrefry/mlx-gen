"""Parity fixtures for the Z-Image DiT sub-modules (sc-2344, denoiser PR 1):
TimestepEmbedder, RopeEmbedder, FinalLayer, ZImageContextBlock.

Small dims for the weighted modules (dim=96, heads=4 → head_dim=24, half=12); the
weightless RopeEmbedder uses the REAL config (axes_dims=[32,48,48], theta=256). Norm
weights / biases are randomized so every weight path is exercised. fp32.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_z_submodules.py
"""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_transformer.timestep_embedder import TimestepEmbedder
from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder
from mflux.models.z_image.model.z_image_transformer.final_layer import FinalLayer
from mflux.models.z_image.model.z_image_transformer.context_block import ZImageContextBlock

mx.random.seed(0)
rng = np.random.default_rng(0)
DIM, N_HEADS, SEQ = 96, 4, 4
HEAD_DIM = DIM // N_HEADS  # 24
HALF = HEAD_DIM // 2       # 12
OUT_CH = 16
out = {}


def add(prefix, module):
    for k, v in tree_flatten(module.parameters()):
        out[f"{prefix}.{k}"] = v.astype(mx.float32)


# --- TimestepEmbedder ---
te = TimestepEmbedder(out_size=min(DIM, 256), mid_size=1024)
add("te.w", te)
t_in = mx.random.normal((SEQ,)) * 10.0
out["te.in_t"] = t_in.astype(mx.float32)
out["te.out"] = te(t_in).astype(mx.float32)

# --- RopeEmbedder (weightless; REAL config) ---
AXES_DIMS, AXES_LENS, THETA = [32, 48, 48], [1024, 512, 512], 256.0
rope = RopeEmbedder(theta=THETA, axes_dims=AXES_DIMS, axes_lens=AXES_LENS)
N = 7
ids = np.stack([
    rng.integers(0, AXES_LENS[0], N),
    rng.integers(0, AXES_LENS[1], N),
    rng.integers(0, AXES_LENS[2], N),
], axis=1).astype(np.int32)
ids_mx = mx.array(ids)
out["rope.ids"] = ids_mx
out["rope.out"] = rope(ids_mx).astype(mx.float32)  # (N, sum(axes_dims)/2, 2) = (7, 64, 2)

# --- FinalLayer ---
fl = FinalLayer(hidden_size=DIM, out_channels=OUT_CH)
fl.adaLN_modulation[0].bias = 0.05 * mx.random.normal(fl.adaLN_modulation[0].bias.shape)
add("fl.w", fl)
fl_x = mx.random.normal((1, SEQ, DIM))
fl_c = mx.random.normal((1, min(DIM, 256)))
out["fl.in_x"] = fl_x.astype(mx.float32)
out["fl.in_c"] = fl_c.astype(mx.float32)
out["fl.out"] = fl(fl_x, fl_c).astype(mx.float32)

# --- ZImageContextBlock ---
cb = ZImageContextBlock(dim=DIM, n_heads=N_HEADS, norm_eps=1e-5, qk_norm=True)
for nrm in (cb.attention_norm1, cb.attention_norm2, cb.ffn_norm1, cb.ffn_norm2):
    nrm.weight = 1.0 + 0.1 * mx.random.normal(nrm.weight.shape)
cb.attention.norm_q.weight = 1.0 + 0.1 * mx.random.normal((HEAD_DIM,))
cb.attention.norm_k.weight = 1.0 + 0.1 * mx.random.normal((HEAD_DIM,))
add("cb.w", cb)
cb_x = mx.random.normal((1, SEQ, DIM))
cb_fc = mx.random.normal((SEQ, HALF, 2))
out["cb.in_x"] = cb_x.astype(mx.float32)
out["cb.in_freqs_cis"] = cb_fc.astype(mx.float32)
out["cb.out"] = cb(cb_x, None, cb_fc).astype(mx.float32)

path = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/z_submodules.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
for k in ("te.out", "rope.out", "fl.out", "cb.out"):
    print(f"  {k}: {out[k].shape}")
