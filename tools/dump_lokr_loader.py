"""Generate the LoKr-loader parity fixture for the Rust port (sc-2343).

Validates the whole adapter-loading path end-to-end: build a small Z-Image block, write a
synthetic LoKr adapter in SceneWorks' on-disk format (bare module-path keys
`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`, + `networkType=lokr` / `alpha` /
`rank` safetensors metadata), apply it through the fork's REAL `LoKrLoader`, and dump the
post-adapter block output. The Rust test loads the same block + the same adapter file via
the crate's loader, installs onto the block, and must reproduce the output.

Two factor forms are exercised: attention.to_q uses full w1/w2; feed_forward.w1 uses a
low-rank w2 (w2_a @ w2_b).

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_lokr_loader.py
"""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock
from mflux.models.common.lora.mapping.lokr_loader import LoKrLoader

mx.random.seed(0)
rng = np.random.default_rng(0)
DIM, N_HEADS, SEQ = 96, 4, 4
HEAD_DIM = DIM // N_HEADS
SCALE = 0.7
ALPHA, RANK = 8.0, 4

block = ZImageTransformerBlock(dim=DIM, n_heads=N_HEADS, norm_eps=1e-5, qk_norm=True)
for nrm in (block.attention_norm1, block.attention_norm2, block.ffn_norm1, block.ffn_norm2):
    nrm.weight = 1.0 + 0.1 * mx.random.normal(nrm.weight.shape)
block.attention.norm_q.weight = 1.0 + 0.1 * mx.random.normal((HEAD_DIM,))
block.attention.norm_k.weight = 1.0 + 0.1 * mx.random.normal((HEAD_DIM,))
block.adaLN_modulation[0].bias = 0.05 * mx.random.normal(block.adaLN_modulation[0].bias.shape)

x = mx.random.normal((1, SEQ, DIM))
t_emb = mx.random.normal((1, min(DIM, 256)))
freqs_cis = mx.random.normal((SEQ, HEAD_DIM // 2, 2))

# Base block weights + inputs (Rust loads the block from here).
base = {f"w.{k}": v.astype(mx.float32) for k, v in tree_flatten(block.parameters())}
base["in.x"] = x.astype(mx.float32)
base["in.t_emb"] = t_emb.astype(mx.float32)
base["in.freqs_cis"] = freqs_cis.astype(mx.float32)


def f32(*shape):
    return mx.array((0.1 * rng.standard_normal(shape)).astype(np.float32))


# Synthetic LoKr factors (kron(w1, w2) reshapes to each base [out, in]):
#   attention.to_q  [96, 96]  -> w1 [8,8] (full),  w2 [12,12] (full)
#   feed_forward.w1 [256, 96] -> w1 [16,8] (full), w2 [16,12] via w2_a[16,4]@w2_b[4,12]
adapter = {
    "attention.to_q.lokr_w1": f32(8, 8),
    "attention.to_q.lokr_w2": f32(12, 12),
    "feed_forward.w1.lokr_w1": f32(16, 8),
    "feed_forward.w1.lokr_w2_a": f32(16, RANK),
    "feed_forward.w1.lokr_w2_b": f32(RANK, 12),
}
meta = {"networkType": "lokr", "alpha": str(ALPHA), "rank": str(RANK)}

# Apply via the fork's REAL loader, then capture the post-adapter output.
applied, matched = LoKrLoader.apply(block, adapter, meta, SCALE)
assert applied == 2, f"expected 2 applied, got {applied}"
y = block(x, None, freqs_cis, t_emb)
base["out.y"] = y.astype(mx.float32)

base_path = "/Users/michael/repos/mlx-gen/tests/fixtures/lokr_loader.safetensors"
adapter_path = "/Users/michael/repos/mlx-gen/tests/fixtures/lokr_adapter.safetensors"
mx.save_safetensors(base_path, base)
mx.save_safetensors(adapter_path, {k: v.astype(mx.float32) for k, v in adapter.items()}, metadata=meta)
print(f"wrote {base_path} ({len(base)} tensors)")
print(f"wrote {adapter_path} ({len(adapter)} tensors, metadata={meta})")
print(f"applied={applied} matched={len(matched)} keys; scale={SCALE}")
