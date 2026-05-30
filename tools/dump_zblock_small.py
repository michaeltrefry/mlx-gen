"""Generate the committed Z-Image-block parity fixture for the Rust port (sc-2344).

Mirrors the sc-2338 spike harness (~/repos/mlx-rs-spike/dump_zblock.py) but at TINY
dimensions so the fixture commits cleanly (~a few hundred KB vs the spike's 725 MB at
real dims). The block forward is dimension-agnostic, so small dims prove the *function*
just as well — and reduced reduction lengths make Metal fp32 matmul agree more tightly,
not less. Randomized norm weights + adaLN bias exercise every weight-multiply path
(default RMSNorm weight is all-ones, adaLN bias all-zeros, which would hide a bug).

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_zblock_small.py
"""

import mlx.core as mx

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock

mx.random.seed(0)

DIM, N_HEADS, SEQ = 96, 4, 4
HEAD_DIM = DIM // N_HEADS  # 24
ADA_IN = min(DIM, 256)     # adaLN input dim

block = ZImageTransformerBlock(dim=DIM, n_heads=N_HEADS, norm_eps=1e-5, qk_norm=True)

for nrm in (block.attention_norm1, block.attention_norm2, block.ffn_norm1, block.ffn_norm2):
    nrm.weight = 1.0 + 0.1 * mx.random.normal(nrm.weight.shape)
block.attention.norm_q.weight = 1.0 + 0.1 * mx.random.normal((HEAD_DIM,))
block.attention.norm_k.weight = 1.0 + 0.1 * mx.random.normal((HEAD_DIM,))
block.adaLN_modulation[0].bias = 0.05 * mx.random.normal(block.adaLN_modulation[0].bias.shape)

# Fixed inputs (no attention mask → full-valid, i.e. SDPA mask=None).
x = mx.random.normal((1, SEQ, DIM))
t_emb = mx.random.normal((1, ADA_IN))
freqs_cis = mx.random.normal((SEQ, HEAD_DIM // 2, 2))

y = block(x, None, freqs_cis, t_emb)

from mlx.utils import tree_flatten  # noqa: E402

out = {f"w.{k}": v.astype(mx.float32) for k, v in tree_flatten(block.parameters())}
out["in.x"] = x.astype(mx.float32)
out["in.t_emb"] = t_emb.astype(mx.float32)
out["in.freqs_cis"] = freqs_cis.astype(mx.float32)
out["out.y"] = y.astype(mx.float32)

path = "/Users/michael/repos/mlx-gen/tests/fixtures/zblock_small.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path}  ({len(out)} tensors, dim={DIM} heads={N_HEADS} seq={SEQ})")
print("y shape:", y.shape, "y dtype:", y.dtype)
