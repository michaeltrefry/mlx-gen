"""Full Z-Image DiT forward parity fixture (sc-2344, denoiser PR 2).

A tiny synthetic ZImageTransformer (dim=96, 4 heads, 1 refiner + 2 main layers, in_ch=4,
patch=2) run end-to-end. Dumps all weights, the inputs, the final output, AND per-stage
intermediates (t_emb, embedded x/cap, post-refiner, unified, output) so a Rust parity
failure localizes to a stage instead of the whole forward.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_z_transformer.py
"""

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

mx.random.seed(0)

CFG = dict(
    patch_size=2, f_patch_size=1, in_channels=4, dim=96, n_layers=2, n_refiner_layers=1,
    n_heads=4, norm_eps=1e-5, qk_norm=True, cap_feat_dim=32, rope_theta=256.0, t_scale=1000.0,
    axes_dims=[8, 8, 8], axes_lens=[64, 64, 64],
)
model = ZImageTransformer(**CFG)
# Exercise the assembly-only norm (cap RMSNorm); block/final norms are covered elsewhere.
model.cap_embedder[0].weight = 1.0 + 0.1 * mx.random.normal(model.cap_embedder[0].weight.shape)

x = mx.random.normal((4, 1, 4, 4))          # (C=in_channels, F, H, W)
cap_feats = mx.random.normal((5, 32))        # (cap_len, cap_feat_dim)
timestep = mx.array(0.7, dtype=mx.float32)
sigmas = mx.linspace(1.0, 0.0, 8)            # unused for float timestep, kept for signature

out = {f"w.{k}": v.astype(mx.float32) for k, v in tree_flatten(model.parameters())}
out["in.x"] = x.astype(mx.float32)
out["in.cap_feats"] = cap_feats.astype(mx.float32)

# Replicate the forward with per-stage capture (mirrors transformer.py.__call__).
key = f"{CFG['patch_size']}-{CFG['f_patch_size']}"
t = (timestep.reshape((1,)) if timestep.ndim == 0 else timestep).astype(mx.float32) * CFG["t_scale"]
t_emb = model.t_embedder(t)
out["mid.t_emb"] = t_emb.astype(mx.float32)

x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(
    image=x, cap_feats=cap_feats, patch_size=CFG["patch_size"], f_patch_size=CFG["f_patch_size"]
)
out["mid.x_tokens"] = x_emb.astype(mx.float32)        # patchified, pre-embed
out["mid.x_pos_ids"] = x_pos_ids.astype(mx.int32)
out["mid.cap_pos_ids"] = cap_pos_ids.astype(mx.int32)

x_emb = model.all_x_embedder[key](x_emb)
x_emb = mx.where(x_pad_mask[:, None], model.x_pad_token, x_emb)
x_freqs_cis = model.rope_embedder(x_pos_ids)
x_emb = mx.expand_dims(x_emb, axis=0)
for layer in model.noise_refiner:
    x_emb = layer(x=x_emb, attn_mask=None, freqs_cis=x_freqs_cis, t_emb=t_emb)
out["mid.x_refined"] = x_emb.astype(mx.float32)

cap_emb = model.cap_embedder[1](model.cap_embedder[0](cap_emb))
cap_emb = mx.where(cap_pad_mask[:, None], model.cap_pad_token, cap_emb)
cap_freqs_cis = model.rope_embedder(cap_pos_ids)
cap_emb = mx.expand_dims(cap_emb, axis=0)
for layer in model.context_refiner:
    cap_emb = layer(x=cap_emb, attn_mask=None, freqs_cis=cap_freqs_cis)
out["mid.cap_refined"] = cap_emb.astype(mx.float32)

x_len = x_emb.shape[1]
unified = mx.concatenate([x_emb, cap_emb], axis=1)
unified_freqs_cis = mx.concatenate([x_freqs_cis, cap_freqs_cis], axis=0)
for layer in model.layers:
    unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs_cis, t_emb=t_emb)
out["mid.unified"] = unified.astype(mx.float32)

unified = model.all_final_layer[key](unified, t_emb)
output = ZImageTransformer._unpatchify(
    x=unified[0, :x_len], size=x_size, patch_size=CFG["patch_size"],
    f_patch_size=CFG["f_patch_size"], out_channels=model.out_channels,
)
y_ref = model(x, timestep, sigmas, cap_feats)
assert mx.allclose(-output, y_ref, atol=1e-5).item(), "staged replication diverged from __call__"
out["out.y"] = y_ref.astype(mx.float32)

path = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/z_transformer.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
print("out.y:", y_ref.shape, "| x_tokens:", out["mid.x_tokens"].shape, "| unified:", out["mid.unified"].shape)
