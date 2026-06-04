"""Cross-build bf16 op probe (sc-2787): dump a fixed bf16 matmul + bf16 SDPA computed on the mflux
wheel MLX (non-NAX) so the Rust/NAX build can run the SAME bf16 inputs and measure the NAX-vs-wheel
bf16 rounding delta. f32 ops were already proven bit-identical cross-build; this NAMES the residual
in the bf16 CLIP + AdaLN-modulation paths (it should be ~1e-3, i.e. sub-bf16-ULP reduction-order).

Run: /Users/michael/Repos/mflux/.venv/bin/python tools/dump_bf16_crossbuild_probe.py
"""

import os

import mlx.core as mx
from mlx import nn

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_DIR, exist_ok=True)

mx.random.seed(0)
tensors = {}

# Representative CLIP / AdaLN shapes (M, K, N). CLIP proj 77x768x768, fc1 77x768x3072;
# AdaLN gemv 1x3072x18432 (M=1, the "safe" path); a generic deep-stack GEMM 256x3072x3072.
shapes = {
    "clip_proj": (77, 768, 768),
    "clip_fc1": (77, 768, 3072),
    "adaln_gemv": (1, 3072, 18432),
    "joint_qkv": (256, 3072, 3072),
}
for name, (m, k, n) in shapes.items():
    a = mx.random.normal((m, k)).astype(mx.bfloat16)
    b = mx.random.normal((k, n)).astype(mx.bfloat16)
    c = (a @ b).astype(mx.bfloat16)
    mx.eval(c)
    tensors[f"{name}_a"] = a
    tensors[f"{name}_b"] = b
    tensors[f"{name}_c"] = c

# Biased linear = FUSED addmm(bias, x, Wᵀ) — the actual CLIP/AdaLN op (sc-2779). matmul alone was
# bit-identical cross-build; addmm fuses the bias-add so test it separately. CLIP proj + AdaLN gemv.
for name, (m, k, n) in {"clip_proj": (77, 768, 768), "adaln_gemv": (1, 3072, 18432)}.items():
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    w = mx.random.normal((n, k)).astype(mx.bfloat16)  # nn.Linear weight [out, in]
    bias = mx.random.normal((n,)).astype(mx.bfloat16)
    y = mx.addmm(bias, x, w.T).astype(mx.bfloat16)
    mx.eval(y)
    tensors[f"addmm_{name}_x"] = x
    tensors[f"addmm_{name}_w"] = w
    tensors[f"addmm_{name}_bias"] = bias
    tensors[f"addmm_{name}_y"] = y

# bf16 fused SDPA, CLIP head shape [1, 12, 77, 64] — UNMASKED and CAUSAL-masked (CLIP uses a mask).
q = mx.random.normal((1, 12, 77, 64)).astype(mx.bfloat16)
k = mx.random.normal((1, 12, 77, 64)).astype(mx.bfloat16)
v = mx.random.normal((1, 12, 77, 64)).astype(mx.bfloat16)
o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 8.0, mask=None).astype(mx.bfloat16)
mask = (1 - mx.tril(mx.ones((77, 77)), k=0)) * -3.4e38  # f32, then cast like the fork
mask = mask.reshape(1, 1, 77, 77).astype(mx.bfloat16)
om = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 8.0, mask=mask).astype(mx.bfloat16)
mx.eval(o, om, mask)
tensors["sdpa_q"] = q
tensors["sdpa_k"] = k
tensors["sdpa_v"] = v
tensors["sdpa_o"] = o
tensors["sdpa_mask"] = mask
tensors["sdpa_om"] = om

# f32 GEMM + f32 SDPA (the transformer main stream is f32). The NAX build runs these in TF32 by
# default (MLX_ENABLE_TF32=1); the wheel runs TRUE f32. This isolates whether the transformer's
# joint/single-stack residual is the TF32-vs-f32 gap (sc-2787 investigation).
fa = mx.random.normal((256, 3072))  # f32
fb = mx.random.normal((3072, 3072))  # f32
fc = (fa @ fb).astype(mx.float32)
fq = mx.random.normal((1, 24, 256, 128))  # f32 joint attention head shape
fk = mx.random.normal((1, 24, 256, 128))
fv = mx.random.normal((1, 24, 256, 128))
fo = mx.fast.scaled_dot_product_attention(fq, fk, fv, scale=1 / 128.0**0.5, mask=None).astype(mx.float32)
mx.eval(fc, fo)
tensors["f32_mm_a"] = fa
tensors["f32_mm_b"] = fb
tensors["f32_mm_c"] = fc
tensors["f32_sdpa_q"] = fq
tensors["f32_sdpa_k"] = fk
tensors["f32_sdpa_v"] = fv
tensors["f32_sdpa_o"] = fo

# Remaining transformer ops (f32 main stream): affine=False LayerNorm, RMSNorm (QK-norm), the two
# FFN activations (exact gelu + tanh-approx), and silu. Any non-bit-identical one seeds the stack.
xt = mx.random.normal((1, 256, 3072))  # f32
tensors["tln_x"] = xt
tensors["tln_y"] = mx.fast.layer_norm(xt, None, None, 1e-6).astype(mx.float32)  # affine=False
rq = mx.random.normal((1, 24, 256, 128))  # f32 QK-norm input
rw = mx.random.normal((128,))
tensors["rms_x"] = rq
tensors["rms_w"] = rw
tensors["rms_y"] = mx.fast.rms_norm(rq, rw, 1e-5).astype(mx.float32)
tensors["gelu_x"] = xt
tensors["gelu_exact"] = nn.gelu(xt).astype(mx.float32)
tensors["gelu_approx"] = nn.gelu_approx(xt).astype(mx.float32)
tensors["silu_f32"] = nn.silu(xt).astype(mx.float32)
xtb = xt.astype(mx.bfloat16)
tensors["silu_bf16_x"] = xtb
tensors["silu_bf16_y"] = nn.silu(xtb).astype(mx.bfloat16)
mx.eval(*[tensors[k] for k in ["tln_y", "rms_y", "gelu_exact", "gelu_approx", "silu_f32", "silu_bf16_y"]])

# EXACT joint-block f32 shapes (single-shape micro-tests don't prove other shapes — divergence memory):
# f32 BIASED linear (addmm) at M=256, and f32 SDPA over the joint seq=512 (txt256+img256).
ja_x = mx.random.normal((1, 256, 3072))
ja_w = mx.random.normal((3072, 3072))
ja_b = mx.random.normal((3072,))
tensors["f32_addmm_x"] = ja_x
tensors["f32_addmm_w"] = ja_w
tensors["f32_addmm_b"] = ja_b
tensors["f32_addmm_y"] = mx.addmm(ja_b, ja_x.reshape(256, 3072), ja_w.T).astype(mx.float32)
ff_x = mx.random.normal((256, 3072))
ff_w = mx.random.normal((12288, 3072))
ff_b = mx.random.normal((12288,))
tensors["f32_ff1_x"] = ff_x
tensors["f32_ff1_w"] = ff_w
tensors["f32_ff1_b"] = ff_b
tensors["f32_ff1_y"] = mx.addmm(ff_b, ff_x, ff_w.T).astype(mx.float32)
jq = mx.random.normal((1, 24, 512, 128))
jk = mx.random.normal((1, 24, 512, 128))
jv = mx.random.normal((1, 24, 512, 128))
tensors["j512_q"] = jq
tensors["j512_k"] = jk
tensors["j512_v"] = jv
tensors["j512_o"] = mx.fast.scaled_dot_product_attention(jq, jk, jv, scale=1 / 128.0**0.5, mask=None).astype(mx.float32)
mx.eval(tensors["f32_addmm_y"], tensors["f32_ff1_y"], tensors["j512_o"])

# bf16 layer_norm [1, 77, 768] (affine) and bf16 sigmoid (quick_gelu uses it).
ln_x = mx.random.normal((1, 77, 768)).astype(mx.bfloat16)
ln_w = mx.random.normal((768,)).astype(mx.bfloat16)
ln_b = mx.random.normal((768,)).astype(mx.bfloat16)
ln_y = mx.fast.layer_norm(ln_x, ln_w, ln_b, 1e-5).astype(mx.bfloat16)
sig_y = mx.sigmoid(ln_x).astype(mx.bfloat16)
mx.eval(ln_y, sig_y)
tensors["ln_x"] = ln_x
tensors["ln_w"] = ln_w
tensors["ln_b"] = ln_b
tensors["ln_y"] = ln_y
tensors["sig_y"] = sig_y

out = os.path.join(_DIR, "bf16_crossbuild_probe.safetensors")
mx.save_safetensors(out, tensors, {"mlx": mx.__version__})
print("wrote", out, "mlx", mx.__version__)
