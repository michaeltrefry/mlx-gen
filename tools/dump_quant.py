"""Generate the Q4/Q8 quantization parity fixture for the Rust port (sc-2342).

mflux quantizes via `nn.quantize(model, bits=bits)` — group-wise affine quantization at
MLX's default group_size=64, bits ∈ {4, 8}, over every Linear. This dumps the underlying
`mx.quantize` / `mx.quantized_matmul` / `mx.dequantize` reference results so the Rust crate
(mlx-rs 0.25, an OLDER bundled MLX than mflux's 0.27–0.32) can be checked for byte-level
packing parity — the version-drift risk the epic flagged.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_quant.py
"""

import mlx.core as mx

mx.random.seed(0)

GROUP_SIZE = 64
OUT, IN, BATCH = 128, 256, 4  # IN divisible by group_size

w = mx.random.normal((OUT, IN)).astype(mx.float32)
x = mx.random.normal((BATCH, IN)).astype(mx.float32)

out = {"w": w, "x": x}
for bits in (8, 4):
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
    deq = mx.dequantize(wq, scales, biases, group_size=GROUP_SIZE, bits=bits)
    qmm = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=bits)
    p = f"q{bits}"
    out[f"{p}.wq"] = wq
    out[f"{p}.scales"] = scales.astype(mx.float32)
    out[f"{p}.biases"] = biases.astype(mx.float32)
    out[f"{p}.deq"] = deq.astype(mx.float32)
    out[f"{p}.qmm"] = qmm.astype(mx.float32)
    print(f"bits={bits}: wq{wq.shape}/{wq.dtype} scales{scales.shape} biases{biases.shape} qmm{qmm.shape}")

path = "/Users/michael/repos/mlx-gen/tests/fixtures/quant_q4q8.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors); mlx {mx.__version__ if hasattr(mx, '__version__') else '?'}")
