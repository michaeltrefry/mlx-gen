"""Dump a tiny full Z-Image TextEncoder forward parity fixture from the fork.

Run from the fork:  cd ~/repos/mflux && uv run python /Users/michael/repos/mlx-gen/tools/dump_text_encoder.py

Overrides ModelConfig.precision -> float32 so the encoder's final cast is a no-op (clean f32
parity; the bf16 cast for the DiT is a downstream concern). attention_mask is all-ones (causal
path; the padding combination is unit-tested separately). Tiny random config, 2 layers, so the
returned all_hidden_states[-2] is the output after layer 0.
"""

import mlx.core as mx
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder

ModelConfig.precision = mx.float32  # make the encoder's final .astype(precision) a no-op

OUT = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/text_encoder.safetensors"

mx.random.seed(1)
VOCAB, H, NL, NH, NKV, HD, INTER, SEQ = 256, 64, 2, 4, 2, 16, 128, 6

enc = TextEncoder(
    vocab_size=VOCAB,
    hidden_size=H,
    num_hidden_layers=NL,
    num_attention_heads=NH,
    num_key_value_heads=NKV,
    intermediate_size=INTER,
    head_dim=HD,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
)

input_ids = mx.random.randint(0, VOCAB, (1, SEQ)).astype(mx.int32)
attn = mx.ones((1, SEQ), dtype=mx.int32)
out = enc(input_ids, attn)
mx.eval(out)

tensors = {
    "input_ids": input_ids,
    "attention_mask": attn,
    "out": out.astype(mx.float32),
    "embed_tokens.weight": enc.embed_tokens.weight,
}
for i, layer in enumerate(enc.layers):
    p = f"layers.{i}"
    tensors[f"{p}.input_layernorm.weight"] = layer.input_layernorm.weight
    tensors[f"{p}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
    tensors[f"{p}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight
    tensors[f"{p}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight
    tensors[f"{p}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight
    tensors[f"{p}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
    tensors[f"{p}.self_attn.q_norm.weight"] = layer.self_attn.q_norm.weight
    tensors[f"{p}.self_attn.k_norm.weight"] = layer.self_attn.k_norm.weight
    tensors[f"{p}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight
    tensors[f"{p}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight
    tensors[f"{p}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight

tensors = {k: v.astype(mx.float32) if v.dtype != mx.int32 else v for k, v in tensors.items()}
meta = {"cfg": f"{VOCAB},{H},{NL},{NH},{NKV},{HD},{INTER},{SEQ}"}

mx.save_safetensors(OUT, tensors, meta)
print(f"wrote {OUT}: {len(tensors)} tensors, out_shape={tuple(out.shape)} meta={meta}")
