//! # mlx-gen-lens
//!
//! The **Microsoft Lens / Lens-Turbo** provider crate for [`mlx-gen`](mlx_gen) — a native-MLX port
//! of the turbo-distilled text-to-image model that today runs only in a transformers-5 Python
//! sidecar venv (`/opt/lens-venv`). See epic 3164.
//!
//! Lens-Turbo is three components:
//!
//! 1. **Text encoder — gpt-oss-20b** (`GptOssForCausalLM`, subclassed `LensGptOssEncoder`): a
//!    24-layer **MoE** LLM (hidden 2880, 64 query / 8 KV heads, 32 experts top-4) with **attention
//!    sinks**, **alternating sliding/full attention**, **YaRN RoPE**, and clamped-SwiGLU experts.
//!    Used *encoder-only* — forward to layer 23, capture hidden states at layers `[5, 11, 17, 23]`,
//!    no LM head / KV cache / generation. The 32 expert stacks are MXFP4 in the checkpoint; the
//!    attention/router/embedding modules stay dense bf16 (`modules_to_not_convert`).
//! 2. **Denoising DiT — 48-layer dual-stream MMDiT** (`LensTransformer2DModel`): a near-twin of the
//!    `mlx-gen-qwen-image` MMDiT with a multi-layer text-feature front-end.
//! 3. **VAE — `AutoencoderKLFlux2`** — already ported in [`mlx_gen_flux2`]; only a thin Lens decode
//!    shim is new.
//!
//! ## Status
//!
//! Under construction (epic 3164). Shipped so far:
//! - **sc-3165** — crate scaffold, the [`config`] parser, and the gpt-oss **attention core**
//!   ([`text_encoder::gpt_oss::GptOssAttention`]): GQA + learned attention sinks + alternating
//!   sliding/full causal masks + YaRN RoPE.
//! - **sc-3166** — the **MoE** feed-forward ([`text_encoder::gpt_oss::GptOssMoe`]: top-k router +
//!   clamped-SwiGLU experts) + full **decoder-layer** assembly
//!   ([`text_encoder::gpt_oss::GptOssDecoderLayer`]), with MXFP4 expert dequant
//!   ([`text_encoder::mxfp4`]). Validated single-layer against `transformers.GptOssDecoderLayer`.
//! - **sc-3167** — the **harmony tokenizer + Lens chat-template** ([`text::LensTokenizer`]):
//!   byte-exact `input_ids` vs `LensPipeline._build_chat_inputs` (`txt_offset = 97`).
//! - **sc-3171** — the **encoder-only stack + multi-layer hidden capture**
//!   ([`text_encoder::encoder::LensTextEncoder`]): `embed_tokens` → 24 decoder layers with per-layer
//!   sliding/full masks → capture the layer outputs at `[5, 11, 17, 23]` → early-exit at the last
//!   selected layer. Validated end-to-end against the vendor `LensGptOssEncoder` on real bf16 weights.
//! - **sc-3168** — the **denoising DiT** ([`dit::LensTransformer`]): a 48-layer dual-stream MMDiT
//!   (multi-layer text front-end + fused-QKV joint attention with complex axial RoPE + AdaLN
//!   modulation + SwiGLU GateMLP + `AdaLayerNormContinuous`). Validated against the vendor
//!   `LensTransformer2DModel` on real f32 weights (block-0 peak_rel 1.2e-3, full-forward cosine 0.99998).
//!
//! Still to come: the memory-efficient weight conversion / Q4-Q8 re-quant (sc-3172), VAE shim
//! (sc-3169), scheduler (sc-3170), and the generate/e2e integration (sc-3173).

pub mod config;
pub mod dit;
pub mod text;
pub mod text_encoder;
