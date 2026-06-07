//! # mlx-gen-wan
//!
//! Wan2.2 **video** provider crate for [`mlx-gen`]. Port of the `mlx-video-with-audio` package's
//! Wan path (`generate_wan.py`, `models/wan/*`) onto Rust + `mlx-rs`.
//!
//! **First-class target:** the **Wan2.2 TI2V-5B** — the dense 5B (dim 3072, 30 layers, in/out 48)
//! with its own z48 VAE (`vae22`), delivering text-to-video (T2V) plus native image-conditioned
//! (TI2V) video (sc-2680). The shared infra here (UMT5-XXL TE, the Wan DiT, 3-axis RoPE, 3-D
//! patchify, the flow-match solvers, the T2V pipeline) is the Wan core (sc-2678); the dense/MoE
//! 14B variants reuse it via additional configs + dual-expert routing.
//!
//! This crate self-registers three models into the `mlx-gen` registry: **`wan2_2_t2v_14b`** (the
//! dual-expert MoE T2V, fully wired — `mlx_gen::load("wan2_2_t2v_14b", spec)` runs the complete
//! pipeline), **`wan2_2_i2v_14b`** (the dual-expert MoE channel-concat image→video, fully wired —
//! shares the T2V pipeline with the 20-channel `y` conditioning + in_dim-36 patch-embed, sc-2681) and
//! **`wan2_2_ti2v_5b`** (the dense 5B, fully wired — sc-2680: text→video plus native image-conditioned
//! (TI2V) mask-blend video, with its own z48 [`Wan22Vae`]; Q4/Q8 + LoRA/LoKr supported).
//!
//! ## Status (S0–S6)
//! S0 — foundation: registry + config (`config.json`-driven, all Wan presets) + the three
//! flow-match solvers (Euler / DPM++2M / UniPC default) with the shifted-sigma schedule + integer
//! timesteps + 3-axis factorized 3-D RoPE (θ=10000) + 3-D patchify/unpatchify.
//! S1 — the [`Umt5Encoder`] UMT5-XXL text encoder (f32) + `_clean_text`-faithful prompt cleaning,
//! parity-gated against the `mlx_video` reference (bit-exact).
//! S2 — the [`WanVae`] Wan **2.1** VAE (z16, stride 4×8×8): 3-D causal-conv decoder + chunked
//! encoder, channel-L2 norm, per-frame spatial attention, temporal up/down `time_conv`. f32,
//! parity-gated against the reference. (The 5B's distinct z48 `vae22` is sc-2680.)
//! S3 — the [`WanTransformer`] Wan DiT (5B: 30 blocks, qk-RMSNorm self-attn + 3-axis RoPE,
//! text cross-attn, adaLN-6vec modulation, gated-GELU FFN, modulated head). f32 activations,
//! parity-gated f32-against-f32 vs the reference (patch-embed bit-exact).
//! S4 — the [`pipeline`] dense **T2V** machinery: resolution/seq-len math + the CFG denoise loop
//! (`pipeline::denoise`) + VAE decode → uint8 frames (`pipeline::decode_to_frames`). Parity-gated
//! e2e against the reference on a tiny seeded model (injected noise+context).
//! S5 — dual-expert **MoE** routing ([`pipeline::denoise_moe`] + [`Expert`]): a per-step boundary
//! swap (`t ≥ boundary·num_train`) between the high/low-noise experts, each with its own contexts,
//! cross-KV, and guidance. Parity-gated e2e on two tiny seeded experts (both fired across the
//! boundary).
//! S6 — the live `wan2_2_t2v_14b` [`Generator::generate`] ([`model::Wan14b`]): the staged product
//! pipeline (UMT5 encode → two real 40-layer experts → `denoise_moe` → z16 VAE decode → RGB8
//! frames), verified end-to-end on the **real converted Wan2.2-T2V-A14B checkpoint** against a
//! `mlx_video`-reference golden on matched injected noise (`tests/s6_real_parity.rs`, `#[ignore]` —
//! the 54 GB weights live outside CI; the tiny S1–S5 gates carry CI).
//!
//! sc-2680 — the dense **TI2V-5B** ([`model::Wan`]): the z48 [`Wan22Vae`] (`vae22`: channels-last
//! causal-conv decoder/encoder, spatial 2×2 patchify, `DupUp3D`/`AvgDown3D`, RMS-L2 norm; gated by
//! `tests/vae22_parity.rs`) + the dense [`Generator::generate`] — **T2V** ([`denoise`]) and
//! **image-conditioned TI2V** mask-blend ([`denoise_ti2v`] + the DiT's per-token-timestep
//! [`WanTransformer::forward_tokens`], gated by `tests/ti2v_parity.rs`). Q4/Q8 (`spec.quantize`) +
//! LoRA/LoKr merge onto the single dense model. The full e2e on the real converted 5B checkpoint is
//! `tests/ti2v_real_parity.rs` (`#[ignore]` — heavy weights outside CI).

pub mod adapters;
pub mod config;
pub mod convert;
pub mod model;
pub mod model_vace;
pub mod patchify;
pub mod pipeline;
pub mod pth;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vace;
pub mod vae;
pub mod vae22;

pub use adapters::{merge_wan_adapters, WanLoraReport};
pub use config::{GuideScale, WanModelConfig, WanQuant, WanVaceConfig, SAMPLE_NEG_PROMPT};
pub use model::{
    descriptor, descriptor_i2v_14b, descriptor_t2v_14b, load, Wan, Wan14b, MODEL_ID,
    MODEL_ID_I2V_14B, MODEL_ID_T2V_14B,
};
pub use model_vace::{descriptor_vace, WanVace, MODEL_ID_VACE};
pub use pipeline::{
    best_output_size, build_i2v_y, build_ti2v_keyframe_z, build_ti2v_mask, build_ti2v_multi_mask,
    decode_to_frames, decode_to_frames_22, denoise, denoise_moe, denoise_ti2v, frames_to_images,
    preprocess_i2v_image, preprocess_ti2v_image, ti2v_blend_init, Expert,
};
pub use rope::{rope_apply, RopeTable};
pub use scheduler::{
    compute_sigmas, make_scheduler, FlowDpmpp2m, FlowMatchEuler, FlowUniPC, SolverKind,
    WanScheduler,
};
pub use text_encoder::{clean_text, load_tokenizer, umt5_tokenizer_config, Umt5Encoder};
pub use training::{load_trainer, WanMoeTrainer};
pub use transformer::WanTransformer;
pub use vace::{
    binarize_mask, build_vace_control, prepare_masks, prepare_video_latents, WanVaceTransformer,
};
pub use vae::WanVae;
pub use vae22::Wan22Vae;
