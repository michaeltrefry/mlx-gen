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
//! This crate self-registers two models into the `mlx-gen` registry: **`wan2_2_t2v_14b`** (the
//! dual-expert MoE T2V, fully wired — `mlx_gen::load("wan2_2_t2v_14b", spec)` runs the complete
//! pipeline) and **`wan2_2_ti2v_5b`** (the dense 5B, whose z48 VAE + dense denoise are sc-2680, so
//! its `generate` still stubs).
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

pub mod config;
pub mod model;
pub mod patchify;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use config::{GuideScale, WanModelConfig, SAMPLE_NEG_PROMPT};
pub use model::{descriptor, descriptor_t2v_14b, load, Wan, Wan14b, MODEL_ID, MODEL_ID_T2V_14B};
pub use pipeline::{decode_to_frames, denoise, denoise_moe, frames_to_images, Expert};
pub use rope::{rope_apply, RopeTable};
pub use scheduler::{
    compute_sigmas, make_scheduler, FlowDpmpp2m, FlowMatchEuler, FlowUniPC, SolverKind,
    WanScheduler,
};
pub use text_encoder::{clean_text, load_tokenizer, umt5_tokenizer_config, Umt5Encoder};
pub use transformer::WanTransformer;
pub use vae::WanVae;
