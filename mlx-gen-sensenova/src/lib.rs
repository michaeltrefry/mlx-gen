//! # mlx-gen-sensenova
//!
//! The **SenseNova-U1** (NEO-Unify) provider crate for [`mlx-gen`](mlx_gen). NEO-Unify is a
//! *unified* multimodal model — one network does both understanding and image generation, with
//! **no separate VAE or text encoder** (unlike every diffusion-pipeline provider). The first-class
//! target is **`sensenova/SenseNova-U1-8B-MoT`**, which powers SceneWorks **Document Studio**
//! (interleaved text-image) plus Image / image-edit / character / VQA. See epic 3180.
//!
//! ## Architecture as it actually loads (validated against the 8B-MoT checkpoint, sc-3181)
//!
//! "MoT" is **Mixture of *Transformers***, not Mixture of Experts. For the 8B-MoT checkpoint the
//! backbone is the **dense** `Qwen3` (`NEOLLMConfig`, `modeling_qwen3.py`) — there are **no expert
//! stacks and no router** in the weights. Each of the 42 decoder layers carries **two parallel
//! dense transformer paths**:
//!
//! * the **understanding** path — `input_layernorm`, `self_attn.{q,k,v,o}_proj`, the QK-norms
//!   (`q_norm`/`k_norm`) and their **spatial** counterparts (`q_norm_hw`/`k_norm_hw`),
//!   `post_attention_layernorm`, and a SwiGLU `mlp`;
//! * the **generation** path — the same modules with a `_mot_gen` suffix.
//!
//! Tokens are dispatched between the paths per-token by the `image_gen_indicators` mask;
//! **attention K/V is shared/joint across the full sequence** (only Q/O, the norms, and the MLP
//! fork per path). RoPE layers three independent rotations over `head_dim`: temporal
//! (`rope_theta`) + height + width (`rope_theta_hw`). The "vision tower" is **not** a transformer
//! here — `vision_model` (and the generation-path `fm_modules.vision_model_mot_gen`) are just a
//! Conv patch-embedder + 2D-RoPE + Conv dense-embedder. The latent→pixel path is the `fm_head`
//! (flow-matching head) → unpatchify to RGB (`use_pixel_head=false`, so the conv pixel decoders in
//! the reference are dead code for this checkpoint).
//!
//! ## Status
//!
//! The foundation tier (sc-3181 … sc-3186) ships the crate scaffold, the [`config`] parser, the
//! [`loader`] weight-map foundation, the [`qwen3`] backbone, the [`vision`] embedder, the [`fm`]
//! flow-matching head, and the [`text`] tokenizer/template. sc-3187 (this slice) adds the
//! [`runtime`] AR text-generation layer — the [`KvCache`](qwen3::KvCache) + cached forward, token
//! [`Sampler`], greedy/sampled `generate`, and the `_generate_think` rollout — the runtime the
//! generation modes build on. The generation modes (T2I / edit / interleave / VQA) and the
//! `Generator` impl + `inventory` registration land in the following stories (sc-3188 … sc-3194).

pub mod config;
pub mod fm;
pub mod loader;
pub mod model;
pub mod qwen3;
pub mod runtime;
pub mod t2i;
pub mod text;
pub mod vision;

pub use config::{NeoChatConfig, NeoLlmConfig, NeoVisionConfig};
pub use fm::{
    apply_time_schedule, euler_step, patchify, unpatchify, velocity, FmHead, TimestepEmbedder,
};
pub use loader::{check_coverage, expected_keys, load_raw, Coverage};
pub use model::{descriptor, load, SenseNova, MODEL_ID};
pub use qwen3::{KvCache, Path, Qwen3Backbone};
pub use runtime::{Sampler, ThinkRollout};
pub use t2i::{
    interleave_resolution_for, smart_resize, CfgNorm, InterleaveOutput, T2iModel, T2iOptions,
    T2iOutput, INTERLEAVE_RESOLUTIONS,
};
pub use text::{
    build_neo1_query, image_indexes, load_tokenizer, text_indexes, INTERLEAVE_SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_FOR_GEN,
};
pub use vision::NeoVisionEmbedder;
