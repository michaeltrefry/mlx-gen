//! Kolors U-Net (sc-3093) — the SDXL `UNet2DConditionModel` reused as-is, plus the ChatGLM3 context
//! projection. Kolors' U-Net is structurally identical to SDXL base (same blocks/channels/heads,
//! `cross_attention_dim` 2048) with two ChatGLM-driven deltas, both handled in `mlx-gen-sdxl`:
//!
//!  - an **`encoder_hid_proj`** Linear (4096→2048) that projects the ChatGLM3 context to the
//!    cross-attention width — auto-detected from the checkpoint by `UNet2DConditionModel`
//!    (`UNetConfig::kolors`);
//!  - the **5632**-wide `add_embedding.linear_1` (pooled 4096 + 6·256 time-ids), loaded by shape.
//!
//! So the wiring is: feed the encoder the ChatGLM3 **context** (`[B, S, 4096]`) and **pooled**
//! (`[B, 4096]`) + SDXL-style `time_ids` (`[B, 6]`) straight into the U-Net `forward` — the
//! projection to 2048 and the 5632 added-conditioning happen inside. Re-exported here so the Kolors
//! provider has one entry point; the T2I denoise loop + scheduler are sc-3094.

pub use mlx_gen_sdxl::{load_unet_kolors_dtype, UNet2DConditionModel};

/// Kolors **ControlNet** (sc-3097): the SDXL `ControlNetModel` primitive reused as-is. The Kolors
/// ControlNet (`Kwai-Kolors/Kolors-ControlNet-{Pose,Canny,Depth}`) is a standard SDXL ControlNet
/// whose only deltas are the same two ChatGLM-driven pieces as the Kolors U-Net — its **own**
/// `encoder_hid_proj` (4096→2048, distinct learned weights) auto-detected from the checkpoint, and
/// the 5632 add-embedding loaded by shape. `load_controlnet` therefore loads it with no Kolors-
/// specific config. Conditioned with the ChatGLM3 context (the branch projects it internally).
pub use mlx_gen_sdxl::{load_controlnet, ControlNet};
