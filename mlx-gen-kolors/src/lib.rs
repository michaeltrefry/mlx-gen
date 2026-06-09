//! Kolors provider for mlx-gen — a bilingual (Chinese/English) SDXL-family T2I model.
//!
//! Kolors keeps the SDXL U-Net + SDXL VAE but swaps dual-CLIP conditioning for a **ChatGLM3-6B**
//! text encoder (penultimate hidden state = context, last-token last-layer state = pooled). This
//! crate is built up across epic 3090:
//!
//!  - [`chatglm3`] — the ChatGLM3-6B encoder-only forward (sc-3091).
//!  - [`tokenizer`] — the ChatGLM3 SentencePiece tokenizer (sc-3092).
//!  - [`unet`] — the SDXL U-Net + ChatGLM3 context-projection wiring (sc-3093).
//!  - the T2I / img2img pipelines (sc-3094/3095), quant (sc-3096), ControlNet / IP-Adapter-Plus
//!    (sc-3097/98).

pub mod chatglm3;
pub mod ip_adapter;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod unet;

pub use model::Kolors;
