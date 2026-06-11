//! Snapshot-layout loaders for Chroma. The diffusers checkpoint tree is
//! `tokenizer/` (spiece + configs), `text_encoder/` (T5-XXL), `transformer/` (sharded Chroma DiT),
//! `vae/` (AutoencoderKL), `scheduler/`, `model_index.json`.
//!
//! T5 encoder, VAE, and the pack/unpack/sigma helpers are reused from `mlx-gen-flux`. The only
//! Chroma-specific loading concerns are (1) T5 lives in `text_encoder/` not flux's `text_encoder_2/`,
//! and (2) the tokenizer ships only `spiece.model`, so we load a vendored, prebuilt `tokenizer.json`
//! (materialized by `tools/build_chroma_t5_tokenizer.py`) — never the network.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_gen_flux::T5TextEncoder;
use mlx_gen_z_image::vae::Vae;

use crate::config::{ChromaTransformerConfig, MAX_SEQUENCE_LENGTH};
use crate::transformer::ChromaTransformer;

/// The vendored T5-XXL tokenizer (google t5-v1.1-xxl, converted from Chroma's `spiece.model`).
const T5_TOKENIZER_JSON: &str = include_str!("../assets/t5_tokenizer.json");

pub fn load_tokenizer() -> Result<TextTokenizer> {
    load_tokenizer_with_max_len(MAX_SEQUENCE_LENGTH)
}

/// The vendored T5 tokenizer at a given padded length (production uses [`MAX_SEQUENCE_LENGTH`]; the
/// parity tests use a smaller length — the mask logic is length-agnostic).
pub fn load_tokenizer_with_max_len(max_length: usize) -> Result<TextTokenizer> {
    let config = TokenizerConfig {
        max_length,
        // T5 `<pad>`.
        pad_token_id: 0,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    TextTokenizer::from_json_str(T5_TOKENIZER_JSON, config).map_err(Into::into)
}

pub fn load_t5_encoder(root: &Path) -> Result<T5TextEncoder> {
    // Chroma diffusers layout: T5 is `text_encoder/` (FLUX puts it in `text_encoder_2/`).
    let w = Weights::from_dir(root.join("text_encoder"))?;
    T5TextEncoder::from_weights(&w, "")
}

pub fn load_vae(root: &Path) -> Result<Vae> {
    // Identical AutoencoderKL layout to FLUX — reuse the flux loader (decoder/encoder remap included).
    mlx_gen_flux::load_vae(root)
}

pub fn load_transformer(root: &Path, cfg: ChromaTransformerConfig) -> Result<ChromaTransformer> {
    let w = Weights::from_dir(root.join("transformer"))?;
    ChromaTransformer::from_weights(w, cfg)
}
