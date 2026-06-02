//! Ignored real-weights check for the FLUX.1 text path.
//!
//! Run intentionally with a local FLUX.1 snapshot:
//!
//! ```text
//! MLX_GEN_FLUX_SNAPSHOT=/path/to/FLUX.1-schnell/snapshot \
//!   cargo test -p mlx-gen-flux --test text_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen_flux::FluxVariant;
use mlx_gen_flux::{load_clip_encoder, load_clip_tokenizer, load_t5_encoder, load_t5_tokenizer};

#[test]
#[ignore = "loads multi-GB FLUX.1 CLIP/T5 text weights; set MLX_GEN_FLUX_SNAPSHOT"]
fn flux_text_path_runs_on_real_weights() {
    let root = PathBuf::from(
        std::env::var("MLX_GEN_FLUX_SNAPSHOT")
            .expect("set MLX_GEN_FLUX_SNAPSHOT to a FLUX.1 snapshot directory"),
    );
    let variant = match std::env::var("MLX_GEN_FLUX_VARIANT").as_deref() {
        Ok("dev") => FluxVariant::Dev,
        _ => FluxVariant::Schnell,
    };

    let t5_tok = load_t5_tokenizer(&root, variant).unwrap();
    let clip_tok = load_clip_tokenizer(&root).unwrap();
    let t5 = load_t5_encoder(&root).unwrap();
    let clip = load_clip_encoder(&root).unwrap();

    let prompt = "a red fox";
    let t5_out = t5_tok.tokenize(prompt).unwrap();
    let clip_out = clip_tok.tokenize(prompt).unwrap();
    let prompt_embeds = t5.forward(&t5_out.input_ids).unwrap();
    let pooled = clip.forward(&clip_out.input_ids).unwrap();

    assert_eq!(prompt_embeds.shape()[0], 1);
    assert_eq!(
        prompt_embeds.shape()[1] as usize,
        variant.max_sequence_length()
    );
    assert_eq!(prompt_embeds.shape()[2], 4096);
    assert_eq!(pooled.shape(), &[1, 768]);
}
