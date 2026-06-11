//! Chroma text conditioning (sc-3838) — the T5-XXL **masked** encode + the MMDiT attention-mask
//! construction. This is the parity-critical path that differs from FLUX (which runs T5 unmasked and
//! applies no transformer mask).
//!
//! `ChromaPipeline._get_t5_prompt_embeds`:
//! 1. tokenize at `max_length` (`pad_to_max_length`), giving `input_ids` + the padding mask;
//! 2. run T5 **with** the padding mask (so padded tokens don't pollute the real-token embeddings);
//! 3. build the transformer mask `(arange(L) <= seq_lengths)` — Chroma's quirk that **keeps one
//!    extra padding token** past the content.
//!
//! Real tokens are a contiguous prefix; we derive their count from `input_ids != pad_id` rather than
//! the tokenizer's `attention_mask` (the vendored `tokenizer.json` has padding baked in, so the HF
//! fast tokenizer auto-pads and its returned mask is all-ones). The full-sequence mask (this text
//! mask ++ image ones) is assembled in the generate path (sc-3839).

use mlx_gen::tokenizer::{to_arrays, TextTokenizer};
use mlx_gen::Result;
use mlx_gen_flux::T5TextEncoder;
use mlx_rs::{Array, Dtype};

/// Large negative added to padded keys in the T5 self-attention (softmax → exactly 0 weight in f32).
const T5_MASK_NEG: f32 = -1e9;

/// Encode one prompt → `(prompt_embeds [1, L, 4096], text_mask [1, L])`.
///
/// `prompt_embeds` is the T5-XXL last hidden state computed **with** the padding mask. `text_mask` is
/// the per-token transformer mask (0/1) with the keep-one-extra-pad quirk — the caller extends it
/// with image-token ones before threading it into the DiT.
pub fn encode_prompt(
    tokenizer: &TextTokenizer,
    t5: &T5TextEncoder,
    prompt: &str,
) -> Result<(Array, Array)> {
    let tok = tokenizer.tokenize(prompt)?;
    let (input_ids, _) = to_arrays(&tok);
    let pad = tokenizer.config().pad_token_id;
    let key_mask = t5_key_mask(&input_ids, pad)?;
    let embeds = t5.forward_masked(&input_ids, Some(&key_mask))?;
    let text_mask = transformer_text_mask(&input_ids, pad)?;
    Ok((embeds, text_mask))
}

/// Per-row count of real (non-pad) tokens — a contiguous prefix. Returns `Result` so an MLX cast
/// failure surfaces as the crate's `Error` to the worker instead of aborting the process (F-104).
fn real_lengths(input_ids: &Array, pad: i32) -> Result<(usize, usize, Vec<usize>)> {
    let b = input_ids.shape()[0] as usize;
    let l = input_ids.shape()[1] as usize;
    let ids: Vec<i32> = input_ids.as_dtype(Dtype::Int32)?.as_slice::<i32>().to_vec();
    let lens = (0..b)
        .map(|bi| {
            ids[bi * l..(bi + 1) * l]
                .iter()
                .filter(|&&x| x != pad)
                .count()
        })
        .collect();
    Ok((b, l, lens))
}

/// The additive T5 key-padding mask `[B,1,1,L]` = `(1 - real) * NEG` (real = `id != pad`),
/// broadcastable to the T5 attention scores.
pub fn t5_key_mask(input_ids: &Array, pad: i32) -> Result<Array> {
    let (b, l, lens) = real_lengths(input_ids, pad)?;
    let mut data = vec![0f32; b * l];
    for (bi, &len) in lens.iter().enumerate() {
        for (i, slot) in data[bi * l..(bi + 1) * l].iter_mut().enumerate() {
            *slot = if i < len { 0.0 } else { T5_MASK_NEG };
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, 1, l as i32]))
}

/// The transformer per-token mask `(arange(L) <= seq_lengths)` as `[B, L]` f32 — **keeps one extra
/// padding token** past the content (Chroma's `_get_t5_prompt_embeds` quirk).
pub fn transformer_text_mask(input_ids: &Array, pad: i32) -> Result<Array> {
    let (b, l, lens) = real_lengths(input_ids, pad)?;
    let mut out = vec![0f32; b * l];
    for (bi, &len) in lens.iter().enumerate() {
        for (i, slot) in out[bi * l..(bi + 1) * l].iter_mut().enumerate() {
            *slot = if i <= len { 1.0 } else { 0.0 };
        }
    }
    Ok(Array::from_slice(&out, &[b as i32, l as i32]))
}
