//! S6 — the LTX-2.3 prompt tokenizer (Gemma-3). The reference `LTX2TextEncoder.encode` runs the
//! HF Gemma tokenizer on the **raw prompt** (no chat template) with `add_special_tokens=True` (a
//! leading `<bos>`, no EOS), truncates to `max_length`, and **left-pads** to `max_length` with
//! `<pad>` (id 0) — `padding_side="left"`. Left-padding matters: it places the real tokens at the
//! high RoPE positions `[max_length−L, max_length)`, which is what the S1 Gemma forward was validated
//! against.
//!
//! Built on the shared core [`TextTokenizer`] (HF `tokenizer.json`, `ChatTemplate::None`); the
//! left-pad is applied here (core pads right) to keep the change crate-local.

use std::path::Path;

use mlx_rs::Array;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::{Error, Result};

/// Gemma-3 `<pad>` token id (`tokenizer_config.json`).
const GEMMA_PAD_ID: i32 = 0;

/// The LTX-2.3 Gemma prompt tokenizer.
pub struct LtxTokenizer {
    inner: TextTokenizer,
}

impl LtxTokenizer {
    /// Load `tokenizer.json` from a Gemma snapshot directory.
    pub fn from_dir(gemma_dir: &Path) -> Result<Self> {
        let path = gemma_dir.join("tokenizer.json");
        if !path.exists() {
            return Err(Error::Msg(format!(
                "ltx tokenizer: {} not found (set LTX_GEMMA_DIR to a gemma-3-12b-it snapshot)",
                path.display()
            )));
        }
        // No core-side truncation/padding: encode() truncates + left-pads itself (core pads right).
        let cfg = TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: GEMMA_PAD_ID,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        };
        Ok(Self {
            inner: TextTokenizer::from_file(path, cfg)?,
        })
    }

    /// Encode a raw prompt → `(1, max_length)` left-padded int32 `input_ids` + `attention_mask`.
    /// Mirrors the reference: `<bos>` prepended (via `add_special_tokens`), right-truncated to
    /// `max_length`, then left-padded with `<pad>`.
    pub fn encode(&self, prompt: &str, max_length: usize) -> Result<(Array, Array)> {
        if prompt.is_empty() {
            return Err(Error::Msg("ltx tokenizer: empty prompt".into()));
        }
        let out = self.inner.tokenize(prompt)?; // (1, L): <bos> + tokens, mask all 1
        let mut ids: Vec<i32> = out.ids.clone();
        if ids.len() > max_length {
            ids.truncate(max_length); // HF truncation=True keeps the leading max_length tokens
        }
        let valid = ids.len();
        let pad = max_length - valid;
        // Left-pad: <pad>×pad ++ ids ; mask 0×pad ++ 1×valid.
        let mut padded = vec![GEMMA_PAD_ID; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0i32; pad];
        mask.resize(max_length, 1); // pad zeros already in place; fill the valid tail with 1s
        let n = max_length as i32;
        Ok((
            Array::from_slice(&padded, &[1, n]),
            Array::from_slice(&mask, &[1, n]),
        ))
    }

    /// Tokenize an **already chat-templated** string to a flat id list with `add_special_tokens=false`
    /// (no auto BOS — the template supplies the `<start_of_turn>` markers itself). The prompt-enhancer
    /// path (sc-2845) uses this, mirroring the reference `processor(formatted, add_special_tokens=False)`.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<i32>> {
        self.inner.encode_ids(text, false).map_err(Into::into)
    }

    /// Detokenize generated ids → text, dropping special tokens — the reference
    /// `processor.decode(generated_tokens, skip_special_tokens=True)`.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner.decode(&u, true).map_err(Into::into)
    }
}
