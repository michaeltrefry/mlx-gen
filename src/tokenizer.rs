//! Text tokenization — thin wrapper over the HF [`tokenizers`] crate (the same Rust core
//! `transformers` wraps), so loading a model's `tokenizer.json` reproduces the Python fork's
//! token IDs exactly. Adds the orchestration the fork's `LanguageTokenizer` does on top:
//! chat-template rendering, max-length padding/truncation, and the attention mask.
//!
//! Clean-Rust per ARCHITECTURE.md: chat templating is a small typed [`ChatTemplate`] rather
//! than a full Jinja2 engine. For a single user message the 4 KB Qwen3 template collapses to
//! one deterministic line (verified against the fork) — `minijinja` only earns its place if a
//! future model needs multi-turn/tools rendering (non-speculative).

use std::path::Path;

use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::{Error, Result};

fn tok_err(e: tokenizers::Error) -> Error {
    Error::Msg(format!("tokenizer: {e}"))
}

/// How a raw prompt is wrapped before tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Encode the prompt verbatim (e.g. T5 / CLIP text encoders).
    None,
    /// Qwen single-user-turn + generation prompt — the form Z-Image uses (Qwen2/Qwen3
    /// `apply_chat_template` with one user message, `add_generation_prompt=True`, and
    /// `enable_thinking=True`, which adds no `<think>` block):
    /// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
    QwenInstruct,
    /// Qwen-Image's T2I prompt template (the fork's `LanguageTokenizer` `template=`): a fixed
    /// system instruction + the user prompt + generation prompt. The text encoder later drops the
    /// leading 34 template tokens (`prompt_drop_idx`), keeping the prompt + trailing
    /// `<|im_end|>\n<|im_start|>assistant\n` as conditioning. Verified token-for-token against the
    /// fork's `Qwen2Tokenizer` (the system prefix tokenizes to exactly 34 tokens).
    QwenImage,
}

impl ChatTemplate {
    fn render(&self, prompt: &str) -> String {
        match self {
            ChatTemplate::None => prompt.to_string(),
            ChatTemplate::QwenInstruct => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
            ChatTemplate::QwenImage => format!(
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
                 texture, quantity, text, spatial relationships of the objects and \
                 background:<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ),
        }
    }
}

/// Per-model tokenization config (mirrors the knobs of the fork's `LanguageTokenizer`).
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub max_length: usize,
    pub pad_token_id: i32,
    pub chat_template: ChatTemplate,
    /// `true` = pad every output to `max_length` (the fork's `padding="max_length"`).
    pub pad_to_max_length: bool,
}

/// Result of tokenizing one prompt: `(1, L)` int32 ids + attention mask.
pub struct TokenizerOutput {
    pub input_ids: Array,
    pub attention_mask: Array,
}

/// A loaded text tokenizer plus the model's tokenization policy.
pub struct TextTokenizer {
    inner: Tokenizer,
    config: TokenizerConfig,
}

impl TextTokenizer {
    /// Load from a `tokenizer.json` (the fast-tokenizer file shipped in every HF repo).
    pub fn from_file(path: impl AsRef<Path>, config: TokenizerConfig) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref()).map_err(tok_err)?;
        Ok(Self { inner, config })
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Tokenize one prompt → `(1, L)` int32 `input_ids` + `attention_mask`. An empty prompt
    /// returns `(1, 0)` empty arrays, matching the fork's all-empty short-circuit.
    pub fn tokenize(&self, prompt: &str) -> Result<TokenizerOutput> {
        if prompt.is_empty() {
            return Ok(TokenizerOutput {
                input_ids: empty_row(),
                attention_mask: empty_row(),
            });
        }

        let rendered = self.config.chat_template.render(prompt);
        let encoding = self.inner.encode(rendered, true).map_err(tok_err)?;

        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let max = self.config.max_length;
        if ids.len() > max {
            ids.truncate(max); // right-truncation, as HF does for a single sequence
        }
        let mut mask: Vec<i32> = vec![1; ids.len()];

        if self.config.pad_to_max_length && ids.len() < max {
            ids.resize(max, self.config.pad_token_id);
            mask.resize(max, 0);
        }

        let len = ids.len() as i32;
        Ok(TokenizerOutput {
            input_ids: Array::from_slice(&ids, &[1, len]),
            attention_mask: Array::from_slice(&mask, &[1, len]),
        })
    }
}

fn empty_row() -> Array {
    Array::from_slice::<i32>(&[], &[1, 0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_instruct_template_collapses_to_single_turn() {
        let r = ChatTemplate::QwenInstruct.render("a red fox");
        assert_eq!(
            r,
            "<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn none_template_is_passthrough() {
        assert_eq!(ChatTemplate::None.render("hello"), "hello");
    }

    #[test]
    fn qwen_image_template_wraps_system_and_user() {
        let r = ChatTemplate::QwenImage.render("a red fox");
        assert!(r.starts_with("<|im_start|>system\nDescribe the image by detailing"));
        assert!(r.ends_with("<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n"));
    }
}
