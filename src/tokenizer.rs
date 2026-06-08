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
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::template::TemplateProcessing;
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
    /// Qwen single-user-turn + generation prompt with `enable_thinking=False` — the form FLUX.2's
    /// Qwen3 text encoder uses. Identical to [`QwenInstruct`](Self::QwenInstruct) but the
    /// generation prompt appends an empty `<think>…</think>` block (verified against the fork's
    /// `apply_chat_template(..., enable_thinking=False)`):
    /// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
    QwenInstructNoThink,
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
            ChatTemplate::QwenInstructNoThink => format!(
                "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
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

    /// Load from an in-memory `tokenizer.json` string — a vendored HF fast-tokenizer file compiled
    /// into the crate (e.g. via `include_str!`). Use this when a model repo ships only
    /// `vocab.json`+`merges.txt` and the byte-level [`from_clip_bpe`](Self::from_clip_bpe) path is
    /// wrong for that tokenizer family (e.g. CLIP, which is lowercased word-BPE with `</w>`, not
    /// GPT-2 byte-level): bundle the correct json and load it explicitly rather than silently
    /// mis-tokenizing (sc-2787).
    pub fn from_json_str(json: &str, config: TokenizerConfig) -> Result<Self> {
        let inner = Tokenizer::from_bytes(json.as_bytes()).map_err(tok_err)?;
        Ok(Self { inner, config })
    }

    /// Load a **GPT-2-style byte-level** BPE tokenizer from the split HF files (`vocab.json` +
    /// `merges.txt`) and install a BOS/EOS post-processor.
    ///
    /// ⚠️ This is byte-level (GPT-2/RoBERTa) BPE, NOT CLIP's lowercased word-BPE with `</w>`
    /// end-of-word suffixes — despite the historical name it does **not** reproduce CLIP ids. Prefer
    /// [`from_file`](Self::from_file)/[`from_json_str`](Self::from_json_str) with a real
    /// `tokenizer.json`. Kept only for byte-level-BPE callers (sc-2787).
    pub fn from_clip_bpe(
        vocab: impl AsRef<Path>,
        merges: impl AsRef<Path>,
        config: TokenizerConfig,
    ) -> Result<Self> {
        let vocab = vocab
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::Msg("tokenizer: vocab path is not UTF-8".into()))?;
        let merges = merges
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::Msg("tokenizer: merges path is not UTF-8".into()))?;
        let bpe = BPE::from_file(vocab, merges)
            .unk_token("<|endoftext|>".into())
            .build()
            .map_err(tok_err)?;
        let mut inner = Tokenizer::new(bpe);
        inner.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
        inner.with_decoder(Some(ByteLevel::default()));
        inner.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<|startoftext|> $A <|endoftext|>")
                .map_err(|e| Error::Msg(format!("tokenizer: {e}")))?
                .try_pair("<|startoftext|> $A <|endoftext|> <|endoftext|> $B:1 <|endoftext|>:1")
                .map_err(|e| Error::Msg(format!("tokenizer: {e}")))?
                .special_tokens(vec![("<|startoftext|>", 49406), ("<|endoftext|>", 49407)])
                .build()
                .map_err(|e| Error::Msg(format!("tokenizer: {e}")))?,
        ));
        Ok(Self { inner, config })
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Tokenize one prompt → `(1, L)` int32 `input_ids` + `attention_mask`. An empty prompt returns
    /// `(1, 0)` empty arrays (the fork's all-empty short-circuit) **unless** `pad_to_max_length` is
    /// set — then it pads to `max_length` like HF `padding="max_length"` (so an empty *negative*
    /// prompt for true-CFG, e.g. FLUX.1 IP-Adapter, encodes to the special-token + pad sequence
    /// diffusers produces, not a 0-length tensor that crashes the transformer).
    pub fn tokenize(&self, prompt: &str) -> Result<TokenizerOutput> {
        if prompt.is_empty() && !self.config.pad_to_max_length {
            return Ok(TokenizerOutput {
                input_ids: empty_row(),
                attention_mask: empty_row(),
            });
        }

        self.tokenize_preformatted(&self.config.chat_template.render(prompt))
    }

    /// Tokenize an **already-formatted** string (no chat-template wrapping) → `(1, L)` ids + mask,
    /// with the same truncation/padding policy as [`tokenize`](Self::tokenize). For callers that
    /// build the full templated text themselves — e.g. the Qwen-Image-Edit VL path, which
    /// interleaves an expanded run of `<|image_pad|>` tokens that depends on the image grid.
    pub fn tokenize_preformatted(&self, text: &str) -> Result<TokenizerOutput> {
        if text.is_empty() && !self.config.pad_to_max_length {
            return Ok(TokenizerOutput {
                input_ids: empty_row(),
                attention_mask: empty_row(),
            });
        }
        let encoding = self.inner.encode(text, true).map_err(tok_err)?;

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

    /// Encode `text` to a flat `Vec<i32>` of ids — no padding, no truncation, no chat-template
    /// wrapping — honoring `add_special_tokens` (the post-processor's BOS/EOS template). Used by
    /// autoregressive-LM callers that build the full prompt string themselves and need exact ids
    /// (e.g. the LTX Gemma prompt enhancer, which formats the chat template by hand and tokenizes
    /// with `add_special_tokens=false`, like the reference `processor(..., add_special_tokens=False)`).
    /// Special-token *strings* already present in `text` (e.g. `<start_of_turn>`) are still mapped to
    /// their ids regardless of the flag — the flag only governs the auto-added BOS/EOS.
    pub fn encode_ids(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(tok_err)?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Detokenize `ids` back to text via the loaded tokenizer's decoder. `skip_special_tokens`
    /// drops special tokens (BOS/EOS/turn markers) from the output, matching HF
    /// `tokenizer.decode(ids, skip_special_tokens=…)`.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(tok_err)
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
    fn qwen_instruct_no_think_appends_empty_think_block() {
        let r = ChatTemplate::QwenInstructNoThink.render("a red fox");
        assert_eq!(
            r,
            "<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
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
