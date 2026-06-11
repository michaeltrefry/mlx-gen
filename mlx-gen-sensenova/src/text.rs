//! Tokenizer loading, the `neo1_0` conversation template, special-token ids, and the (t,h,w)
//! position-index builders (sc-3186).
//!
//! SenseNova-U1 uses a Qwen2/3 byte-level BPE tokenizer (vocab 151936) with NEO-Unify's added
//! special tokens (`<img>`/`</img>`/`<IMG_CONTEXT>`, `<think>`/`</think>`, the ChatML markers). The
//! snapshot ships only `vocab.json` + `merges.txt` + `added_tokens.json`, so — mirroring the
//! Qwen-Image provider — a fast `tokenizer.json` is materialized into the snapshot by
//! `tools/build_sensenova_tokenizer.py`; [`load_tokenizer`] reads it.
//!
//! The `neo1_0` template is ChatML (the reference `conversation.py` MPT style): an optional system
//! block, the user turn, and the empty assistant turn that primes generation. Image generation
//! prepends [`SYSTEM_MESSAGE_FOR_GEN`].

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::{Error, Result};

/// NEO-Unify special-token ids (from the snapshot's `added_tokens.json`).
pub mod tokens {
    pub const ENDOFTEXT: i32 = 151643;
    pub const IM_START: i32 = 151644;
    pub const IM_END: i32 = 151645;
    pub const THINK: i32 = 151667;
    pub const THINK_END: i32 = 151668;
    pub const IMG_CONTEXT: i32 = 151669;
    /// `<img>` — the reference's `img_start_token_id`.
    pub const IMG_START: i32 = 151670;
    /// `</img>`.
    pub const IMG_END: i32 = 151671;
    pub const PAD: i32 = 151643;
}

/// The image-generation system message (verbatim from the reference `utils.SYSTEM_MESSAGE_FOR_GEN`).
pub const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands and executes ",
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you ",
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. ",
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final ",
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n",
    "Task Types:\n\nA. Text-to-Image Generation:\n",
    "- Generate a high-quality image based on the user's description.\n",
    "- Ensure visual clarity, semantic consistency, and completeness.\n",
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n",
    "B. Image Editing:\n",
    "- Use the provided image(s) as input or reference for modification or transformation.\n",
    "- The result can be an edited image or a new image based on the reference(s).\n",
    "- Preserve all unspecified attributes unless explicitly changed.\n\n",
    "General Rules:\n",
    "- For any visible text in the image, follow the language specified for the rendered text in ",
    "the user's description, not the language of the prompt. If no language is specified, use the ",
    "user's input language."
);

/// The interleaved text-image system message (verbatim from the reference
/// `examples/interleave/inference.py::DEFAULT_SYSTEM_MESSAGE`) — required for Document Studio's
/// think-mode interleave protocol or the model won't interleave correctly.
pub const INTERLEAVE_SYSTEM_MESSAGE: &str = concat!(
    "You are a multimodal assistant capable of reasoning with both text and images. You support ",
    "two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> ",
    "block and place all reasoning inside it. You MUST interleave text with generated images using ",
    "tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and ",
    "may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly ",
    "provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any ",
    "images naturally alongside the text.\n\nAfter the think block, always provide a concise, ",
    "user-facing final answer. The answer may include text, images, or both. Match the user's ",
    "language in both reasoning and the final answer."
);

/// Build the `neo1_0` ChatML prompt: optional system block + the user turn + the empty assistant
/// turn that primes generation. Mirrors the reference `conversation.py` MPT style — an empty
/// `system_message` omits the system block entirely.
pub fn build_neo1_query(prompt: &str, system_message: &str) -> String {
    let mut s = String::new();
    if !system_message.is_empty() {
        s.push_str("<|im_start|>system\n");
        s.push_str(system_message);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(prompt);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    s
}

/// Load the fast tokenizer from `<root>/tokenizer.json`. The crate builds the prompt strings itself
/// and tokenizes them with [`TextTokenizer::encode_ids`], so no chat-template wrapping is applied
/// here ([`ChatTemplate::None`]).
pub fn load_tokenizer(root: impl AsRef<Path>) -> Result<TextTokenizer> {
    let path = root.as_ref().join("tokenizer.json");
    if !path.exists() {
        return Err(Error::Msg(format!(
            "missing {}: the SenseNova-U1 snapshot ships only vocab.json + merges.txt; run \
             tools/build_sensenova_tokenizer.py to materialize the fast tokenizer.json",
            path.display()
        )));
    }
    Ok(TextTokenizer::from_file(
        path,
        TokenizerConfig {
            max_length: 32_768,
            pad_token_id: tokens::PAD,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )?)
}

/// The three position rows for a run of `len` **text** tokens: temporal = `0..len`, height = width
/// = 0 (the reference `_build_t2i_text_inputs`).
pub fn text_indexes(len: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let t = (0..len as i32).collect();
    let zeros = vec![0i32; len];
    (t, zeros.clone(), zeros)
}

/// The three position rows for a `token_h × token_w` image block placed after `text_len` text
/// tokens: temporal = `text_len` (all image tokens share one block index → bidirectional attention),
/// height = `idx / token_w`, width = `idx % token_w` (row-major; the reference
/// `_build_t2i_image_indexes`).
pub fn image_indexes(
    token_h: usize,
    token_w: usize,
    text_len: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = token_h * token_w;
    let mut t = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        t.push(text_len as i32);
        h.push((i / token_w) as i32);
        w.push((i % token_w) as i32);
    }
    (t, h, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neo1_query_empty_system_has_no_system_block() {
        let q = build_neo1_query("a fox", "");
        assert_eq!(
            q,
            "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn neo1_query_with_system_block() {
        let q = build_neo1_query("a fox", "SYS");
        assert_eq!(
            q,
            "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn text_indexes_are_causal_positions() {
        let (t, h, w) = text_indexes(4);
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 0, 0, 0]);
        assert_eq!(w, vec![0, 0, 0, 0]);
    }

    #[test]
    fn image_indexes_are_grid_positions_after_text() {
        // 2×3 grid placed after 5 text tokens.
        let (t, h, w) = image_indexes(2, 3, 5);
        assert_eq!(t, vec![5, 5, 5, 5, 5, 5]);
        assert_eq!(h, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(w, vec![0, 1, 2, 0, 1, 2]);
    }
}
