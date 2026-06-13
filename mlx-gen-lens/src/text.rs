//! Lens text input — the gpt-oss **o200k_harmony** tokenizer + the Lens chat-template (sc-3167).
//!
//! Reproduces `LensPipeline._build_chat_inputs`: wrap the prompt in the harmony chat format (a fixed
//! `system` preamble + the `developer` instruction + the `user` prompt + an `assistant`/`analysis`
//! thinking turn), then tokenize the rendered text via the model's `tokenizer.json` (loaded through
//! the shared [`mlx_gen::tokenizer`] seam — the same HF `tokenizers` core `transformers` wraps, so
//! the ids are byte-identical).
//!
//! **txt_offset = 97.** The encoder runs the *whole* sequence (the 97-token preamble is real causal
//! context), but the DiT conditioning is only `input_ids[97:]` — the user caption + the trailing
//! assistant scaffold. The preamble's `Current date:` line is dynamic, so [`LensTokenizer::encode`]
//! takes the date as a parameter (the worker passes today's date; tests pass the golden's). The
//! preamble is always exactly [`TXT_OFFSET`] tokens, which is why the offset is a fixed constant.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig, TokenizerOutput};
use mlx_gen::Result;

/// Number of fixed harmony-preamble tokens the DiT conditioning skips (`DEFAULT_TXT_OFFSET`).
pub const TXT_OFFSET: usize = 97;

/// The Lens `developer` instruction (`_CHAT_SYSTEM`).
const SYSTEM_INSTRUCTION: &str =
    "Describe the image by detailing the color, shape, size, texture, \
     quantity, text, spatial relationships of the objects and background.";
/// The Lens `assistant`/`analysis` thinking turn (`_CHAT_ASSISTANT_THINKING`).
const ASSISTANT_THINKING: &str = "Need to generate one image according to the description.";

/// Render the harmony-formatted Lens prompt (== `_build_chat_inputs` after the `<|return|>` split,
/// pre-tokenization). `date` fills the preamble's `Current date:` line (ISO `YYYY-MM-DD`).
fn render(prompt: &str, date: &str) -> String {
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Knowledge cutoff: 2024-06\n\
         Current date: {date}\n\n\
         Reasoning: medium\n\n\
         # Valid channels: analysis, commentary, final. Channel must be included for every message.\
         <|end|><|start|>developer<|message|># Instructions\n\n\
         {SYSTEM_INSTRUCTION}\n\n\
         <|end|><|start|>user<|message|>{prompt}\
         <|end|><|start|>assistant<|channel|>analysis<|message|>{ASSISTANT_THINKING}\
         <|end|><|start|>assistant<|channel|>final<|message|>"
    )
}

/// The Lens text tokenizer: the model's `tokenizer.json` + the harmony chat-template wrapping.
pub struct LensTokenizer {
    inner: TextTokenizer,
}

impl LensTokenizer {
    /// Load from the snapshot's `tokenizer/tokenizer.json`.
    pub fn from_file(tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        // The harmony wrapping is done here in [`render`]; the core tokenizer encodes verbatim.
        let cfg = TokenizerConfig {
            max_length: 512,
            pad_token_id: 199_999, // gpt-oss `pad_token_id`
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        };
        Ok(Self {
            inner: TextTokenizer::from_file(tokenizer_json, cfg)?,
        })
    }

    /// Tokenize `prompt` into `(1, L)` `input_ids` + attention mask (mask all-`1`; a single prompt is
    /// unpadded). `date` is the ISO `YYYY-MM-DD` for the harmony preamble. The DiT consumes
    /// `ids[TXT_OFFSET..]`.
    pub fn encode(&self, prompt: &str, date: &str) -> Result<TokenizerOutput> {
        let ids = self.inner.encode_ids(&render(prompt, date), true)?;
        let mask = vec![1i32; ids.len()];
        Ok(TokenizerOutput { ids, mask })
    }

    /// Tokenize the **PromptReasoner** harmony prompt (sc-3176): the rewriter system instruction +
    /// `reasoning_effort="low"` + the `<|start|>assistant` generation prompt (the model then emits the
    /// `analysis` then `final` channel). Byte-identical to the vendor `PromptReasoner._refine_via_local`
    /// `apply_chat_template(..., add_generation_prompt=True, reasoning_effort="low")`.
    pub fn encode_reasoner(&self, prompt: &str, date: &str) -> Result<Vec<i32>> {
        Ok(self
            .inner
            .encode_ids(&render_reasoner(prompt, date), true)?)
    }

    /// Decode generated token ids back to text (special tokens kept — the harmony channel markers are
    /// needed by [`clean_reasoner_output`]).
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        Ok(self.inner.decode(&ids, false)?)
    }
}

/// The vendor `PromptReasoner.SYSTEM_PROMPT` (the rewriter instruction), verbatim.
const REASONER_SYSTEM_PROMPT: &str = "You are a prompt rewriter for a text-to-image model.\n\
Your task is to convert the user's input into a single, precise, descriptive image prompt suitable for a text-to-image model.\n\
Follow these rules strictly:\n\
\n\
1. The output must be a clear and accurate description of a single image scene, written in the style of a text-to-image prompt.\n\
\u{20}\u{20}- Do not include explanations, reasoning, commentary, or meta text.\n\
\u{20}\u{20}- Do not ask questions.\n\
\u{20}\u{20}- Do not output multiple options.\n\
\u{20}\u{20}- Do not use uncertain, speculative, or alternative wording such as \"maybe\", \"possibly\", \"perhaps\", \"or\", \"might\", or \"could\".\n\
\n\
2. Preserve the user's intended scene faithfully.\n\
\u{20}\u{20}- Do not change the objects, entities, attributes, actions, relationships, or core setting explicitly described by the user.\n\
\u{20}\u{20}- You may add reasonable visual details only when they help make the image concrete and coherent.\n\
\u{20}\u{20}- Any added details must be consistent with the user's description and must not introduce new important objects or alter the meaning.\n\
\n\
3. If the image contains many main subjects of the same kind, describe each subject in detail, including humans, animals, objects, and any other prominent elements.\n\
\u{20}\u{20}- For each subject, include its appearance, color, size, shape, material, pose, expression, and position if applicable in the scene.\n\
\u{20}\u{20}- Make sure every main subject is clearly distinguishable from the others, such as in a scene with \"4 dogs,\" describing each dog separately.\n\
\n\
4. The output must fully cover the scene implied by the user's input.\n\
\u{20}\u{20}- Include the main subjects, relevant attributes, actions, spatial relationships, environment, and visible details necessary to render the scene.\n\
\u{20}\u{20}- If the user input is already sufficiently detailed and already suitable for image generation, keep it unchanged or only make minimal edits for fluency and clarity.\n\
\n\
5. Resolve content that requires simple inference into explicit visual results when the result is unambiguous and visually representable.\n\
\u{20}\u{20}- Example: if the user says \"the answer to 2+2 is written on the blackboard\", output should explicitly describe \"the blackboard shows 2+2=4\".\n\
\u{20}\u{20}- Use only direct, necessary inference that is clearly implied by the user input.\n\
\u{20}\u{20}- Do not invent hidden facts, backstory, or ambiguous details.\n\
\n\
6. Language rule:\n\
\u{20}\u{20}- If the user input is not in English, output in the same language.\n\
\u{20}\u{20}- Otherwise, output in English.\n\
\n\
7. Output format:\n\
\u{20}\u{20}- Output exactly one final rewritten prompt.\n\
\u{20}\u{20}- Do not use bullet points, numbering, JSON, XML, Markdown, or quotation marks unless they are part of the scene itself.\n\
\n\
Your goal is to produce a prompt that is concrete, visual, faithful to the user intent, and directly usable as input to a text-to-image model.";

/// The reasoner developer-message suffix appended after [`REASONER_SYSTEM_PROMPT`] (the vendor
/// `_refine_via_local` system-prompt augmentation).
const REASONER_SUFFIX: &str =
    "Keep any reasoning private. The visible answer must contain only the final rewritten prompt.";

/// Render the harmony reasoner prompt (== `apply_chat_template([system, user], add_generation_prompt=True,
/// reasoning_effort="low")`). Same preamble as [`render`] but **`Reasoning: low`**, the rewriter
/// system instruction as the `developer` message, and the trailing bare `<|start|>assistant` generation
/// prompt (the model emits the channel itself).
fn render_reasoner(prompt: &str, date: &str) -> String {
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Knowledge cutoff: 2024-06\n\
         Current date: {date}\n\n\
         Reasoning: low\n\n\
         # Valid channels: analysis, commentary, final. Channel must be included for every message.\
         <|end|><|start|>developer<|message|># Instructions\n\n\
         {REASONER_SYSTEM_PROMPT}\n\n\
         {REASONER_SUFFIX}\n\n\
         <|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant"
    )
}

/// Harmony stop token id — the assistant turn terminates with `<|return|>` (the model's `eos`).
pub const HARMONY_RETURN: i32 = 200002;
/// Harmony `<|end|>` — also terminates a message.
pub const HARMONY_END: i32 = 200007;

/// Port of the vendor `reasoner._clean_reasoner_output`: pull the **final**-channel text out of the
/// raw harmony decode, drop any `<think>` block + residual harmony control tokens, and tidy
/// (code-fence / surrounding-quote strip, whitespace collapse). Returns `""` if the output is only an
/// `analysis` preamble (the caller then falls back to the original prompt).
pub fn clean_reasoner_output(raw: &str) -> String {
    let mut text = raw.trim().to_string();

    // Prefer the LAST `final`-channel message; else the last `<|message|>`; cut at the terminator.
    let final_marker = "<|channel|>final<|message|>";
    if let Some(pos) = text.rfind(final_marker) {
        text = text[pos + final_marker.len()..].to_string();
    } else if let Some(pos) = text.rfind("<|message|>") {
        text = text[pos + "<|message|>".len()..].to_string();
    }
    for term in ["<|return|>", "<|end|>"] {
        if let Some(pos) = text.find(term) {
            text = text[..pos].to_string();
        }
    }

    // Drop a <think>...</think> block (and anything before a closing </think>).
    text = strip_think(&text);

    // Strip residual harmony control tokens.
    for token in [
        "<|channel|>analysis<|message|>",
        "<|start|>assistant<|channel|>analysis<|message|>",
        "<|channel|>final<|message|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "<|start|>assistant<|message|>",
        "<|start|>assistant",
        "<|channel|>",
        "<|return|>",
        "<|end|>",
        "<|endoftext|>",
        "<|im_end|>",
    ] {
        text = text.replace(token, "");
    }

    let mut text = text.trim().to_string();
    // An output that is only an `analysis` preamble carries no usable prompt.
    let low = text.to_lowercase();
    if low.starts_with("analysis") || low.starts_with("assistant analysis") {
        return String::new();
    }
    // Strip a ```…``` code fence.
    if text.starts_with("```") && text.ends_with("```") {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 3 {
            text = lines[1..lines.len() - 1].join("\n").trim().to_string();
        }
    }
    // Strip surrounding double quotes.
    if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
        text = text[1..text.len() - 1].trim().to_string();
    }
    // Collapse internal whitespace runs to single spaces (`" ".join(text.split())`).
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove a `<think>...</think>` block; if a stray `</think>` remains, keep only what follows it
/// (case-insensitive), mirroring the vendor `THINK_BLOCK_RE` + the `</think>` split.
fn strip_think(text: &str) -> String {
    let mut out = text.to_string();
    let lower = out.to_lowercase();
    if let (Some(s), Some(e)) = (lower.find("<think>"), lower.find("</think>")) {
        if e >= s {
            out = format!("{}{}", &out[..s], &out[e + "</think>".len()..]);
        }
    }
    let lower = out.to_lowercase();
    if let Some(pos) = lower.rfind("</think>") {
        out = out[pos + "</think>".len()..].to_string();
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoner_template_shape() {
        // The reasoner render is the harmony preamble with `Reasoning: low`, the rewriter system
        // instruction as the `developer` message, and the bare `<|start|>assistant` generation prompt.
        let r = render_reasoner("a cat", "2026-06-13");
        assert!(r.starts_with("<|start|>system<|message|>You are ChatGPT"));
        assert!(r.contains("Current date: 2026-06-13"));
        assert!(r.contains("\nReasoning: low\n"));
        assert!(
            r.contains("# Instructions\n\nYou are a prompt rewriter for a text-to-image model.")
        );
        assert!(r.contains("<|start|>user<|message|>a cat<|end|>"));
        assert!(r.ends_with("<|start|>assistant")); // generation prompt — no channel/message yet
    }

    #[test]
    fn clean_extracts_final_channel() {
        // The model emits an analysis turn then a final turn; we keep only the final text.
        let raw = "<|channel|>analysis<|message|>The user wants a cat. I'll add detail.<|end|>\
                   <|start|>assistant<|channel|>final<|message|>a fluffy orange cat sitting on a windowsill<|return|>";
        assert_eq!(
            clean_reasoner_output(raw),
            "a fluffy orange cat sitting on a windowsill"
        );
    }

    #[test]
    fn clean_strips_think_and_quotes_and_fences() {
        assert_eq!(
            clean_reasoner_output("<think>reasoning here</think>\"a red car on a road\""),
            "a red car on a road"
        );
        assert_eq!(
            clean_reasoner_output("```\na lone tree in a field\n```"),
            "a lone tree in a field"
        );
        // Internal whitespace collapses.
        assert_eq!(
            clean_reasoner_output("a   cat   on\na  mat"),
            "a cat on a mat"
        );
    }

    #[test]
    fn clean_analysis_only_is_empty() {
        // When the *cleaned* text itself begins with an `analysis` marker (no final channel ever
        // emitted), it carries no usable final prompt → "" (the vendor's `re.match(analysis…)` guard;
        // the caller then falls back to the original prompt).
        assert_eq!(clean_reasoner_output("analysis: still thinking"), "");
        assert_eq!(
            clean_reasoner_output("assistant analysis the user asked for a cat"),
            ""
        );
    }
}
