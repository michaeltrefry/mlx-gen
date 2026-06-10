//! CLIP byte-pair tokenizer — a faithful Rust port of the vendored Apple
//! `_vendor/mlx_sd/tokenizer.py` (`Tokenizer`) + `model_io.load_tokenizer`. SDXL ships a CLIP
//! tokenizer as `vocab.json` + `merges.txt` (no `tokenizer.json`), so the core `TextTokenizer`
//! (which loads `tokenizer.json` and pads to a fixed `max_length`) does NOT apply. The vendored
//! reference uses this **char-level** BPE (not the real CLIP byte-level BPE — a deliberate
//! "95% of cases" simplification) with **dynamic batch-max padding** and no 77-token cap; matching
//! it is what gives token-id parity with the reference path.
//!
//! Both SDXL tokenizers (`tokenizer/`, `tokenizer_2/`) ship byte-identical `vocab.json` +
//! `merges.txt`, so one instance serves both CLIP-L and OpenCLIP-bigG encoders.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use regex::Regex;

use mlx_gen::{Error, Result};

/// The vendored CLIP word-splitting pattern (`Tokenizer.pat`), case-insensitive. Order matters:
/// the two special tokens, then apostrophe contractions, then letter runs, a single digit, and a
/// run of "other" (non-space, non-letter, non-digit) characters.
const CLIP_PATTERN: &str =
    r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+";

/// Merges-file slice end: `49152 - 256 - 2 + 1` (the vendored `bpe_merges` slice `[1:48895]`),
/// dropping the `#version` header line and the trailing unused entries.
const MERGES_END: usize = 49152 - 256 - 2 + 1;

/// The padding token id the vendored `StableDiffusion._tokenize` uses (`0`, the CLIP `!` token) —
/// NOT EOS, so the encoder's `argmax(-1)` EOS-pooling still finds the real end-of-text token.
pub const PAD_ID: i32 = 0;

/// CLIP context length (`ClipTextConfig.max_length`). The position-embedding table is `[77, D]`, so a
/// prompt that tokenizes to more than this many ids would gather out-of-bounds position rows (silent
/// garbage in MLX). diffusers truncates at 77; `tokenize` does the same (F-062).
pub const MAX_LENGTH: usize = 77;

/// A loaded CLIP BPE tokenizer.
pub struct ClipBpeTokenizer {
    /// Adjacent-symbol bigram → merge rank (lower = merged first).
    bpe_ranks: HashMap<(String, String), usize>,
    /// Token string → id.
    vocab: HashMap<String, i32>,
    pat: Regex,
    bos_id: i32,
    eos_id: i32,
}

impl ClipBpeTokenizer {
    /// Load from a CLIP tokenizer directory (`<dir>/vocab.json` + `<dir>/merges.txt`).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        Self::from_files(dir.join("vocab.json"), dir.join("merges.txt"))
    }

    /// Load from explicit `vocab.json` + `merges.txt` paths.
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> Result<Self> {
        let vocab_raw = std::fs::read_to_string(vocab_path.as_ref())
            .map_err(|e| Error::Msg(format!("sdxl tokenizer: read vocab.json: {e}")))?;
        let vocab: HashMap<String, i32> = serde_json::from_str(&vocab_raw)
            .map_err(|e| Error::Msg(format!("sdxl tokenizer: parse vocab.json: {e}")))?;

        let merges_raw = std::fs::read_to_string(merges_path.as_ref())
            .map_err(|e| Error::Msg(format!("sdxl tokenizer: read merges.txt: {e}")))?;
        // `f.read().strip().split("\n")[1 : MERGES_END]` — drop the header line, cap the count.
        let lines: Vec<&str> = merges_raw.trim().split('\n').collect();
        let end = MERGES_END.min(lines.len());
        let mut bpe_ranks = HashMap::new();
        for (rank, line) in lines[1..end].iter().enumerate() {
            let mut it = line.split_whitespace();
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                bpe_ranks.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let bos_id = *vocab
            .get("<|startoftext|>")
            .ok_or_else(|| Error::Msg("sdxl tokenizer: vocab lacks <|startoftext|>".into()))?;
        let eos_id = *vocab
            .get("<|endoftext|>")
            .ok_or_else(|| Error::Msg("sdxl tokenizer: vocab lacks <|endoftext|>".into()))?;

        let pat = Regex::new(CLIP_PATTERN)
            .map_err(|e| Error::Msg(format!("sdxl tokenizer: bad pattern: {e}")))?;

        Ok(Self {
            bpe_ranks,
            vocab,
            pat,
            bos_id,
            eos_id,
        })
    }

    /// BPE-merge one whitespace-split word into its sub-token strings (the vendored `Tokenizer.bpe`).
    /// The last symbol carries the `</w>` end-of-word marker.
    fn bpe(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        // `unigrams = list(text[:-1]) + [text[-1] + "</w>"]`
        let mut unigrams: Vec<String> = Vec::with_capacity(chars.len());
        for (i, c) in chars.iter().enumerate() {
            if i + 1 == chars.len() {
                unigrams.push(format!("{c}</w>"));
            } else {
                unigrams.push(c.to_string());
            }
        }
        if unigrams.len() < 2 {
            return unigrams;
        }

        loop {
            // Find the adjacent bigram with the lowest merge rank.
            let mut best: Option<(usize, (&str, &str))> = None;
            for w in unigrams.windows(2) {
                if let Some(&rank) = self.bpe_ranks.get(&(w[0].clone(), w[1].clone())) {
                    if best.map(|(r, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, (&w[0], &w[1])));
                    }
                }
            }
            let Some((_, (a, b))) = best else { break };
            let (a, b) = (a.to_string(), b.to_string());

            // Merge every non-overlapping occurrence of (a, b), left to right.
            let mut merged: Vec<String> = Vec::with_capacity(unigrams.len());
            let mut i = 0;
            while i < unigrams.len() {
                if i + 1 < unigrams.len() && unigrams[i] == a && unigrams[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(unigrams[i].clone());
                    i += 1;
                }
            }
            unigrams = merged;
            if unigrams.len() < 2 {
                break;
            }
        }
        unigrams
    }

    /// Tokenize one prompt to CLIP token ids: lowercase + collapse whitespace, regex-split, BPE each
    /// word, map to ids, then prepend BOS and append EOS (the vendored `Tokenizer.tokenize`
    /// defaults). Errors on an out-of-vocabulary sub-token — matching the vendored `self.vocab[t]`
    /// (which would `KeyError`); the char-level BPE covers the ASCII prompt domain SDXL is used with.
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // `clean_text = regex.sub(r"\s+", " ", text.lower())`
        let lowered = text.to_lowercase();
        let clean = collapse_whitespace(&lowered);

        let mut ids = Vec::new();
        ids.push(self.bos_id);
        for m in self.pat.find_iter(&clean) {
            for sub in self.bpe(m.as_str()) {
                let id = *self.vocab.get(&sub).ok_or_else(|| {
                    Error::Msg(format!("sdxl tokenizer: token {sub:?} not in vocab"))
                })?;
                ids.push(id);
            }
        }
        ids.push(self.eos_id);
        // Cap at the CLIP context length (F-062): a longer prompt would gather out-of-bounds rows
        // from the `[77, D]` position-embedding table → silent garbage conditioning. diffusers
        // truncates the same way — keep BOS + the first content tokens, force EOS into the last slot.
        // Parity is unaffected for the <=77-token domain the goldens cover.
        if ids.len() > MAX_LENGTH {
            ids.truncate(MAX_LENGTH);
            ids[MAX_LENGTH - 1] = self.eos_id;
        }
        Ok(ids)
    }

    /// Tokenize a CFG batch the way the vendored `StableDiffusion._tokenize` does: one row for the
    /// prompt, plus (when `negative` is `Some`) a second row, both padded with [`PAD_ID`] to the
    /// batch-max length. Returns an int32 `[batch, N]` array. When `negative` is `None` the batch is
    /// `[1, N]` (CFG off).
    pub fn tokenize_batch(&self, prompt: &str, negative: Option<&str>) -> Result<Array> {
        let mut rows = vec![self.tokenize(prompt)?];
        if let Some(neg) = negative {
            rows.push(self.tokenize(neg)?);
        }
        let n = rows.iter().map(Vec::len).max().unwrap_or(0);
        let batch = rows.len() as i32;
        let mut flat = Vec::with_capacity(rows.len() * n);
        for row in &rows {
            flat.extend_from_slice(row);
            flat.extend(std::iter::repeat_n(PAD_ID, n - row.len()));
        }
        Ok(Array::from_slice(&flat, &[batch, n as i32]))
    }
}

/// Replace every run of (unicode) whitespace with a single ASCII space — the vendored
/// `regex.sub(r"\s+", " ", ...)`. Leading/trailing whitespace becomes a single space (not stripped),
/// matching the reference; the splitting pattern then ignores those spaces.
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(c);
            in_ws = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_whitespace_matches_reference() {
        assert_eq!(collapse_whitespace("a  b\t c\n"), "a b c ");
        assert_eq!(collapse_whitespace("  x"), " x");
    }

    #[test]
    fn pattern_splits_words_digits_and_punct() {
        let pat = Regex::new(CLIP_PATTERN).unwrap();
        let toks: Vec<&str> = pat
            .find_iter("a red fox, 1024!")
            .map(|m| m.as_str())
            .collect();
        // letters run together; each digit is separate; punctuation runs grouped.
        assert_eq!(toks, vec!["a", "red", "fox", ",", "1", "0", "2", "4", "!"]);
    }

    /// Build a tiny tokenizer with no merges: each single-letter word `x` → one `x</w>` token. Enough
    /// to exercise the length cap without loading the real vocab/merges.
    fn tiny_tokenizer() -> ClipBpeTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("<|startoftext|>".to_string(), 49406);
        vocab.insert("<|endoftext|>".to_string(), 49407);
        vocab.insert("a</w>".to_string(), 320);
        ClipBpeTokenizer {
            bpe_ranks: HashMap::new(),
            vocab,
            pat: Regex::new(CLIP_PATTERN).unwrap(),
            bos_id: 49406,
            eos_id: 49407,
        }
    }

    #[test]
    fn tokenize_caps_at_max_length_with_eos_last() {
        // F-062: a prompt longer than the CLIP context window must be truncated to MAX_LENGTH, not
        // produce ids that gather out of bounds from the [77, D] position table.
        let tok = tiny_tokenizer();
        // 100 single-letter words → 100 content tokens + BOS + EOS = 102 ids before the cap.
        let prompt = "a ".repeat(100);
        let ids = tok.tokenize(&prompt).unwrap();
        assert_eq!(ids.len(), MAX_LENGTH, "must cap at the context length");
        assert_eq!(ids[0], 49406, "BOS preserved");
        assert_eq!(ids[MAX_LENGTH - 1], 49407, "EOS forced into the last slot");
    }

    #[test]
    fn tokenize_short_prompt_is_unchanged() {
        // The cap must not touch prompts within the window — parity for the golden domain.
        let tok = tiny_tokenizer();
        let ids = tok.tokenize("a a a").unwrap();
        assert_eq!(ids, vec![49406, 320, 320, 320, 49407]);
    }
}
