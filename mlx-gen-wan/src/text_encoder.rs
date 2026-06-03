//! UMT5-XXL text encoder — port of `models/wan/text_encoder.py` (`T5Encoder` and friends) plus the
//! `_clean_text` / `encode_text` orchestration from `models/wan/loading.py`.
//!
//! Wan conditions on Google's **UMT5-XXL** (24 layers, dim 4096, 64 heads, dim_ffn 10240). It
//! differs from the standard HF T5 in two ways the port must honor:
//!   1. **Per-layer relative-position bias** (`shared_pos=False`): every block owns its own
//!      `[num_buckets, num_heads]` bucket-embedding table (24 distinct tables), rather than sharing
//!      block-0's. The bucket *grid* is identical across layers, so it is computed once and only the
//!      per-layer table lookup differs.
//!   2. **Gated-GELU FFN** named `gate_proj` / `fc1` / `fc2`: `fc2(fc1(x) · gelu_tanh(gate_proj(x)))`.
//!
//! The whole encoder runs **f32** (the reference upcasts every weight to f32 and computes the
//! attention softmax in f32 — unscaled T5 logits can be large, so bf16 softmax loses precision).
//! We keep the large Linear weights as loaded (bf16) and run f32 activations: `matmul(f32, bf16)`
//! promotes to an f32 GEMM, which is bit-identical to the reference's explicit-f32 weights (bf16→f32
//! is lossless) and is the same proven pattern the FLUX T5 path uses. The tiny norm / position-bias
//! tables are upcast to f32 so `fast::rms_norm` and the bias add see f32 operands.
//!
//! Bit-exactness target = the `mlx_video` reference (itself MLX), so `T5LayerNorm` maps to
//! `mlx_rs::fast::rms_norm` (the reference's `mx.fast.rms_norm`) and the FFN gate to the hand-rolled
//! [`gelu_tanh`] (NOT `mlx_rs::nn::gelu_approximate`, whose `√(2/π)` constant is 1 ULP off — see its
//! doc). The encoder is verified **bit-exact** (max|Δ| = 0.0) to the reference on every test prompt.

use mlx_gen::array::scalar;
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, matmul, multiply, power, softmax_axis, subtract, tanh};
use mlx_rs::{Array, Dtype};

use crate::config::WanModelConfig;

/// The additive value the reference uses to mask padding keys (`F.softmax`'s `dtype.min` analogue;
/// `mx.where(mask == 0, -3.389e38, 0.0)`). Large enough that `exp` underflows to exactly 0.
const MASK_FILL: f32 = -3.389e38;

/// Build the tokenizer policy for the UMT5-XXL encoder: encode the (cleaned) prompt verbatim,
/// right-truncate + pad to `text_len` with pad id 0, emit the attention mask. The HF
/// `google/umt5-xxl` `tokenizer.json` post-processor appends the `</s>` (id 1) EOS and adds no BOS.
pub fn umt5_tokenizer_config(text_len: usize) -> TokenizerConfig {
    TokenizerConfig {
        max_length: text_len,
        pad_token_id: 0,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    }
}

/// UMT5-XXL encoder (Wan text conditioning).
pub struct Umt5Encoder {
    token_embedding: Array, // [vocab, dim], as loaded (bf16); gathered rows are cast to f32.
    blocks: Vec<Umt5Block>,
    final_norm_w: Array, // [dim], f32
    num_heads: usize,
    head_dim: usize,
    num_buckets: usize,
    eps: f32,
}

struct Umt5Block {
    norm1_w: Array, // f32
    q: Array,       // [dim, dim] linear weight (no bias), as loaded
    k: Array,
    v: Array,
    o: Array,
    norm2_w: Array, // f32
    gate_proj: Array,
    fc1: Array,
    fc2: Array,
    pos_embedding: Array, // [num_buckets, num_heads], f32
}

impl Umt5Encoder {
    /// Build from the converted `t5_encoder.safetensors` weights (the MLX-layout keys
    /// `token_embedding.weight`, `blocks.{i}.{norm1,norm2}.weight`, `blocks.{i}.attn.{q,k,v,o}.weight`,
    /// `blocks.{i}.ffn.{gate_proj,fc1,fc2}.weight`, `blocks.{i}.pos_embedding.embedding.weight`,
    /// `norm.weight`).
    pub fn from_weights(w: &Weights, cfg: &WanModelConfig) -> Result<Self> {
        let f32 = |a: &Array| -> Result<Array> { Ok(a.as_dtype(Dtype::Float32)?) };
        let mut blocks = Vec::with_capacity(cfg.t5_num_layers);
        for i in 0..cfg.t5_num_layers {
            let p = format!("blocks.{i}");
            // Large Linear weights stay as loaded (bf16): `matmul(f32_acts, bf16_w)` promotes to the
            // exact same f32 GEMM as f32 weights — verified bit-identical this session (S1 parity is
            // byte-for-byte unchanged whether these are bf16 or f32-upcast), so bf16 halves T5 memory
            // for free. The reference upcasts everything to f32; the *values* are identical because
            // the checkpoint is bf16 (bf16→f32 is lossless). Tiny norm / position-bias tables are
            // upcast so `fast::rms_norm` and the bias add see f32 operands.
            blocks.push(Umt5Block {
                norm1_w: f32(w.require(&format!("{p}.norm1.weight"))?)?,
                q: w.require(&format!("{p}.attn.q.weight"))?.clone(),
                k: w.require(&format!("{p}.attn.k.weight"))?.clone(),
                v: w.require(&format!("{p}.attn.v.weight"))?.clone(),
                o: w.require(&format!("{p}.attn.o.weight"))?.clone(),
                norm2_w: f32(w.require(&format!("{p}.norm2.weight"))?)?,
                gate_proj: w.require(&format!("{p}.ffn.gate_proj.weight"))?.clone(),
                fc1: w.require(&format!("{p}.ffn.fc1.weight"))?.clone(),
                fc2: w.require(&format!("{p}.ffn.fc2.weight"))?.clone(),
                pos_embedding: f32(w.require(&format!("{p}.pos_embedding.embedding.weight"))?)?,
            });
        }
        Ok(Self {
            token_embedding: w.require("token_embedding.weight")?.clone(),
            blocks,
            final_norm_w: f32(w.require("norm.weight")?)?,
            num_heads: cfg.t5_num_heads,
            head_dim: cfg.t5_dim_attn / cfg.t5_num_heads,
            num_buckets: cfg.t5_num_buckets,
            eps: 1e-6,
        })
    }

    /// Run the encoder over a `[1, L]` int32 id row + `[1, L]` mask → `[1, L, dim]` f32 hidden
    /// states. `L` is the padded length; callers slice to the non-pad prefix.
    pub fn forward(&self, ids: &Array, mask: &Array) -> Result<Array> {
        let seq = ids.shape()[1];
        // Token embedding: gather rows, start the f32 activation stream.
        let mut x = self
            .token_embedding
            .take_axis(ids, 0)?
            .as_dtype(Dtype::Float32)?; // [1, L, dim]

        // Bucket grid (shared across layers) + additive padding mask (shared across layers).
        let buckets = self.bucket_grid(seq);
        let add_mask = additive_mask(mask)?; // [1, 1, 1, L] f32

        for block in &self.blocks {
            x = block.forward(
                &x,
                &buckets,
                &add_mask,
                self.num_heads,
                self.head_dim,
                self.eps,
            )?;
        }
        Ok(rms_norm(&x, &self.final_norm_w, self.eps)?)
    }

    /// Per-stage capture for parity bisection (S3 DiT reuses this template): returns the hidden
    /// state after the token embedding, after each block, and after the final norm.
    pub fn forward_capture(&self, ids: &Array, mask: &Array) -> Result<Vec<Array>> {
        let seq = ids.shape()[1];
        let mut x = self
            .token_embedding
            .take_axis(ids, 0)?
            .as_dtype(Dtype::Float32)?;
        let buckets = self.bucket_grid(seq);
        let add_mask = additive_mask(mask)?;
        let mut stages = vec![x.clone()];
        for block in &self.blocks {
            x = block.forward(
                &x,
                &buckets,
                &add_mask,
                self.num_heads,
                self.head_dim,
                self.eps,
            )?;
            stages.push(x.clone());
        }
        stages.push(rms_norm(&x, &self.final_norm_w, self.eps)?);
        Ok(stages)
    }

    /// Clean → tokenize → encode → drop padding, exactly as the reference `encode_text`. Returns the
    /// `[seq_len, dim]` non-pad prompt embedding the DiT consumes.
    pub fn encode(&self, tok: &TextTokenizer, prompt: &str) -> Result<Array> {
        let cleaned = clean_text(prompt);
        let out = tok.tokenize_preformatted(&cleaned)?;
        let seq_len: i32 = out.attention_mask.sum(None)?.item();
        let embeds = self.forward(&out.input_ids, &out.attention_mask)?;
        let dim = embeds.shape()[2];
        let flat = embeds.reshape(&[embeds.shape()[1], dim])?;
        let idx = Array::from_slice(&(0..seq_len).collect::<Vec<i32>>(), &[seq_len]);
        Ok(flat.take_axis(&idx, 0)?)
    }

    /// The `[seq, seq]` int32 bucket-index grid (`bucket[q][k] = bucket(k − q)`), built host-side.
    fn bucket_grid(&self, seq: i32) -> Array {
        let n = seq as usize;
        let mut data = Vec::with_capacity(n * n);
        for q in 0..seq {
            for k in 0..seq {
                data.push(relative_position_bucket(k - q, self.num_buckets as i32));
            }
        }
        Array::from_slice(&data, &[seq, seq])
    }
}

impl Umt5Block {
    fn forward(
        &self,
        x: &Array,
        buckets: &Array,
        add_mask: &Array,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Array> {
        // Self-attention (pre-norm, residual outside).
        let normed = rms_norm(x, &self.norm1_w, eps)?;
        let attn = self.self_attention(&normed, buckets, add_mask, num_heads, head_dim)?;
        let x = add(x, &attn)?;
        // Gated-GELU FFN (pre-norm, residual outside): fc2(fc1(h) · gelu_tanh(gate_proj(h))).
        let normed = rms_norm(&x, &self.norm2_w, eps)?;
        let gate = gelu_tanh(&linear(&normed, &self.gate_proj)?)?;
        let up = linear(&normed, &self.fc1)?;
        let ff = linear(&multiply(&up, &gate)?, &self.fc2)?;
        Ok(add(&x, &ff)?)
    }

    fn self_attention(
        &self,
        x: &Array,
        buckets: &Array,
        add_mask: &Array,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Array> {
        let seq = x.shape()[1];
        let h = num_heads as i32;
        let c = head_dim as i32;
        // [1, L, dim] → [1, heads, L, head_dim]
        let shape = |w: &Array| -> Result<Array> {
            Ok(linear(x, w)?
                .reshape(&[1, seq, h, c])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        // T5 uses NO 1/sqrt(d) scaling — compute QKᵀ and softmax in f32 (acts are already f32).
        let q = shape(&self.q)?;
        let k = shape(&self.k)?;
        let v = shape(&self.v)?;
        let scores = matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?; // [1, heads, L, L]
        let bias = self.position_bias(buckets)?; // [1, heads, L, L]
        let scores = add(&add(&scores, &bias)?, add_mask)?;
        let weights = softmax_axis(&scores, -1, true)?;
        let out = matmul(&weights, &v)? // [1, heads, L, head_dim]
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, seq, h * c])?;
        linear(&out, &self.o)
    }

    /// Per-layer relative-position bias: `embedding[buckets]` → `[1, heads, L, L]`.
    fn position_bias(&self, buckets: &Array) -> Result<Array> {
        let seq = buckets.shape()[0];
        let flat = buckets.reshape(&[seq * seq])?;
        let embeds = self.pos_embedding.take_axis(&flat, 0)?; // [L*L, heads]
        let heads = embeds.shape()[1];
        Ok(embeds
            .reshape(&[seq, seq, heads])?
            .transpose_axes(&[2, 0, 1])? // [heads, L, L]
            .expand_dims(0)?) // [1, heads, L, L]
    }
}

/// `y = x · Wᵀ` for a `[out, in]` weight with no bias (T5 linears). `x` is f32, `w` is as-loaded
/// (bf16) — `matmul` promotes to an f32 GEMM, bit-identical to the reference's f32 weights.
fn linear(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

/// GELU tanh approximation, **bit-exact** to MLX-Python's `nn.GELU(approx="tanh")`:
/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
///
/// Deliberately hand-rolled instead of `mlx_rs::nn::gelu_approximate`: mlx-rs computes the `√(2/π)`
/// constant with an f32 MLX `sqrt` op, whereas MLX-Python computes it in **f64 on the host** then
/// casts to f32 — a 1-ULP difference in the constant. That seeds a ~5e-7 per-element gap which the
/// 24 unscaled-attention layers amplify into a ~1e-3 end-to-end divergence. Computing the constant
/// in f64 here collapses it: with this form the per-op floor (matmul / rms_norm / softmax / gelu)
/// is all 0.0, so the encoder is bit-exact to the reference. (`x³` via integer-exponent `power`, as
/// both the reference and mlx-rs do.)
pub(crate) fn gelu_tanh(x: &Array) -> Result<Array> {
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, scalar(0.044_715))?)?, scalar(c))?;
    let gate = add(&tanh(&inner)?, scalar(1.0))?;
    Ok(multiply(&multiply(x, scalar(0.5))?, &gate)?)
}

/// `[1, L]` int/float mask → `[1, 1, 1, L]` f32 additive mask (`0` where kept, `MASK_FILL` where
/// padded). `(mask − 1) · |MASK_FILL|` gives `0`/`MASK_FILL` for mask ∈ {1, 0}.
fn additive_mask(mask: &Array) -> Result<Array> {
    let m = mask.as_dtype(Dtype::Float32)?;
    let seq = mask.shape()[1];
    let add = multiply(&subtract(&m, scalar(1.0))?, scalar(-MASK_FILL))?;
    Ok(add.reshape(&[1, 1, 1, seq])?)
}

/// T5 bucketing for `bidirectional=True` (`num_buckets`, `max_distance=128`). Port of
/// `T5RelativeEmbedding._relative_position_bucket`; matches the FLUX T5 bucket logic.
fn relative_position_bucket(relative_position: i32, num_buckets: i32) -> i32 {
    let max_distance = 128.0_f32;
    let half = num_buckets / 2;
    let mut bucket = 0;
    let mut n = relative_position;
    if n > 0 {
        bucket += half;
    }
    n = n.abs();
    let max_exact = half / 2;
    let val = if n < max_exact {
        n
    } else {
        let log_ratio = (n as f32 / max_exact as f32).ln() / (max_distance / max_exact as f32).ln();
        let large = max_exact + (log_ratio * (half - max_exact) as f32) as i32; // trunc == floor (≥0)
        large.min(half - 1)
    };
    bucket + val
}

// ── `_clean_text` (loading.py) ─────────────────────────────────────────────────────────────────
//
// Port of the reference `ftfy.fix_text(text)` → `html.unescape(html.unescape(text))` →
// `re.sub(r"\s+", " ", text).strip()`. ftfy is reproduced for the transforms a clean UTF-8 prompt
// actually exercises (verified bit-for-bit against `ftfy.fix_text` on a 19-case battery incl. the
// full Chinese negative prompt): block-scoped fullwidth/halfwidth fold (`fix_character_width`),
// latin-ligature expansion (`fix_latin_ligatures`), quote uncurling (`uncurl_quotes`), and final
// NFC normalization. ftfy's mojibake/encoding-repair fixes only trigger on *corrupted* input and
// are out of scope (a prompt is natural-language text, not mis-decoded bytes).

/// ftfy `LIGATURES` table (FB00–FB06). FB05 is the long-s + t ligature (`ſt`), not `st`.
const LIGATURES: &[(char, &str)] = &[
    ('\u{FB00}', "ff"),
    ('\u{FB01}', "fi"),
    ('\u{FB02}', "fl"),
    ('\u{FB03}', "ffi"),
    ('\u{FB04}', "ffl"),
    ('\u{FB05}', "\u{017F}t"),
    ('\u{FB06}', "st"),
];

/// ftfy `uncurl_quotes`: curly single/double quotes and primes → straight ASCII.
fn uncurl(ch: char) -> Option<char> {
    match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => Some('\''),
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => Some('"'),
        _ => None,
    }
}

/// Clean a prompt exactly as the reference `_clean_text` does (see module note above).
pub fn clean_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    // 1. ftfy-equivalent: fullwidth fold + latin ligatures + uncurl quotes.
    let mut folded = String::with_capacity(text.len());
    for ch in text.chars() {
        if let Some((_, rep)) = LIGATURES.iter().find(|(c, _)| *c == ch) {
            folded.push_str(rep);
        } else if ch == '\u{3000}' || ('\u{FF00}'..='\u{FFEF}').contains(&ch) {
            // `fix_character_width`: NFKC scoped to the Halfwidth/Fullwidth Forms block + the
            // ideographic space (e.g. fullwidth comma `，` U+FF0C → ASCII `,`).
            folded.extend(ch.nfkc());
        } else if let Some(rep) = uncurl(ch) {
            folded.push(rep);
        } else {
            folded.push(ch);
        }
    }
    // 2. ftfy final `normalization='NFC'`.
    let normalized: String = folded.nfc().collect();
    // 3. Double HTML unescape.
    let unescaped = html_unescape(&html_unescape(&normalized));
    // 4. Collapse whitespace runs to a single space + strip.
    unescaped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Minimal HTML entity decoder covering the entities a prompt realistically contains: the five
/// predefined XML entities, a handful of common named entities, and the full numeric forms
/// (`&#DDD;` / `&#xHHH;`). Exotic named entities (the full HTML5 table) are out of scope.
fn html_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'&' {
            // Copy this whole UTF-8 char.
            let ch_len = utf8_len(bytes[i]);
            out.push_str(&s[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        // Find the terminating ';' within a small window.
        if let Some(semi) = s[i..].find(';').filter(|&p| p <= 32) {
            let entity = &s[i + 1..i + semi];
            if let Some(decoded) = decode_entity(entity) {
                out.push(decoded);
                i += semi + 1;
                continue;
            }
        }
        out.push('&');
        i += 1;
    }
    out
}

fn decode_entity(entity: &str) -> Option<char> {
    if let Some(num) = entity.strip_prefix('#') {
        let code = if let Some(hex) = num.strip_prefix(['x', 'X']) {
            u32::from_str_radix(hex, 16).ok()?
        } else {
            num.parse::<u32>().ok()?
        };
        return char::from_u32(code);
    }
    Some(match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{00A0}',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "hellip" => '…',
        "mdash" => '—',
        "ndash" => '–',
        _ => return None,
    })
}

fn utf8_len(first: u8) -> usize {
    match first {
        b if b < 0x80 => 1,
        b if b >> 5 == 0b110 => 2,
        b if b >> 4 == 0b1110 => 3,
        _ => 4,
    }
}

/// Load the UMT5-XXL tokenizer from a `tokenizer.json` (HF `google/umt5-xxl`).
pub fn load_tokenizer(path: impl AsRef<std::path::Path>, text_len: usize) -> Result<TextTokenizer> {
    TextTokenizer::from_file(path, umt5_tokenizer_config(text_len))
        .map_err(|e| Error::Msg(format!("wan umt5 tokenizer: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_reference_edges() {
        // Same bidirectional T5 bucketing as the FLUX T5 (num_buckets=32, max_dist=128).
        assert_eq!(relative_position_bucket(0, 32), 0);
        assert_eq!(relative_position_bucket(1, 32), 17);
        assert_eq!(relative_position_bucket(-1, 32), 1);
        assert_eq!(relative_position_bucket(128, 32), 31);
        assert_eq!(relative_position_bucket(-128, 32), 15);
    }

    #[test]
    fn clean_text_collapses_whitespace_and_unescapes() {
        assert_eq!(
            clean_text("  a  cat\tplaying\n piano "),
            "a cat playing piano"
        );
        assert_eq!(
            clean_text("fox &amp; hound &lt;tag&gt;"),
            "fox & hound <tag>"
        );
    }

    #[test]
    fn clean_text_folds_fullwidth_punctuation() {
        // The load-bearing case: the Chinese negative prompt's fullwidth commas → ASCII commas.
        assert_eq!(clean_text("色调艳丽，过曝"), "色调艳丽,过曝");
        assert_eq!(clean_text("ＡＢＣ１２３"), "ABC123");
        assert_eq!(clean_text("100％ ＃tag"), "100% #tag");
    }

    #[test]
    fn clean_text_uncurls_quotes_and_expands_ligatures() {
        assert_eq!(clean_text("“curly” ‘quotes’"), "\"curly\" 'quotes'");
        assert_eq!(clean_text("ﬁle ﬂag office"), "file flag office");
        // Em-dash and ellipsis are preserved (ftfy does not touch them).
        assert_eq!(clean_text("a — b…"), "a — b…");
    }

    #[test]
    fn clean_text_handles_numeric_entities() {
        assert_eq!(clean_text("A&#38;B"), "A&B");
        assert_eq!(clean_text("&#x41;&#x42;"), "AB");
    }
}
