//! The optional **PromptReasoner** (sc-3176) — the local gpt-oss `generate` path of
//! `_vendor/lens/reasoner.py::PromptReasoner`. It rewrites the user prompt into a richer
//! text-to-image prompt *before* encoding. **Off by default** (the pipeline's `enable_reasoner`); the
//! OpenAI-compatible-API path is host-agnostic and needs no MLX, so this story is only the local path.
//!
//! Turning the encoder-only gpt-oss into a **generating** model adds: the full 24-layer stack + final
//! `norm` + `lm_head`, an incremental KV-cache decode ([`GptOssDecoderLayer::forward_cached`] over a
//! per-layer [`KvCache`]), greedy / temperature sampling, the harmony `reasoning_effort="low"`
//! template ([`crate::text::LensTokenizer::encode_reasoner`]), and the harmony-channel output parse
//! ([`crate::text::clean_reasoner_output`]).
//!
//! The MoE experts can be quantized (Q4/Q8, sc-3172) so the reasoner loads at the same `~12 GB` as the
//! encoder.

use mlx_rs::ops::indexing::{argmax, argmax_axis};
use mlx_rs::ops::{matmul, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};

use crate::config::GptOssConfig;
use crate::text::{LensTokenizer, HARMONY_RETURN};
use crate::text_encoder::gpt_oss::{attention_mask, GptOssDecoderLayer, KvCache};

/// Generation defaults from the vendor `PromptReasoner.__init__`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4096;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;

/// The generating gpt-oss-20b model: the full decoder stack + final norm + LM head (the
/// encoder-only [`crate::text_encoder::encoder::LensTextEncoder`] truncates the stack and drops these).
pub struct LensReasonerModel {
    embed_tokens: Array, // [vocab, hidden]
    layers: Vec<GptOssDecoderLayer>,
    final_norm: Array, // [hidden]
    lm_head: Array,    // [vocab, hidden]
    inv_freq: Array,
    attn_scaling: f32,
    sliding_window: i32,
    cfg: GptOssConfig,
}

impl LensReasonerModel {
    /// Load the full generating model from the `text_encoder` weights at `dtype`. `quant` (Q4/Q8)
    /// quantizes the MoE experts per-layer (sc-3172) so the reasoner stays `~12 GB`.
    pub fn from_weights(
        w: &Weights,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let embed_tokens = w.require("model.embed_tokens.weight")?.as_dtype(dtype)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(GptOssDecoderLayer::from_weights(
                w,
                &format!("model.layers.{i}"),
                cfg,
                dtype,
                quant,
            )?);
        }
        let (inv_freq, attn_scaling) = cfg.yarn_rope();
        Ok(Self {
            embed_tokens,
            layers,
            final_norm: w.require("model.norm.weight")?.as_dtype(dtype)?,
            lm_head: w.require("lm_head.weight")?.as_dtype(dtype)?,
            inv_freq: Array::from_slice(&inv_freq, &[inv_freq.len() as i32]),
            attn_scaling,
            sliding_window: cfg.sliding_window,
            cfg: *cfg,
        })
    }

    /// Run all layers over `hidden` `[1, T, hidden]` with the per-layer caches. `prefill` ⇒ build the
    /// per-layer causal(+sliding) mask for the `T` prompt tokens; otherwise (`T == 1` decode) every
    /// cached key is valid (`mask = None`).
    fn run_layers(
        &self,
        mut hidden: Array,
        caches: &mut [KvCache],
        position: i32,
        prefill: bool,
    ) -> Result<Array> {
        let l = hidden.shape()[1];
        for (i, layer) in self.layers.iter().enumerate() {
            let sliding = if self.cfg.is_sliding(i) {
                Some(self.sliding_window)
            } else {
                None
            };
            let mask = if prefill {
                Some(attention_mask(l, sliding, hidden.dtype())?)
            } else {
                None
            };
            hidden = layer.forward_cached(
                &hidden,
                &self.inv_freq,
                self.attn_scaling,
                position,
                &mut caches[i],
                sliding,
                mask.as_ref(),
            )?;
        }
        Ok(hidden)
    }

    /// The greedy next-token id from the **last** position of `hidden` `[1, T, hidden]`
    /// (`argmax(lm_head(norm(h_last)))`). Ties break to the lowest index (MLX `argmax` ≡ `torch.argmax`).
    fn argmax_token(&self, hidden: &Array) -> Result<i32> {
        let t = hidden.shape()[1];
        let last = if t > 1 {
            split_sections(hidden, &[t - 1], 1)?[1].clone() // [1, 1, hidden]
        } else {
            hidden.clone()
        };
        let normed = mlx_rs::fast::rms_norm(&last, &self.final_norm, self.cfg.rms_eps)?;
        let logits = matmul(&normed, self.lm_head.t())?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        let idx = argmax(&logits.reshape(&[vocab])?, None)?;
        Ok(idx.item::<u32>() as i32)
    }

    /// Teacher-forced per-position next-token argmax over the whole `input_ids` in **one** prefill
    /// forward (no decode loop): returns `pred[i] = argmax(logits at position i)`, the model's greedy
    /// next-token prediction given the true prefix `input_ids[..=i]`. Used to (a) compare to torch
    /// greedy at every position and (b) prove the incremental KV-cache decode is bit-equivalent to a
    /// full recompute.
    pub fn next_token_argmax(&self, input_ids: &[i32]) -> Result<Vec<i32>> {
        let mut caches: Vec<KvCache> = (0..self.cfg.num_layers).map(|_| KvCache::new()).collect();
        let prompt = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let hidden = self.embed_tokens.take_axis(&prompt, 0)?;
        let hidden = self.run_layers(hidden, &mut caches, 0, true)?;
        let normed = mlx_rs::fast::rms_norm(&hidden, &self.final_norm, self.cfg.rms_eps)?;
        let logits = matmul(&normed, self.lm_head.t())?; // [1, L, vocab]
        let pred = argmax_axis(&logits, 2, None)?.reshape(&[input_ids.len() as i32])?; // [L]
        Ok(pred.as_slice::<u32>().iter().map(|&i| i as i32).collect())
    }

    /// **Greedy** autoregressive generation (the parity path): prefill `input_ids`, then decode until
    /// the harmony `<|return|>` stop or `max_new_tokens`. Returns the **new** tokens (including the
    /// trailing stop, which [`clean_reasoner_output`](crate::text::clean_reasoner_output) strips) —
    /// mirroring the vendor `out_ids[0, input_len:]`.
    pub fn generate_greedy(&self, input_ids: &[i32], max_new_tokens: usize) -> Result<Vec<i32>> {
        let mut caches: Vec<KvCache> = (0..self.cfg.num_layers).map(|_| KvCache::new()).collect();

        // Prefill (positions 0..input_len).
        let prompt = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let hidden = self.embed_tokens.take_axis(&prompt, 0)?;
        let hidden = self.run_layers(hidden, &mut caches, 0, true)?;
        mlx_rs::transforms::eval([&hidden])?;
        let mut next = self.argmax_token(&hidden)?;

        // The token just produced sits at the next absolute position; each decode step feeds it there.
        let mut position = input_ids.len() as i32;
        let mut out = vec![next];
        while out.len() < max_new_tokens && next != HARMONY_RETURN {
            let tok = Array::from_slice(&[next], &[1, 1]);
            let h = self.embed_tokens.take_axis(&tok, 0)?; // [1, 1, hidden]
            let h = self.run_layers(h, &mut caches, position, false)?;
            position += 1;
            next = self.argmax_token(&h)?;
            out.push(next);
        }
        Ok(out)
    }
}

/// The local PromptReasoner: the generating model + the tokenizer (harmony reasoner template +
/// output parse). The vendor default `enable=False` is the caller's concern; this is the `enable=True`
/// local-`generate` path.
pub struct LensReasoner {
    model: LensReasonerModel,
    tokenizer: LensTokenizer,
}

impl LensReasoner {
    /// Load from a Lens snapshot dir (`text_encoder/` + `tokenizer/tokenizer.json`). `quant` keeps the
    /// reasoner at `~12 GB`.
    pub fn load(
        snapshot_dir: impl AsRef<std::path::Path>,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let root = snapshot_dir.as_ref();
        let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
        let w = Weights::from_dir(root.join("text_encoder"))?;
        let model = LensReasonerModel::from_weights(&w, &GptOssConfig::lens(), dtype, quant)?;
        Ok(Self { model, tokenizer })
    }

    /// Refine one prompt via the local gpt-oss (greedy decode). `date` fills the harmony preamble
    /// (`Current date:`). Returns the cleaned final-channel rewrite, or the original `prompt` when the
    /// reasoner produced no usable final text (the vendor `clean_text_out or prompt`).
    pub fn refine(&self, prompt: &str, max_new_tokens: usize, date: &str) -> Result<String> {
        let input_ids = self.tokenizer.encode_reasoner(prompt, date)?;
        if input_ids.is_empty() {
            return Err(Error::Msg("lens reasoner: empty tokenization".into()));
        }
        let new_tokens = self.model.generate_greedy(&input_ids, max_new_tokens)?;
        let raw = self.tokenizer.decode(&new_tokens)?;
        let cleaned = crate::text::clean_reasoner_output(&raw);
        Ok(if cleaned.is_empty() {
            prompt.to_string()
        } else {
            cleaned
        })
    }
}
