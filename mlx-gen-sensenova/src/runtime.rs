//! The autoregressive text-generation runtime (sc-3187) — KV-cache incremental decode, token
//! sampling, and the `_generate_think` think/no-think rollout.
//!
//! SenseNova-U1 is a *generating* LLM, so beyond a forward pass it needs an AR runtime: a prefix is
//! prefilled into a [`KvCache`](crate::qwen3::KvCache), then tokens are decoded one at a time —
//! each new token forwarded through the cached backbone at the next temporal position to produce
//! the logits for the token after it. This module ports the reference's three pieces
//! (`modeling_neo_chat.py`):
//!
//! * [`Qwen3Backbone::decode_logits`] — one cached single-token forward → next-token logits (the
//!   inner step of the reference's `self.language_model(input_ids=next_token.unsqueeze(0), …)`).
//! * [`Qwen3Backbone::append_tokens`] — splice a run of known tokens into the cache without
//!   sampling (the reference `_append_text_tokens_to_cache`; e.g. the `\n\n<img>` that follows a
//!   think block).
//! * [`Qwen3Backbone::generate`] — greedy/sampled rollout to an EOS or token budget (the runtime
//!   under `chat`/`answer_question`, sc-3191).
//! * [`Qwen3Backbone::generate_think`] — the `_generate_think` think rollout: greedy-decode a
//!   `<think>…</think>` block, then append `\n\n<img>` to the cache, leaving it primed for image
//!   generation (sc-3187's deliverable for T2I think-mode + interleave).
//!
//! Positions: text tokens advance the temporal axis by one per token (`h = w = 0`), matching the
//! reference, which sets `model.current_index = t_idx` before each step and lets the forward
//! increment it. The understanding path ([`Path::Und`]) drives text decode.

use mlx_rs::ops::indexing::argmax as argmax_device;
use mlx_rs::Array;

use mlx_gen::Result;

use crate::qwen3::{KvCache, Path, Qwen3Backbone};

/// How the next token is chosen from a logits row.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Argmax — the reference `_generate_think` rollout and the deterministic chat path.
    Greedy,
    /// Temperature + nucleus (top-p) + top-k sampling. `top_p`/`top_k` of `1.0`/`0` disable that
    /// stage; `temperature` must be `> 0`.
    Sample {
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    },
}

impl Sampler {
    /// Pick a token id from a `[vocab]` logits row, advancing `rng` for the stochastic variants.
    fn pick(&self, logits: &[f32], rng: &mut SplitMix64) -> i32 {
        match *self {
            Sampler::Greedy => argmax(logits),
            Sampler::Sample {
                temperature,
                top_p,
                top_k,
                ..
            } => sample(logits, temperature, top_p, top_k, rng),
        }
    }

    fn seed(&self) -> u64 {
        match *self {
            Sampler::Greedy => 0,
            Sampler::Sample { seed, .. } => seed,
        }
    }
}

/// The result of a [`Qwen3Backbone::generate_think`] rollout.
pub struct ThinkRollout {
    /// The think-block token ids (everything the model emitted up to and including `</think>`, or
    /// up to EOS). Decode with the tokenizer for the human-readable reasoning text.
    pub think_token_ids: Vec<i32>,
    /// The temporal index after the rollout and the appended `\n\n<img>` — the `text_len` the first
    /// image block is placed after.
    pub t_idx: i32,
}

impl Qwen3Backbone {
    /// One cached single-token forward on the understanding path: embed `token`, run it at temporal
    /// position `pos_t` (`h = w = 0`), persist its K/V, and return the `[vocab]` next-token logits.
    pub fn decode_logits(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> Result<Vec<f32>> {
        let ids = Array::from_slice(&[token], &[1, 1]);
        let embeds = self.embed(&ids)?;
        let hidden = self.forward_cached(&embeds, &[pos_t], &[0], &[0], Path::Und, cache, true)?;
        let logits = self.lm_head(&hidden)?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        Ok(logits.reshape(&[vocab])?.as_slice::<f32>().to_vec())
    }

    /// Like [`decode_logits`](Self::decode_logits) but reduces to the greedy next token **on device**
    /// — only the single argmax index is copied to host, not the whole `[vocab]` f32 row (~600 KB).
    /// MLX `argmax` breaks ties to the lowest index, matching the host [`argmax`] (`torch.argmax`), so
    /// the greedy stream is bit-identical (F-140).
    pub fn decode_argmax(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> Result<i32> {
        let ids = Array::from_slice(&[token], &[1, 1]);
        let embeds = self.embed(&ids)?;
        let hidden = self.forward_cached(&embeds, &[pos_t], &[0], &[0], Path::Und, cache, true)?;
        let logits = self.lm_head(&hidden)?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        let idx = argmax_device(&logits.reshape(&[vocab])?, None)?;
        Ok(idx.item::<u32>() as i32)
    }

    /// Splice a run of known tokens into the cache (no sampling), advancing the temporal axis from
    /// `t_idx`. Returns the new `t_idx`. Mirrors the reference `_append_text_tokens_to_cache`: the
    /// tokens take positions `t_idx+1 .. t_idx+len` (`h = w = 0`), so the within-run mask is causal
    /// and they attend to all cached context.
    pub fn append_tokens(&self, ids: &[i32], t_idx: i32, cache: &mut KvCache) -> Result<i32> {
        if ids.is_empty() {
            return Ok(t_idx);
        }
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let embeds = self.embed(&ids_arr)?;
        let temporal: Vec<i32> = (t_idx + 1..=t_idx + n).collect();
        let zeros = vec![0i32; ids.len()];
        self.forward_cached(&embeds, &temporal, &zeros, &zeros, Path::Und, cache, true)?;
        Ok(t_idx + n)
    }

    /// Greedy/sampled AR text rollout. `first_logits` are the prefix's last-position logits (the
    /// distribution over the first generated token); `t_idx` is the prefix's max temporal index.
    /// Decoding stops at any id in `eos` (not emitted) or after `max_new_tokens`. Returns the
    /// generated token ids. This is the runtime under `chat`/`answer_question` (sc-3191).
    pub fn generate(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
    ) -> Result<Vec<i32>> {
        // Greedy decodes argmax on device (single-index host transfer per token); sampling needs the
        // full logits row on host. Split so the common greedy path avoids the ~600 KB copy (F-140).
        if let Sampler::Greedy = sampler {
            let mut next = argmax(first_logits);
            let mut out = Vec::new();
            let mut t = t_idx;
            for _ in 0..max_new_tokens {
                if eos.contains(&next) {
                    break;
                }
                out.push(next);
                t += 1;
                next = self.decode_argmax(next, t, cache)?;
            }
            return Ok(out);
        }

        let mut rng = SplitMix64::new(sampler.seed());
        let mut logits = first_logits.to_vec();
        let mut out = Vec::new();
        let mut t = t_idx;
        for _ in 0..max_new_tokens {
            let next = sampler.pick(&logits, &mut rng);
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            t += 1;
            logits = self.decode_logits(next, t, cache)?;
        }
        Ok(out)
    }

    /// The `_generate_think` think/no-think rollout. Greedily decodes a think block from
    /// `first_logits` (the prefix's last-position logits) until `</think>` (`think_end_id`) or any
    /// `eos`, forwarding each emitted token into `cache`; on `</think>` it forwards that token too
    /// (keeping the cache aligned). It then appends `append_ids` (the tokenizer's `\n\n<img>`,
    /// `add_special_tokens=False`) so the cache is primed at the image boundary. Returns the think
    /// token ids and the post-append temporal index. Greedy-only, matching the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_think(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        think_end_id: i32,
        eos: i32,
        append_ids: &[i32],
        max_think_tokens: usize,
    ) -> Result<ThinkRollout> {
        let mut t = t_idx;
        let mut next = argmax(first_logits);
        let mut think_token_ids = Vec::new();
        for _ in 0..max_think_tokens {
            if next == eos {
                break;
            }
            if next == think_end_id {
                // Forward `</think>` so the cache includes it, then stop. No logits needed here, so
                // splice it in without an lm_head projection (F-140).
                t = self.append_tokens(&[next], t, cache)?;
                think_token_ids.push(next);
                break;
            }
            think_token_ids.push(next);
            // Greedy: argmax the next-token logits on device (single-index transfer; F-140).
            next = self.decode_argmax(next, t + 1, cache)?;
            t += 1;
        }
        t = self.append_tokens(append_ids, t, cache)?;
        Ok(ThinkRollout {
            think_token_ids,
            t_idx: t,
        })
    }
}

/// Index of the maximum logit (ties → lowest index, matching `torch.argmax`).
pub(crate) fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

/// Temperature + top-k + nucleus (top-p) sampling over a logits row.
fn sample(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut SplitMix64) -> i32 {
    let temperature = temperature.max(1e-6);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Total order: descending logit, ties broken by ascending index. This reproduces the previous
    // stable `sort_by` (logit-only) + `truncate` exactly — equal logits kept ascending-index order —
    // so the selected top-k set and its order are identical.
    let by_logit_then_index = |&a: &usize, &b: &usize| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    };

    // top-k truncation. For a small `top_k`, partition the k highest indices into `order[..k]` with
    // `select_nth` instead of sorting all ~152k indices, then sort only those k (F-140).
    let k = if top_k == 0 {
        order.len()
    } else {
        top_k.min(order.len())
    };
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, by_logit_then_index);
        order.truncate(k);
    }
    order.sort_unstable_by(by_logit_then_index);

    // Softmax (in the truncated set) at the given temperature, numerically stabilised.
    let max_logit = logits[order[0]];
    let mut probs: Vec<f32> = order
        .iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // top-p (nucleus): keep the smallest prefix whose cumulative prob ≥ top_p.
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        order.truncate(cutoff);
        probs.truncate(cutoff);
        let renorm: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= renorm;
        }
    }

    // Inverse-CDF sample.
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return order[i] as i32;
        }
    }
    order[order.len() - 1] as i32
}

/// SplitMix64 increment (the golden-ratio odd constant). Single source for the seed-advance step so
/// the value-producing constants can't drift between callers (F-133).
pub(crate) const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 — a tiny deterministic PRNG for reproducible sampling (mirrors the joycaption runtime).
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(SPLITMIX64_INCREMENT);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_breaks_ties_to_lowest_index() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = [0.1, 2.0, 0.5, 1.0];
        let mut rng = SplitMix64::new(123);
        // top_k = 1 collapses the distribution to the single max → deterministic argmax.
        for _ in 0..16 {
            assert_eq!(sample(&logits, 1.0, 1.0, 1, &mut rng), 1);
        }
    }

    /// Brute-force reference: the pre-F-140 selection (full stable sort by logit, then truncate),
    /// with the same softmax / top-p / inverse-CDF tail. The optimized `sample` (select_nth + total
    /// order) must match it exactly, including ties at the top-k boundary.
    fn sample_ref(
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        top_k: usize,
        rng: &mut SplitMix64,
    ) -> i32 {
        let temperature = temperature.max(1e-6);
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = if top_k == 0 {
            order.len()
        } else {
            top_k.min(order.len())
        };
        order.truncate(k);
        let max_logit = logits[order[0]];
        let mut probs: Vec<f32> = order
            .iter()
            .map(|&i| ((logits[i] - max_logit) / temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
        if top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            order.truncate(cutoff);
            probs.truncate(cutoff);
            let renorm: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= renorm;
            }
        }
        let r = rng.next_f32();
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                return order[i] as i32;
            }
        }
        order[order.len() - 1] as i32
    }

    #[test]
    fn select_nth_matches_full_sort_with_ties() {
        // Logits with many exact ties (including at plausible top-k boundaries), so the optimized
        // select_nth selection and the reference full-sort selection must agree index-for-index.
        let logits = [
            2.0f32, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.5, 3.0, 2.0, 0.5, 1.0, 3.0, 0.0,
        ];
        for &top_k in &[0usize, 1, 2, 3, 5, 8, 13, 15, 100] {
            for &top_p in &[1.0f32, 0.9, 0.5, 0.1] {
                for &temperature in &[1.0f32, 0.7] {
                    for seed in 0..6u64 {
                        let mut a = SplitMix64::new(seed);
                        let mut b = SplitMix64::new(seed);
                        assert_eq!(
                            sample(&logits, temperature, top_p, top_k, &mut a),
                            sample_ref(&logits, temperature, top_p, top_k, &mut b),
                            "mismatch at top_k={top_k} top_p={top_p} temp={temperature} seed={seed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sampling_is_seed_deterministic() {
        let logits = [0.2, 1.5, 0.3, 0.9, 0.1];
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
        };
        let run = || {
            let mut rng = SplitMix64::new(s.seed());
            (0..8)
                .map(|_| s.pick(&logits, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed → identical token sequence");
    }
}
