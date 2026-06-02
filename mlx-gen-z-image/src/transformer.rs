//! Z-Image DiT denoiser. Port of `ZImageTransformer.__call__`: patchify image + caption
//! (pad each to a multiple of 32, with 3-D position ids) → embed → noise/context refiners →
//! unify → main transformer stack → final layer → unpatchify, returning the negated velocity.
//!
//! Attention is full-valid everywhere (the fork builds all-ones masks), so the blocks run
//! `mask=None`. Padded token positions are set to the **learned** `x_pad_token` / `cap_pad_token`
//! embeddings (the fork's `where(pad_mask, pad_token, ·)`) — NOT zero. With no attention mask the
//! padded tokens mix into the real tokens, so their value matters; a fresh model zero-inits the
//! pad tokens (which is why a tiny random fixture can't tell zeroing from pad-token), but the
//! trained checkpoint's pad tokens are non-zero. Position ids / coord grids are computed in plain
//! Rust and asserted exact in the parity test.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::Array;

use super::context_block::ZImageContextBlock;
use super::final_layer::FinalLayer;
use super::rope_embedder::RopeEmbedder;
use super::timestep_embedder::TimestepEmbedder;
use super::transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Shape/hyperparameters of a Z-Image transformer.
#[derive(Debug, Clone)]
pub struct ZImageTransformerConfig {
    pub patch_size: i32,
    pub f_patch_size: i32,
    pub in_channels: i32,
    pub dim: i32,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: i32,
    pub norm_eps: f32,
    pub cap_feat_dim: i32,
    pub rope_theta: f32,
    pub t_scale: f32,
    pub axes_dims: Vec<i32>,
    pub axes_lens: Vec<i32>,
    pub frequency_embedding_size: i32,
}

impl ZImageTransformerConfig {
    /// The production Z-Image-turbo config (dim 3840, 30 layers).
    pub fn turbo() -> Self {
        Self {
            patch_size: 2,
            f_patch_size: 1,
            in_channels: 16,
            dim: 3840,
            n_layers: 30,
            n_refiner_layers: 2,
            n_heads: 30,
            norm_eps: 1e-5,
            cap_feat_dim: 2560,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![32, 48, 48],
            axes_lens: vec![1024, 512, 512],
            frequency_embedding_size: 256,
        }
    }

    fn block_cfg(&self) -> ZImageBlockConfig {
        ZImageBlockConfig {
            dim: self.dim,
            n_heads: self.n_heads,
            norm_eps: self.norm_eps,
        }
    }
    fn embed_key(&self) -> String {
        format!("{}-{}", self.patch_size, self.f_patch_size)
    }
}

pub struct ZImageTransformer {
    // Fields are `pub(crate)` so the ControlNet variant ([`crate::control_transformer`]) can
    // compose the base submodules into its dual-injection forward without forking the
    // parity-proven base `forward` (mirrors the fork's `ZImageControlTransformer(ZImageTransformer)`
    // inheritance).
    pub(crate) cfg: ZImageTransformerConfig,
    pub(crate) x_embedder: AdaptableLinear,
    pub(crate) cap_norm_w: Array,
    pub(crate) cap_linear: AdaptableLinear,
    pub(crate) t_embedder: TimestepEmbedder,
    pub(crate) noise_refiner: Vec<ZImageTransformerBlock>,
    pub(crate) context_refiner: Vec<ZImageContextBlock>,
    pub(crate) layers: Vec<ZImageTransformerBlock>,
    pub(crate) rope: RopeEmbedder,
    pub(crate) final_layer: FinalLayer,
    pub(crate) x_pad_token: Array,
    pub(crate) cap_pad_token: Array,
}

/// `where(keep == 1, emb, pad)` for `emb` `[N, dim]`, `keep` `[N, 1]` (1 = real, 0 = padded),
/// `pad` `[1, dim]` — set padded token positions to the learned pad-token embedding.
pub(crate) fn apply_pad(emb: &Array, keep: &Array, pad: &Array) -> Result<Array> {
    let inv = subtract(Array::from_slice(&[1.0f32], &[1]), keep)?; // 1 - keep
    Ok(add(&multiply(emb, keep)?, &multiply(pad, &inv)?)?)
}

impl ZImageTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ZImageTransformerConfig) -> Result<Self> {
        // Tolerate an empty prefix (real checkpoints are un-prefixed) without a leading dot.
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let key = cfg.embed_key();
        let bcfg = cfg.block_cfg();

        let block_vec = |base: &str, n: usize| -> Result<Vec<ZImageTransformerBlock>> {
            (0..n)
                .map(|i| ZImageTransformerBlock::from_weights(w, &p(&format!("{base}.{i}")), bcfg))
                .collect()
        };
        let context_vec = |base: &str, n: usize| -> Result<Vec<ZImageContextBlock>> {
            (0..n)
                .map(|i| {
                    ZImageContextBlock::from_weights(
                        w,
                        &p(&format!("{base}.{i}")),
                        cfg.dim,
                        cfg.n_heads,
                        cfg.norm_eps,
                    )
                })
                .collect()
        };

        Ok(Self {
            x_embedder: AdaptableLinear::dense(
                w.require(&p(&format!("all_x_embedder.{key}.weight")))?
                    .clone(),
                Some(
                    w.require(&p(&format!("all_x_embedder.{key}.bias")))?
                        .clone(),
                ),
            ),
            cap_norm_w: w.require(&p("cap_embedder.0.weight"))?.clone(),
            cap_linear: AdaptableLinear::dense(
                w.require(&p("cap_embedder.1.weight"))?.clone(),
                Some(w.require(&p("cap_embedder.1.bias"))?.clone()),
            ),
            t_embedder: TimestepEmbedder::from_weights(
                w,
                &p("t_embedder"),
                cfg.frequency_embedding_size,
            )?,
            noise_refiner: block_vec("noise_refiner", cfg.n_refiner_layers)?,
            context_refiner: context_vec("context_refiner", cfg.n_refiner_layers)?,
            layers: block_vec("layers", cfg.n_layers)?,
            rope: RopeEmbedder::new(cfg.rope_theta, &cfg.axes_dims, &cfg.axes_lens),
            final_layer: FinalLayer::from_weights(w, &p(&format!("all_final_layer.{key}")))?,
            x_pad_token: w.require(&p("x_pad_token"))?.reshape(&[1, cfg.dim])?,
            cap_pad_token: w.require(&p("cap_pad_token"))?.reshape(&[1, cfg.dim])?,
            cfg,
        })
    }

    /// Quantize every Linear in the DiT to Q4/Q8 (group_size 64), the mlx-rs equivalent of the
    /// fork's `nn.quantize(transformer, bits=…)`. That predicate (`hasattr(module,"to_quantized")`)
    /// matches *every* `nn.Linear` — here the image/caption embedders, the timestep + final
    /// layers, and every block/context-block attention, FFN, and adaLN projection. RMSNorm /
    /// LayerNorm scales and the learned pad tokens are not Linears, so they stay dense.
    ///
    /// The fork's full `nn.quantize` also covers the text encoder + VAE; the whole-model quant is
    /// wired in `model.rs::load` (sc-2532). The quantization is byte-identical to the fork's
    /// `mx.quantize` **because `AdaptableLinear::quantize` casts the weight to bf16 first** (the
    /// fork's compute dtype). Z-Image-Turbo ships an f32 transformer checkpoint; quantizing it
    /// as-loaded (f32) yields group scales ~0.13% off the fork's bf16 scales and a ~0.78% px>8
    /// base-Q8 residual (sc-2604, once misread as "source-MLX-vs-wheel toolchain"). With the bf16
    /// cast the Q8/Q4 e2e collapses to the dense floor (~0.03% px>8 @1024²) — see
    /// `tests/e2e_real_weights.rs` and `tests/q8_xemb_diag.rs`.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.x_embedder, &mut self.cap_linear] {
            lin.quantize(bits, None)?;
        }
        self.t_embedder.quantize(bits)?;
        self.final_layer.quantize(bits)?;
        for block in &mut self.noise_refiner {
            block.quantize(bits)?;
        }
        for block in &mut self.context_refiner {
            block.quantize(bits)?;
        }
        for block in &mut self.layers {
            block.quantize(bits)?;
        }
        Ok(())
    }

    /// Diagnostic accessor (sc-2604 Q8 root-cause): the image patch embedder, so a test can
    /// byte-compare its loaded quantization and run its forward in isolation against the fork.
    pub fn x_embedder(&self) -> &AdaptableLinear {
        &self.x_embedder
    }

    /// `x`: latent `(C, F, H, W)`; `cap_feats`: `(cap_len, cap_feat_dim)`; `timestep` in [0,1].
    /// Returns the latent-shaped velocity `(C, F, H, W)`.
    pub fn forward(&self, x: &Array, timestep: f32, cap_feats: &Array) -> Result<Array> {
        let t = Array::from_slice(&[timestep * self.cfg.t_scale], &[1]);
        let t_emb = self.t_embedder.forward(&t)?;

        let patched = self.patchify(x, cap_feats)?;

        // Image stream: embed -> set padded positions to x_pad_token -> noise refiner.
        let mut x_emb = self.x_embedder.forward(&patched.x_tokens)?;
        x_emb = apply_pad(&x_emb, &patched.x_keep, &self.x_pad_token)?;
        let x_freqs = self.rope.forward(&patched.x_pos_ids)?;
        let mut x_emb = x_emb.expand_dims(0)?;
        for layer in &self.noise_refiner {
            x_emb = layer.forward(&x_emb, &x_freqs, &t_emb)?;
        }

        // Caption stream: RMSNorm -> linear -> set padded to cap_pad_token -> context refiner.
        let cap_normed = rms_norm(&patched.cap_tokens, &self.cap_norm_w, self.cfg.norm_eps)?;
        let mut cap_emb = self.cap_linear.forward(&cap_normed)?;
        cap_emb = apply_pad(&cap_emb, &patched.cap_keep, &self.cap_pad_token)?;
        let cap_freqs = self.rope.forward(&patched.cap_pos_ids)?;
        let mut cap_emb = cap_emb.expand_dims(0)?;
        for layer in &self.context_refiner {
            cap_emb = layer.forward(&cap_emb, &cap_freqs)?;
        }

        // Unify and run the main stack.
        let x_len = x_emb.shape()[1];
        let mut unified = concatenate_axis(&[&x_emb, &cap_emb], 1)?;
        let unified_freqs = concatenate_axis(&[&x_freqs, &cap_freqs], 0)?;
        for layer in &self.layers {
            unified = layer.forward(&unified, &unified_freqs, &t_emb)?;
        }

        // Final layer + unpatchify (only the real image tokens survive).
        let unified = self.final_layer.forward(&unified, &t_emb)?;
        let embed_dim = unified.shape()[2];
        let head = unified
            .reshape(&[unified.shape()[1], embed_dim])? // drop batch dim (size 1)
            .take_axis(row_indices(x_len), 0)?; // (x_len, embed_dim)
        let out = self.unpatchify(&head, patched.x_size)?;
        Ok(out.multiply(Array::from_slice(&[-1.0f32], &[1]))?)
    }

    pub(crate) fn patchify(&self, image: &Array, cap_feats: &Array) -> Result<Patchified> {
        let (pf, ph, pw) = (
            self.cfg.f_patch_size,
            self.cfg.patch_size,
            self.cfg.patch_size,
        );
        let sh = image.shape();
        let (c, f, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let (ft, ht, wt) = (f / pf, h / ph, w / pw);

        // Caption: pad to a multiple of 32; pos ids = [(1+i), 0, 0]; keep-mask zeros padding.
        let cap_ori = cap_feats.shape()[0];
        let cap_pad = (-(cap_ori as i64)).rem_euclid(32) as i32;
        let cap_total = cap_ori + cap_pad;
        let cap_pos: Vec<i32> = (0..cap_total).flat_map(|i| [1 + i, 0, 0]).collect();
        let cap_keep: Vec<f32> = (0..cap_total)
            .map(|i| if i < cap_ori { 1.0 } else { 0.0 })
            .collect();
        let cap_tokens = pad_rows(cap_feats, cap_pad)?;

        // Image: patchify (C,F,H,W) -> (tokens, pF·pH·pW·C), pad to a multiple of 32.
        let tokens = image
            .reshape(&[c, ft, pf, ht, ph, wt, pw])
            .and_then(|t| t.transpose_axes(&[1, 3, 5, 2, 4, 6, 0]))
            .and_then(|t| t.reshape(&[ft * ht * wt, pf * ph * pw * c]))?;
        let img_ori = ft * ht * wt;
        let img_pad = (-(img_ori as i64)).rem_euclid(32) as i32;
        let img_total = img_ori + img_pad;
        let s0 = cap_total + 1; // image positions start after the caption block
        let mut img_pos: Vec<i32> = Vec::with_capacity((img_total * 3) as usize);
        for fi in 0..ft {
            for hi in 0..ht {
                for wi in 0..wt {
                    img_pos.extend_from_slice(&[s0 + fi, hi, wi]);
                }
            }
        }
        img_pos.extend(std::iter::repeat_n(0, (img_pad * 3) as usize)); // padded pos ids = 0
        let img_keep: Vec<f32> = (0..img_total)
            .map(|i| if i < img_ori { 1.0 } else { 0.0 })
            .collect();
        let x_tokens = pad_rows(&tokens, img_pad)?;

        Ok(Patchified {
            x_tokens,
            cap_tokens,
            x_size: (f, h, w),
            x_pos_ids: Array::from_slice(&img_pos, &[img_total, 3]),
            cap_pos_ids: Array::from_slice(&cap_pos, &[cap_total, 3]),
            x_keep: Array::from_slice(&img_keep, &[img_total, 1]),
            cap_keep: Array::from_slice(&cap_keep, &[cap_total, 1]),
        })
    }

    pub(crate) fn unpatchify(&self, x: &Array, size: (i32, i32, i32)) -> Result<Array> {
        let (pf, ph, pw) = (
            self.cfg.f_patch_size,
            self.cfg.patch_size,
            self.cfg.patch_size,
        );
        let (f, h, w) = size;
        let oc = self.cfg.in_channels;
        let ori = (f / pf) * (h / ph) * (w / pw);
        Ok(x.take_axis(row_indices(ori), 0)? // drop any padding tokens
            .reshape(&[f / pf, h / ph, w / pw, pf, ph, pw, oc])?
            .transpose_axes(&[6, 0, 3, 1, 4, 2, 5])?
            .reshape(&[oc, f, h, w])?)
    }
}

pub(crate) struct Patchified {
    pub(crate) x_tokens: Array,
    pub(crate) cap_tokens: Array,
    pub(crate) x_size: (i32, i32, i32),
    pub(crate) x_pos_ids: Array,
    pub(crate) cap_pos_ids: Array,
    pub(crate) x_keep: Array,
    pub(crate) cap_keep: Array,
}

/// `[0, 1, …, n-1]` as an int32 index array for `take_axis`.
pub(crate) fn row_indices(n: i32) -> Array {
    Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n])
}

/// Append `pad` zero rows to a 2-D `(N, D)` array (padded positions are zeroed post-embed
/// anyway, so the pre-embed value is irrelevant — zeros match the fork's repeat-then-zero).
pub(crate) fn pad_rows(a: &Array, pad: i32) -> Result<Array> {
    if pad <= 0 {
        return Ok(a.clone());
    }
    let d = a.shape()[1];
    let zeros = Array::from_slice(&vec![0f32; (pad * d) as usize], &[pad, d]);
    Ok(concatenate_axis(&[a, &zeros], 0)?)
}

/// The Z-Image adapter key→module map — the Rust analog of the fork's `ZImageLoRAMapping`. Adapter
/// files address modules by their **trained (diffusers) path**; this routes those paths to the
/// crate's module tree, covering the full fork target surface: the three per-layer stacks
/// (`layers` / `noise_refiner` / `context_refiner`) and the six global targets. Per-block routing
/// is delegated to the block hosts (`attention.to_q/k/v`, `attention.to_out.0`,
/// `feed_forward.w1/w2/w3`, `adaLN_modulation.0`). Trained-file vs internal naming differences are
/// reconciled here (`all_x_embedder.{p}-{pf}`→`x_embedder`, `cap_embedder.1`→`cap_linear`,
/// `t_embedder.mlp.{0,2}`→`linear{1,2}`, `all_final_layer.{p}-{pf}.{linear,adaLN_modulation.1}`).
impl AdaptableHost for ZImageTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["layers", n, rest @ ..] => self
                .layers
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["noise_refiner", n, rest @ ..] => self
                .noise_refiner
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["context_refiner", n, rest @ ..] => self
                .context_refiner
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            // Global targets. The `all_x_embedder` / `all_final_layer` patch-size suffix (e.g.
            // `2-1`) is matched as a wildcard segment.
            ["all_x_embedder", _] => Some(&mut self.x_embedder),
            ["cap_embedder", "1"] => Some(&mut self.cap_linear),
            ["t_embedder", rest @ ..] => self.t_embedder.adaptable_mut(rest),
            ["all_final_layer", _, rest @ ..] => self.final_layer.adaptable_mut(rest),
            _ => None,
        }
    }
}
