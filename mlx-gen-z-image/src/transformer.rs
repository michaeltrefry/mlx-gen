//! Z-Image DiT denoiser. Port of `ZImageTransformer.__call__`: patchify image + caption
//! (pad each to a multiple of 32, with 3-D position ids) → embed → noise/context refiners →
//! unify → main transformer stack → final layer → unpatchify, returning the negated velocity.
//!
//! Attention is full-valid everywhere (the fork builds all-ones masks), so the blocks run
//! `mask=None`. Padded tokens are zero embeddings (the fork's `where(pad, 0, ·)`), which we
//! apply as a multiplicative keep-mask after embedding; their pre-embed values are irrelevant,
//! so patchify pads with zeros instead of repeat-last. Position ids / coord grids are computed
//! in plain Rust and asserted exact in the parity test.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{concatenate_axis, multiply};
use mlx_rs::Array;

use super::context_block::ZImageContextBlock;
use super::final_layer::FinalLayer;
use super::rope_embedder::RopeEmbedder;
use super::timestep_embedder::TimestepEmbedder;
use super::transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
use mlx_gen::nn::linear;
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
    cfg: ZImageTransformerConfig,
    x_embedder_w: Array,
    x_embedder_b: Array,
    cap_norm_w: Array,
    cap_linear_w: Array,
    cap_linear_b: Array,
    t_embedder: TimestepEmbedder,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageContextBlock>,
    layers: Vec<ZImageTransformerBlock>,
    rope: RopeEmbedder,
    final_layer: FinalLayer,
}

impl ZImageTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ZImageTransformerConfig) -> Result<Self> {
        let p = |s: &str| format!("{prefix}.{s}");
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
            x_embedder_w: w
                .require(&p(&format!("all_x_embedder.{key}.weight")))?
                .clone(),
            x_embedder_b: w
                .require(&p(&format!("all_x_embedder.{key}.bias")))?
                .clone(),
            cap_norm_w: w.require(&p("cap_embedder.0.weight"))?.clone(),
            cap_linear_w: w.require(&p("cap_embedder.1.weight"))?.clone(),
            cap_linear_b: w.require(&p("cap_embedder.1.bias"))?.clone(),
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
            cfg,
        })
    }

    /// `x`: latent `(C, F, H, W)`; `cap_feats`: `(cap_len, cap_feat_dim)`; `timestep` in [0,1].
    /// Returns the latent-shaped velocity `(C, F, H, W)`.
    pub fn forward(&self, x: &Array, timestep: f32, cap_feats: &Array) -> Result<Array> {
        let t = Array::from_slice(&[timestep * self.cfg.t_scale], &[1]);
        let t_emb = self.t_embedder.forward(&t)?;

        let patched = self.patchify(x, cap_feats);

        // Image stream: embed -> zero padded positions -> noise refiner.
        let mut x_emb = linear(&patched.x_tokens, &self.x_embedder_w, &self.x_embedder_b)?;
        x_emb = multiply(&x_emb, &patched.x_keep)?;
        let x_freqs = self.rope.forward(&patched.x_pos_ids)?;
        let mut x_emb = x_emb.expand_dims(0)?;
        for layer in &self.noise_refiner {
            x_emb = layer.forward(&x_emb, &x_freqs, &t_emb)?;
        }

        // Caption stream: RMSNorm -> linear -> zero padded -> context refiner.
        let cap_normed = rms_norm(&patched.cap_tokens, &self.cap_norm_w, self.cfg.norm_eps)?;
        let mut cap_emb = linear(&cap_normed, &self.cap_linear_w, &self.cap_linear_b)?;
        cap_emb = multiply(&cap_emb, &patched.cap_keep)?;
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

    fn patchify(&self, image: &Array, cap_feats: &Array) -> Patchified {
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
        let cap_tokens = pad_rows(cap_feats, cap_pad);

        // Image: patchify (C,F,H,W) -> (tokens, pF·pH·pW·C), pad to a multiple of 32.
        let tokens = image
            .reshape(&[c, ft, pf, ht, ph, wt, pw])
            .and_then(|t| t.transpose_axes(&[1, 3, 5, 2, 4, 6, 0]))
            .and_then(|t| t.reshape(&[ft * ht * wt, pf * ph * pw * c]))
            .expect("patchify reshape");
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
        let x_tokens = pad_rows(&tokens, img_pad);

        Patchified {
            x_tokens,
            cap_tokens,
            x_size: (f, h, w),
            x_pos_ids: Array::from_slice(&img_pos, &[img_total, 3]),
            cap_pos_ids: Array::from_slice(&cap_pos, &[cap_total, 3]),
            x_keep: Array::from_slice(&img_keep, &[img_total, 1]),
            cap_keep: Array::from_slice(&cap_keep, &[cap_total, 1]),
        }
    }

    fn unpatchify(&self, x: &Array, size: (i32, i32, i32)) -> Result<Array> {
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

struct Patchified {
    x_tokens: Array,
    cap_tokens: Array,
    x_size: (i32, i32, i32),
    x_pos_ids: Array,
    cap_pos_ids: Array,
    x_keep: Array,
    cap_keep: Array,
}

/// `[0, 1, …, n-1]` as an int32 index array for `take_axis`.
fn row_indices(n: i32) -> Array {
    Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n])
}

/// Append `pad` zero rows to a 2-D `(N, D)` array (padded positions are zeroed post-embed
/// anyway, so the pre-embed value is irrelevant — zeros match the fork's repeat-then-zero).
fn pad_rows(a: &Array, pad: i32) -> Array {
    if pad <= 0 {
        return a.clone();
    }
    let d = a.shape()[1];
    let zeros = Array::from_slice(&vec![0f32; (pad * d) as usize], &[pad, d]);
    concatenate_axis(&[a, &zeros], 0).expect("pad concat")
}
