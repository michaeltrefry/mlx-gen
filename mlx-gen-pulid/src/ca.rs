//! PerceiverAttentionCA ×20 + the FLUX DiT injection (sc-3072). Port of
//! `pulid/encoders_transformer.py PerceiverAttentionCA` plus the `flux/model.py` injection schedule.
//!
//! Each `PerceiverAttentionCA` cross-attends the **image** tokens (queries) onto the IDFormer
//! `id_embedding` (keys/values): `img += id_weight · ca(id_embedding, img)`. 20 of them are injected
//! into the FLUX DiT — after every 2nd double block (10) and every 4th single block (10) — via the
//! generic `mlx_gen_flux::transformer::DitImageInjector` seam (no PuLID code in the flux crate).
//!
//! `ca_idx` runs 0..9 across the double injections then 10..19 across the single injections, exactly
//! matching the reference's shared running counter. LayerNorm ε=1e-5 (nn.LayerNorm default); the
//! factored `(q·s)@(k·s)` + f32-softmax attention is reproduced by MLX SDPA(scale=dim_head^-0.5).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{matmul, multiply, split};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_gen_flux::transformer::DitImageInjector;

const EPS: f32 = 1e-5;

fn join(p: &str, leaf: &str) -> String {
    format!("{p}.{leaf}")
}

/// Cross-attention block: q from `latents` (image tokens, dim=3072), k/v from `x` (id_embedding,
/// kv_dim=2048). All linears bias-free.
///
/// **Shares its structure with [`crate::idformer`]'s `PerceiverAttention`** (same `norm1`/`norm2` +
/// `to_q`/`to_kv`/`to_out` weight keys, head split, `EPS = 1e-5`, SDPA `scale = dim_head^-0.5`, and
/// out-projection). They are kept as separate 1:1 mirrors of the two distinct upstream Python classes
/// (F-085). The **only** behavioral difference is the k/v source: here k/v come from `x` alone (the
/// id_embedding), whereas IDFormer's variant uses `cat(ctx, latents)`. Any change to the shared
/// plumbing — EPS, the SDPA scale, the bias-free linears, a future quantization path — must be applied
/// to **both** to preserve parity.
pub struct PerceiverAttentionCA {
    norm1_w: Array, // over kv_dim (id_embedding)
    norm1_b: Array,
    norm2_w: Array, // over dim (image)
    norm2_b: Array,
    to_q: Array,
    to_kv: Array,
    to_out: Array,
    heads: i32,
    dim_head: i32,
}

impl PerceiverAttentionCA {
    pub fn from_weights(w: &Weights, prefix: &str, heads: i32, dim_head: i32) -> Result<Self> {
        Ok(Self {
            norm1_w: w.require(&join(prefix, "norm1.weight"))?.clone(),
            norm1_b: w.require(&join(prefix, "norm1.bias"))?.clone(),
            norm2_w: w.require(&join(prefix, "norm2.weight"))?.clone(),
            norm2_b: w.require(&join(prefix, "norm2.bias"))?.clone(),
            to_q: w.require(&join(prefix, "to_q.weight"))?.clone(),
            to_kv: w.require(&join(prefix, "to_kv.weight"))?.clone(),
            to_out: w.require(&join(prefix, "to_out.weight"))?.clone(),
            heads,
            dim_head,
        })
    }

    /// `id_embedding`: `[B, 32, kv_dim]`; `img`: `[B, S, dim]` → residual `[B, S, dim]`.
    pub fn forward(&self, id_embedding: &Array, img: &Array) -> Result<Array> {
        let x = layer_norm(id_embedding, Some(&self.norm1_w), Some(&self.norm1_b), EPS)?;
        let lat = layer_norm(img, Some(&self.norm2_w), Some(&self.norm2_b), EPS)?;
        let (b, s) = (lat.shape()[0], lat.shape()[1]);
        let n_kv = x.shape()[1];
        let (h, hd) = (self.heads, self.dim_head);

        let q = matmul(&lat, self.to_q.t())?; // [B, S, inner]
        let kv = matmul(&x, self.to_kv.t())?; // [B, 32, inner*2]
        let parts = split(&kv, 2, 2)?;
        let to_heads = |t: &Array, n: i32| -> Result<Array> {
            Ok(t.reshape(&[b, n, h, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(&q, s)?;
        let k = to_heads(&parts[0], n_kv)?;
        let v = to_heads(&parts[1], n_kv)?;

        let scale = (hd as f32).powf(-0.5);
        let attn = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let out = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, h * hd])?;
        Ok(matmul(&out, self.to_out.t())?)
    }
}

/// The 20 CA modules + the bound `id_embedding`/`id_weight`, implementing the FLUX DiT injection
/// schedule. Build it for a given id_embedding (from the IDFormer) and inject during the denoise
/// forward via [`mlx_gen_flux::transformer::FluxTransformer::forward_injected`].
pub struct PulidCa {
    ca: Vec<PerceiverAttentionCA>,
    id_embedding: Array,
    id_weight: f32,
    double_interval: usize,
    single_interval: usize,
    n_double_inject: usize,
}

impl PulidCa {
    /// `prefix` = `"pulid_ca"`. `num_double_blocks`/`num_single_blocks` are the FLUX DiT block counts
    /// (19 / 38) — used to assert the module count and compute the shared ca_idx base.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        id_embedding: Array,
        id_weight: f32,
        num_double_blocks: usize,
        num_single_blocks: usize,
    ) -> Result<Self> {
        let double_interval = 2usize;
        let single_interval = 4usize;
        let n_double_inject = num_double_blocks.div_ceil(double_interval);
        let n_single_inject = num_single_blocks.div_ceil(single_interval);
        let num_ca = n_double_inject + n_single_inject;
        let ca = (0..num_ca)
            .map(|i| PerceiverAttentionCA::from_weights(w, &join(prefix, &i.to_string()), 16, 128))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ca,
            id_embedding,
            id_weight,
            double_interval,
            single_interval,
            n_double_inject,
        })
    }

    pub fn num_ca(&self) -> usize {
        self.ca.len()
    }

    /// The CA module index injected after double block `idx` (when `idx % double_interval == 0`).
    fn double_ca_idx(&self, idx: usize) -> Option<usize> {
        idx.is_multiple_of(self.double_interval)
            .then(|| idx / self.double_interval)
    }

    /// The CA module index injected after single block `idx` (when `idx % single_interval == 0`),
    /// continuing the shared counter after the double injections.
    fn single_ca_idx(&self, idx: usize) -> Option<usize> {
        idx.is_multiple_of(self.single_interval)
            .then(|| self.n_double_inject + idx / self.single_interval)
    }

    fn scaled(&self, r: Array) -> Result<Array> {
        Ok(multiply(&r, &scalar(self.id_weight).as_dtype(r.dtype())?)?)
    }
}

impl DitImageInjector for PulidCa {
    fn after_double(&self, block_idx: usize, img_hidden: &Array) -> Result<Option<Array>> {
        if self.id_weight == 0.0 {
            return Ok(None); // bit-identical to plain FLUX
        }
        match self.double_ca_idx(block_idx) {
            Some(ca_idx) => Ok(Some(
                self.scaled(self.ca[ca_idx].forward(&self.id_embedding, img_hidden)?)?,
            )),
            None => Ok(None),
        }
    }

    fn injects_after_single(&self, block_idx: usize) -> bool {
        self.id_weight != 0.0 && block_idx.is_multiple_of(self.single_interval)
    }

    fn after_single(&self, block_idx: usize, img_tokens: &Array) -> Result<Option<Array>> {
        if self.id_weight == 0.0 {
            return Ok(None);
        }
        match self.single_ca_idx(block_idx) {
            Some(ca_idx) => Ok(Some(
                self.scaled(self.ca[ca_idx].forward(&self.id_embedding, img_tokens)?)?,
            )),
            None => Ok(None),
        }
    }
}
