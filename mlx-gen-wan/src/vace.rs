//! Wan-VACE transformer — native MLX port of diffusers `WanVACETransformer3DModel`
//! (`models/transformers/transformer_wan_vace.py`) (epic 3040 / sc-3388, S1).
//!
//! **VACE is purely additive on the base Wan DiT.** The base `WanTransformer3DModel` block math is
//! unchanged (self-attn with qk-RMSNorm + 3-axis RoPE, text cross-attn, adaLN-6vec modulation,
//! gated-GELU FFN, modulated head — all already validated in [`crate::transformer`]); VACE adds
//! (a) one `vace_patch_embedding` (a 96-ch patchify→linear), (b) `len(vace_layers)`
//! [`WanVaceBlock`]s that produce per-layer "hints" from the control latent, and (c) hint injection
//! `hidden_states += proj_out(control)·scale` at each main layer in `vace_layers`.
//!
//! ## Why a self-contained module (not a reuse of `WanTransformer`)
//! The VACE checkpoint ships in **diffusers layout** (the SceneWorks worker loads it via
//! `WanVACEPipeline.from_pretrained`), so this module reads **diffusers tensor names** directly
//! (`blocks.{i}.attn1/attn2.{to_q,to_k,to_v,to_out.0}`, `scale_shift_table`, `ffn.net.0.proj`/`net.2`,
//! `norm2`, `vace_blocks.{j}.proj_in/proj_out`, `patch_embedding`/`vace_patch_embedding` Conv3d) —
//! no native conversion. It is also **dtype-generic**: the structural golden (sc-3433) is a pure-f32
//! diffusers model, so the parity path runs everything in f32; the production bf16 checkpoint runs
//! the same forward with `compute_dtype = Bfloat16` (matmuls in bf16, the residual / modulation /
//! norms / embeddings in f32 — the base Wan regime, matching diffusers' `_keep_in_fp32_modules` +
//! `_skip_layerwise_casting_patterns`). The block forward duplicates the (small) base-block sequence
//! rather than threading a dtype through the validated bf16-only [`crate::transformer`].
//!
//! **Numerics (must match diffusers):** non-affine `FP32LayerNorm` on the f32 residual (`norm1`/
//! `norm3` → [`ln`]); affine `FP32LayerNorm` for the cross-attn input (`norm2`); `qk_norm =
//! "rms_norm_across_heads"` (RMSNorm over the full `dim` before the head split); `eps` from config;
//! gelu-tanh FFN ([`crate::text_encoder::gelu_tanh`], NOT `gelu_approximate` — the mlx-rs f64-const
//! note); the sinusoidal time embedding is `flip_sin_to_cos` ([cos|sin]) with `10000^(-j/half)`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, cos, gt, multiply, sigmoid, sin, split, subtract};
use mlx_rs::{Array, Dtype};

use crate::config::WanVaceConfig;
use crate::patchify::{patchify, unpatchify};
use crate::rope::{rope_apply, RopeTable};
use crate::scheduler::{make_scheduler, SolverKind};
use crate::text_encoder::gelu_tanh;
use crate::vae::WanVae;

fn cast(x: &Array, dt: Dtype) -> Result<Array> {
    Ok(x.as_dtype(dt)?)
}

fn f32c(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// Non-affine `FP32LayerNorm` (`elementwise_affine=False`) on an f32 stream.
fn ln(x: &Array, eps: f32) -> Result<Array> {
    Ok(layer_norm(x, None, None, eps)?)
}

/// SiLU `x·σ(x)` (the `condition_embedder`'s `nn.SiLU`), dtype-preserving.
fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

/// Load a biased diffusers `nn.Linear` (`{prefix}.weight` `[out, in]` + `{prefix}.bias`). The
/// `keep_f32` flag casts the loaded weight/bias to f32 (the `_keep_in_fp32_modules` /
/// `_skip_layerwise_casting_patterns` set — patch/condition embedders, the output proj); the
/// attn/FFN Linears load as stored (bf16 in production) and matmul against a `compute_dtype`-cast
/// activation.
fn load_linear(w: &Weights, prefix: &str, keep_f32: bool) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let bias = w.require(&format!("{prefix}.bias"))?.clone();
    let (weight, bias) = if keep_f32 {
        (f32c(&weight)?, f32c(&bias)?)
    } else {
        (weight, bias)
    };
    Ok(AdaptableLinear::dense(weight, Some(bias)))
}

/// Flatten a diffusers Conv3d patch-embedding weight `[dim, in, pt, ph, pw]` → an equivalent Linear
/// `[dim, in·pt·ph·pw]` (kept f32), mirroring the base Wan converter's `patch_embedding` handling.
/// The flatten order `(in, pt, ph, pw)` matches [`patchify`]'s `(C, pt, ph, pw)` token packing.
fn load_patch_embedding(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?;
    let s = weight.shape();
    let cols: i32 = s[1..].iter().product();
    let weight = f32c(&weight.reshape(&[s[0], cols])?)?;
    let bias = f32c(w.require(&format!("{prefix}.bias"))?)?;
    Ok(AdaptableLinear::dense(weight, Some(bias)))
}

/// A Wan attention module (`attn1` self-attn or `attn2` cross-attn): `to_q/to_k/to_v/to_out.0` +
/// qk-RMSNorm `norm_q/norm_k`. Matmuls run in `compute_dtype`; the residual the caller maintains is
/// f32.
struct Attn {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
}

impl Attn {
    fn load(w: &Weights, prefix: &str, cfg: &WanVaceConfig) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            q: load_linear(w, &format!("{prefix}.to_q"), false)?,
            k: load_linear(w, &format!("{prefix}.to_k"), false)?,
            v: load_linear(w, &format!("{prefix}.to_v"), false)?,
            o: load_linear(w, &format!("{prefix}.to_out.0"), false)?,
            norm_q: w.require(&format!("{prefix}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.norm_k.weight"))?.clone(),
            num_heads: cfg.base.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.base.eps as f32,
        })
    }

    /// Self-attention with 3-axis RoPE. `x_mod`: `[1, L, dim]` (f32, already modulated). `cos`/`sin`:
    /// `[L, 1, half_d]` in `dt`. Returns `[1, L, dim]` in `dt`.
    fn self_attn(&self, x_mod: &Array, cos: &Array, sin: &Array, dt: Dtype) -> Result<Array> {
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let b = x_mod.shape()[0];
        let s = x_mod.shape()[1];
        let xw = cast(x_mod, dt)?;
        // qk-RMSNorm over the full dim (before the head split), then RoPE in f32, cast back to dt.
        let q = rms_norm(&self.q.forward(&xw)?, &self.norm_q, self.eps)?;
        let k = rms_norm(&self.k.forward(&xw)?, &self.norm_k, self.eps)?;
        let q = cast(
            &rope_apply(&f32c(&q.reshape(&[b, s, n, d])?)?, cos, sin)?,
            dt,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = cast(
            &rope_apply(&f32c(&k.reshape(&[b, s, n, d])?)?, cos, sin)?,
            dt,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&xw)?
            .reshape(&[b, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?;
        self.o.forward(&out)
    }

    /// Cross-attention over the text context (no RoPE). `x`: `[1, L, dim]` (f32). `context`:
    /// `[1, L_ctx, dim]` (f32, projected). Returns `[1, L, dim]` in `dt`.
    fn cross_attn(&self, x: &Array, context: &Array, dt: Dtype) -> Result<Array> {
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let b = x.shape()[0];
        let s = x.shape()[1];
        let q = rms_norm(&self.q.forward(&cast(x, dt)?)?, &self.norm_q, self.eps)?
            .reshape(&[b, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let ctx = cast(context, dt)?;
        let k = rms_norm(&self.k.forward(&ctx)?, &self.norm_k, self.eps)?
            .reshape(&[b, -1, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&ctx)?
            .reshape(&[b, -1, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?;
        self.o.forward(&out)
    }
}

/// The shared Wan block body (self-attn + cross-attn + gated-GELU FFN) used by **both** the main
/// blocks and the VACE blocks — diffusers `WanTransformerBlock.forward` / the inner half of
/// `WanVACETransformerBlock.forward`. Carries its own `scale_shift_table` (the adaLN-6vec modulation
/// table). The residual stream stays f32 throughout (matmul activations are cast to `dt` inside
/// [`Attn`]).
struct CoreBlock {
    /// `scale_shift_table` reshaped to `[1, 1, 6, dim]` (f32) — adds the time modulation `e0`.
    mod_table: Array,
    self_attn: Attn,
    cross_attn: Attn,
    norm2_w: Array, // affine cross-attn norm (FP32LayerNorm)
    norm2_b: Array,
    ffn_fc1: AdaptableLinear, // ffn.net.0.proj
    ffn_fc2: AdaptableLinear, // ffn.net.2
    eps: f32,
}

impl CoreBlock {
    fn load(w: &Weights, prefix: &str, cfg: &WanVaceConfig) -> Result<Self> {
        let dim = cfg.base.dim as i32;
        Ok(Self {
            mod_table: f32c(
                &w.require(&format!("{prefix}.scale_shift_table"))?
                    .reshape(&[1, 1, 6, dim])?,
            )?,
            self_attn: Attn::load(w, &format!("{prefix}.attn1"), cfg)?,
            cross_attn: Attn::load(w, &format!("{prefix}.attn2"), cfg)?,
            norm2_w: f32c(w.require(&format!("{prefix}.norm2.weight"))?)?,
            norm2_b: f32c(w.require(&format!("{prefix}.norm2.bias"))?)?,
            ffn_fc1: load_linear(w, &format!("{prefix}.ffn.net.0.proj"), false)?,
            ffn_fc2: load_linear(w, &format!("{prefix}.ffn.net.2"), false)?,
            eps: cfg.base.eps as f32,
        })
    }

    /// `x`: `[1, L, dim]` (f32). `e0`: `[1, 1, 6, dim]` (f32, the time modulation). `context`:
    /// `[1, L_ctx, dim]` (f32). Returns the updated `[1, L, dim]` (f32).
    fn forward(
        &self,
        x: &Array,
        e0: &Array,
        context: &Array,
        rope: (&Array, &Array),
        dt: Dtype,
    ) -> Result<Array> {
        let dim = self.self_attn.num_heads as i32 * self.self_attn.head_dim as i32;
        // adaLN-6vec: (scale_shift_table + e0) → split into shift/scale/gate (self) + c_* (FFN).
        let m = add(&self.mod_table, e0)?; // [1,1,6,dim]
        let p = split(&m, 6, 2)?;
        let g = |i: usize| -> Result<Array> { Ok(p[i].reshape(&[1, 1, dim])?) };
        let (shift, scale_, gate) = (g(0)?, g(1)?, g(2)?);
        let (c_shift, c_scale, c_gate) = (g(3)?, g(4)?, g(5)?);

        // Self-attention.
        let x_mod = add(
            &multiply(&ln(x, self.eps)?, &add(&scale_, scalar(1.0))?)?,
            &shift,
        )?;
        let y = self.self_attn.self_attn(&x_mod, rope.0, rope.1, dt)?;
        let x = add(x, &multiply(&y, &gate)?)?;

        // Cross-attention (affine FP32LayerNorm, no modulation).
        let x_cross = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), self.eps)?;
        let x = add(&x, &self.cross_attn.cross_attn(&x_cross, context, dt)?)?;

        // Gated-GELU FFN.
        let x_mod = add(
            &multiply(&ln(&x, self.eps)?, &add(&c_scale, scalar(1.0))?)?,
            &c_shift,
        )?;
        let y = self
            .ffn_fc2
            .forward(&gelu_tanh(&self.ffn_fc1.forward(&cast(&x_mod, dt)?)?)?)?;
        Ok(add(&x, &multiply(&y, &c_gate)?)?)
    }
}

/// A VACE block: a [`CoreBlock`] over the **control** stream, plus `proj_in` (block 0 only — injects
/// the main noisy-latent tokens into the control stream once) and `proj_out` (every block — emits the
/// per-layer hint). Diffusers `WanVACETransformerBlock`.
struct VaceBlock {
    proj_in: Option<AdaptableLinear>,
    core: CoreBlock,
    proj_out: AdaptableLinear,
}

impl VaceBlock {
    fn load(w: &Weights, prefix: &str, has_proj_in: bool, cfg: &WanVaceConfig) -> Result<Self> {
        let proj_in = if has_proj_in {
            Some(load_linear(w, &format!("{prefix}.proj_in"), false)?)
        } else {
            None
        };
        Ok(Self {
            proj_in,
            core: CoreBlock::load(w, prefix, cfg)?,
            proj_out: load_linear(w, &format!("{prefix}.proj_out"), false)?,
        })
    }

    /// `control`: `[1, L, dim]` (f32, the running control stream). `hidden`: `[1, L, dim]` (f32, the
    /// patch-embedded main latent — used only by block 0's `proj_in`). Returns `(hint, control)` both
    /// f32: `hint = proj_out(control)` (added to the main stream at the matching vace layer), and the
    /// updated control stream threaded to the next vace block.
    fn forward(
        &self,
        control: &Array,
        hidden: &Array,
        e0: &Array,
        context: &Array,
        rope: (&Array, &Array),
        dt: Dtype,
    ) -> Result<(Array, Array)> {
        let control = match &self.proj_in {
            Some(proj_in) => add(&proj_in.forward(&cast(control, dt)?)?, hidden)?,
            None => control.clone(),
        };
        let control = self.core.forward(&control, e0, context, rope, dt)?;
        let hint = self.proj_out.forward(&cast(&control, dt)?)?;
        Ok((f32c(&hint)?, control))
    }
}

/// The Wan-VACE DiT — the base Wan transformer plus the VACE control path. Loaded from a
/// diffusers-layout VACE checkpoint; runs in `compute_dtype` (f32 for the structural golden, bf16 for
/// production).
pub struct WanVaceTransformer {
    patch_embedding: AdaptableLinear, // Conv3d(in,dim) flattened → [dim, in·∏patch]
    vace_patch_embedding: AdaptableLinear, // Conv3d(96,dim) flattened → [dim, 96·∏patch]
    time_embedding_0: AdaptableLinear, // condition_embedder.time_embedder.linear_1
    time_embedding_1: AdaptableLinear, // condition_embedder.time_embedder.linear_2
    time_projection: AdaptableLinear, // condition_embedder.time_proj
    text_embedding_0: AdaptableLinear, // condition_embedder.text_embedder.linear_1
    text_embedding_1: AdaptableLinear, // condition_embedder.text_embedder.linear_2
    blocks: Vec<CoreBlock>,
    vace_blocks: Vec<VaceBlock>,
    head_modulation: Array, // scale_shift_table [1, 2, dim] (f32)
    head: AdaptableLinear,  // proj_out [out·∏patch, dim]
    rope: RopeTable,
    inv_freq: Array, // [freq_dim/2] f32
    cfg: WanVaceConfig,
    compute_dtype: Dtype,
}

impl WanVaceTransformer {
    /// Load from a diffusers-layout VACE transformer weight map. `compute_dtype` selects the matmul
    /// precision (f32 for the structural golden; bf16 for the production checkpoint).
    pub fn from_weights(w: &Weights, cfg: &WanVaceConfig, compute_dtype: Dtype) -> Result<Self> {
        let dim = cfg.base.dim as i32;
        let mut blocks = Vec::with_capacity(cfg.base.num_layers);
        for i in 0..cfg.base.num_layers {
            blocks.push(CoreBlock::load(w, &format!("blocks.{i}"), cfg)?);
        }
        let mut vace_blocks = Vec::with_capacity(cfg.vace_layers.len());
        for j in 0..cfg.vace_layers.len() {
            vace_blocks.push(VaceBlock::load(
                w,
                &format!("vace_blocks.{j}"),
                j == 0,
                cfg,
            )?);
        }
        let half = cfg.base.freq_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| (10000.0_f64.powf(-(j as f64) / half as f64)) as f32)
            .collect();
        Ok(Self {
            patch_embedding: load_patch_embedding(w, "patch_embedding")?,
            vace_patch_embedding: load_patch_embedding(w, "vace_patch_embedding")?,
            time_embedding_0: load_linear(w, "condition_embedder.time_embedder.linear_1", true)?,
            time_embedding_1: load_linear(w, "condition_embedder.time_embedder.linear_2", true)?,
            time_projection: load_linear(w, "condition_embedder.time_proj", true)?,
            text_embedding_0: load_linear(w, "condition_embedder.text_embedder.linear_1", true)?,
            text_embedding_1: load_linear(w, "condition_embedder.text_embedder.linear_2", true)?,
            blocks,
            vace_blocks,
            head_modulation: f32c(&w.require("scale_shift_table")?.reshape(&[1, 2, dim])?)?,
            head: load_linear(w, "proj_out", true)?,
            rope: RopeTable::new(cfg.head_dim()),
            inv_freq: Array::from_slice(&inv, &[half as i32]),
            cfg: cfg.clone(),
            compute_dtype,
        })
    }

    /// Patchify grid `(f, h, w)` for a latent `[C, F, H, W]`.
    fn patch_grid(&self, latent: &Array) -> (usize, usize, usize) {
        let sh = latent.shape();
        let (pt, ph, pw) = self.cfg.base.patch_size;
        (
            sh[1] as usize / pt,
            sh[2] as usize / ph,
            sh[3] as usize / pw,
        )
    }

    /// Sinusoidal time embedding: `e` `[1, dim]` (the head modulation) and `e0` `[1, 1, 6, dim]` (the
    /// block modulation). `flip_sin_to_cos` ([cos|sin]); the `time_embedder` + `time_proj` are
    /// `_keep_in_fp32` so this is always f32. Mirrors diffusers `condition_embedder`'s time path.
    fn time_embed(&self, t: f32) -> Result<(Array, Array)> {
        let dim = self.cfg.base.dim as i32;
        let pos = Array::from_slice(&[t], &[1, 1]);
        let sinusoid = multiply(&pos, &self.inv_freq)?; // [1, half]
        let sin_emb = concatenate_axis(&[&cos(&sinusoid)?, &sin(&sinusoid)?], 1)?; // [1, freq_dim]
        let e = self
            .time_embedding_1
            .forward(&silu(&self.time_embedding_0.forward(&sin_emb)?)?)?; // [1, dim]
        let e0 = self
            .time_projection
            .forward(&silu(&e)?)?
            .reshape(&[1, 1, 6, dim])?;
        Ok((e, e0))
    }

    /// Project the raw text context `[1, L_ctx, text_dim]` → `[1, L_ctx, dim]` (f32) via the
    /// `text_embedder` (`PixArtAlphaTextProjection`, `gelu_tanh`). `_keep_in_fp32` → f32.
    fn text_embed(&self, context: &Array) -> Result<Array> {
        let h = gelu_tanh(&self.text_embedding_0.forward(context)?)?;
        self.text_embedding_1.forward(&h)
    }

    /// Output head: modulated non-affine norm + the `proj_out` projection. `e`: `[1, dim]`.
    fn apply_head(&self, x: &Array, e: &Array) -> Result<Array> {
        let dim = self.cfg.base.dim as i32;
        let m = add(&self.head_modulation, &e.reshape(&[1, 1, dim])?)?; // [1,2,dim]
        let p = split(&m, 2, 1)?;
        let shift = p[0].reshape(&[1, 1, dim])?;
        let scale_ = p[1].reshape(&[1, 1, dim])?;
        let x_mod = add(
            &multiply(
                &ln(x, self.cfg.base.eps as f32)?,
                &add(&scale_, scalar(1.0))?,
            )?,
            &shift,
        )?;
        self.head.forward(&x_mod)
    }

    /// Full Wan-VACE forward. `latent`: `[in, F, H, W]` (f32, the noisy latent). `control`:
    /// `[96, F_c, H, W]` (f32, the VACE control latent = video(32) + mask(64)). `t`: integer-valued
    /// timestep. `context`: `[1, L_ctx, text_dim]` (f32, the raw text embedding). `scales`: the
    /// per-vace-layer hint scales (`control_hidden_states_scale`, one per `vace_layers` entry).
    /// Returns the denoised `[out, F, H, W]` (f32).
    pub fn forward_vace(
        &self,
        latent: &Array,
        control: &Array,
        t: f32,
        context: &Array,
        scales: &[f32],
    ) -> Result<Array> {
        if scales.len() != self.cfg.vace_layers.len() {
            return Err(mlx_gen::Error::Msg(format!(
                "wan-vace: control_hidden_states_scale len {} != vace_layers len {}",
                scales.len(),
                self.cfg.vace_layers.len()
            )));
        }
        let dt = self.compute_dtype;
        let dim = self.cfg.base.dim as i32;
        let patch = self.cfg.base.patch_size;

        // RoPE over the main latent grid (cast to compute_dtype, mirroring the base Wan regime).
        let grid = self.patch_grid(latent);
        let (cos_t, sin_t) = self.rope.precompute_cos_sin(grid)?;
        let (cos_t, sin_t) = (cast(&cos_t, dt)?, cast(&sin_t, dt)?);

        // Patch-embed the noisy latent → [1, L, dim] (f32 residual).
        let (tokens, _) = patchify(latent, patch)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let x_tokens = self
            .patch_embedding
            .forward(&tokens)?
            .reshape(&[1, l, dim])?;

        // Patch-embed the control latent via vace_patch_embedding, then zero-pad tokens to L.
        let (ctokens, cgrid) = patchify(control, patch)?;
        let l_c = (cgrid.0 * cgrid.1 * cgrid.2) as i32;
        let control_emb = self
            .vace_patch_embedding
            .forward(&ctokens)?
            .reshape(&[1, l_c, dim])?;
        let control_emb = if l_c < l {
            let pad = Array::zeros::<f32>(&[1, l - l_c, dim])?;
            concatenate_axis(&[&control_emb, &pad], 1)?
        } else {
            control_emb
        };

        // Condition embeddings.
        let (e, e0) = self.time_embed(t)?;
        let context_emb = self.text_embed(context)?;

        // VACE hint prep: thread the control stream through every vace block; collect hints.
        let rope = (&cos_t, &sin_t);
        let mut control_hs = control_emb;
        let mut hints: Vec<(Array, f32)> = Vec::with_capacity(self.vace_blocks.len());
        for (j, vb) in self.vace_blocks.iter().enumerate() {
            let (hint, new_control) =
                vb.forward(&control_hs, &x_tokens, &e0, &context_emb, rope, dt)?;
            hints.push((hint, scales[j]));
            control_hs = new_control;
        }
        hints.reverse();

        // Main blocks with hint injection at each layer in vace_layers.
        let mut x = x_tokens;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &e0, &context_emb, rope, dt)?;
            if self.cfg.vace_layers.contains(&i) {
                let (hint, scale) = hints
                    .pop()
                    .expect("one hint per vace layer (vace_layers.len() == vace_blocks.len())");
                x = add(&x, &multiply(&hint, scalar(scale))?)?;
            }
        }

        // Output norm, projection & unpatchify.
        let x = self.apply_head(&x, &e)?; // [1, L, out·∏patch]
        let op = x.shape()[2];
        let xb = x.reshape(&[l, op])?;
        unpatchify(&xb, grid, self.cfg.base.out_dim, patch)
    }

    /// Per-stage capture for parity bisection (sc-3388 S1) — mirrors [`forward_vace`](
    /// Self::forward_vace) but returns the named intermediates in diffusers-comparable layout:
    /// `(x_tokens, control_emb, temb, timestep_proj, text_emb, vace0_hint, vace0_control, block0_out,
    /// output)`. `timestep_proj` is flattened to `[1, 6·dim]` and `output` is `[1, out, F, H, W]` to
    /// match the `tools/dump_wanvace_bisect.py` hook captures.
    #[doc(hidden)]
    pub fn forward_vace_capture(
        &self,
        latent: &Array,
        control: &Array,
        t: f32,
        context: &Array,
        scales: &[f32],
    ) -> Result<Vec<(&'static str, Array)>> {
        let dt = self.compute_dtype;
        let dim = self.cfg.base.dim as i32;
        let patch = self.cfg.base.patch_size;
        let grid = self.patch_grid(latent);
        let (cos_t, sin_t) = self.rope.precompute_cos_sin(grid)?;
        let (cos_t, sin_t) = (cast(&cos_t, dt)?, cast(&sin_t, dt)?);

        let (tokens, _) = patchify(latent, patch)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let x_tokens = self
            .patch_embedding
            .forward(&tokens)?
            .reshape(&[1, l, dim])?;

        let (ctokens, cgrid) = patchify(control, patch)?;
        let l_c = (cgrid.0 * cgrid.1 * cgrid.2) as i32;
        let control_emb = self
            .vace_patch_embedding
            .forward(&ctokens)?
            .reshape(&[1, l_c, dim])?;

        let (e, e0) = self.time_embed(t)?;
        let context_emb = self.text_embed(context)?;

        let control_padded = if l_c < l {
            let pad = Array::zeros::<f32>(&[1, l - l_c, dim])?;
            concatenate_axis(&[&control_emb, &pad], 1)?
        } else {
            control_emb.clone()
        };

        let rope = (&cos_t, &sin_t);
        let mut control_hs = control_padded;
        let mut hints: Vec<(Array, f32)> = Vec::with_capacity(self.vace_blocks.len());
        let mut vace0_hint = control_emb.clone();
        let mut vace0_control = control_emb.clone();
        for (j, vb) in self.vace_blocks.iter().enumerate() {
            let (hint, new_control) =
                vb.forward(&control_hs, &x_tokens, &e0, &context_emb, rope, dt)?;
            if j == 0 {
                vace0_hint = hint.clone();
                vace0_control = new_control.clone();
            }
            hints.push((hint, scales[j]));
            control_hs = new_control;
        }
        hints.reverse();

        let mut x = x_tokens.clone();
        let mut block0_out = x_tokens.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &e0, &context_emb, rope, dt)?;
            if i == 0 {
                block0_out = x.clone();
            }
            if self.cfg.vace_layers.contains(&i) {
                let (hint, scale) = hints.pop().expect("one hint per vace layer");
                x = add(&x, &multiply(&hint, scalar(scale))?)?;
            }
        }

        let xh = self.apply_head(&x, &e)?;
        let op = xh.shape()[2];
        let unp = unpatchify(&xh.reshape(&[l, op])?, grid, self.cfg.base.out_dim, patch)?;
        let us = unp.shape();
        let output = unp.reshape(&[1, us[0], us[1], us[2], us[3]])?;

        Ok(vec![
            ("x_tokens", x_tokens),
            ("control_emb", control_emb),
            ("temb", e),
            ("timestep_proj", e0.reshape(&[1, 6 * dim])?),
            ("text_emb", context_emb),
            ("vace0_hint", vace0_hint),
            ("vace0_control", vace0_control),
            ("block0_out", block0_out),
            ("output", output),
        ])
    }
}

// ============================================================================================
// S2 (sc-3435) — VACE conditioning construction (the host / VAE side).
//
// Builds the 96-ch `control_hidden_states = cat([video_latents(32), mask_latents(64)], channels)`
// the [`WanVaceTransformer::forward_vace`] consumes. Mirrors diffusers `WanVACEPipeline`'s
// `prepare_video_latents` + `prepare_masks` + the `__call__` concat, in the crate's no-batch
// `[C, F, H, W]` convention. The VAE-encode + `(x−mean)·std` normalization is the already-validated
// [`WanVae::encode`] (it returns the mode/argmax latent normalized — exactly diffusers'
// `retrieve_latents(sample_mode="argmax")` + normalize); the genuinely new pieces here are the
// mask 8×8-unfold + nearest-exact temporal resample, the inactive/reactive masking, and the
// reference-frame prepend — all byte-validated as pure host ops in `tests/wanvace_cond_parity.rs`.
// ============================================================================================

/// Binarize a soft control mask: `where(mask > 0.5, 1.0, 0.0)` (diffusers `prepare_video_latents`).
pub fn binarize_mask(mask: &Array) -> Result<Array> {
    Ok(gt(mask, scalar(0.5))?.as_dtype(Dtype::Float32)?)
}

/// Nearest-exact temporal resample along the frame axis (axis 1) of a `[C, F, H, W]` tensor →
/// `[C, out_t, H, W]`. torch `mode="nearest-exact"`: `src = floor((i + 0.5)·F / out_t)`, clamped to
/// `[0, F−1]`. Spatial dims are unchanged (the VACE mask interp resamples time only).
fn nearest_exact_temporal(x: &Array, out_t: usize) -> Result<Array> {
    let f = x.shape()[1] as usize;
    let idx: Vec<i32> = (0..out_t)
        .map(|i| {
            let s = (((i as f64) + 0.5) * (f as f64) / (out_t as f64)).floor() as i64;
            s.clamp(0, f as i64 - 1) as i32
        })
        .collect();
    let idx = Array::from_slice(&idx, &[out_t as i32]);
    Ok(x.take_axis(&idx, 1)?)
}

/// `prepare_masks`: a soft control mask `[C, F, H, W]` (`C≥1`; channel 0 is used) → the 64-ch mask
/// latent `[64, new_t (+ num_ref), new_h, new_w]`, where `64 = vae_s²`, `new_t = ⌈F / vae_t⌉`,
/// `new_h = H/(vae_s·patch)·patch`. The mask is unfolded `view(F, new_h, vae_s, new_w, vae_s)
/// .permute(2,4,0,1,3).flatten(0,1)` → `[vae_s², F, new_h, new_w]`, nearest-exact resampled in time
/// to `new_t`, then `num_ref` zero frames are prepended along the frame axis. Pure host op (diffusers
/// `WanVACEPipeline.prepare_masks`, single batch).
pub fn prepare_masks(
    mask: &Array,
    vae_t: usize,
    vae_s: usize,
    patch: usize,
    num_ref: usize,
) -> Result<Array> {
    let sh = mask.shape();
    let (f, h, w) = (sh[1], sh[2], sh[3]);
    let new_t = (f as usize).div_ceil(vae_t);
    let new_h = (h as usize / (vae_s * patch) * patch) as i32;
    let new_w = (w as usize / (vae_s * patch) * patch) as i32;
    let vs = vae_s as i32;

    // Channel 0 → [F, H, W].
    let ch0 = mask
        .take_axis(Array::from_slice(&[0i32], &[1]), 0)?
        .reshape(&[f, h, w])?;
    // [F, new_h, vae_s, new_w, vae_s] → permute(2,4,0,1,3) → [vae_s, vae_s, F, new_h, new_w]
    // → flatten(0,1) → [vae_s², F, new_h, new_w].
    let m = ch0
        .reshape(&[f, new_h, vs, new_w, vs])?
        .transpose_axes(&[2, 4, 0, 1, 3])?
        .reshape(&[vs * vs, f, new_h, new_w])?;
    let m = nearest_exact_temporal(&m, new_t)?;
    if num_ref > 0 {
        let pad = Array::zeros::<f32>(&[vs * vs, num_ref as i32, new_h, new_w])?;
        Ok(concatenate_axis(&[&pad, &m], 1)?)
    } else {
        Ok(m)
    }
}

/// Encode one frame-batch `[C, F, H, W]` through the Wan z16 VAE → the normalized latent
/// `[z, F_lat, h, w]` (drops the batch the VAE adds). [`WanVae::encode`] already applies the
/// mode/argmax + `(x−mean)·std` normalization.
fn encode_clip(vae: &WanVae, clip: &Array) -> Result<Array> {
    let sh = clip.shape(); // [C, F, H, W]
    let z = vae.encode(&clip.reshape(&[1, sh[0], sh[1], sh[2], sh[3]])?)?; // [1, z, F_lat, h, w]
    let zs = z.shape();
    Ok(z.reshape(&[zs[1], zs[2], zs[3], zs[4]])?)
}

/// `prepare_video_latents`: the control video `[3, F, H, W]` (+ optional binarized mask + reference
/// images) → the 32-ch (or 16-ch, no mask) video-latent `[32, F_lat (+ num_ref), h, w]`. With a mask:
/// `inactive = video·(1−mask)`, `reactive = video·mask`, each VAE-encoded + normalized, concatenated
/// along channels. Each reference image `[3, H, W]` is encoded to one latent frame, `cat([ref,
/// zeros])` to 32 ch, and prepended along the frame axis. Mirrors diffusers
/// `WanVACEPipeline.prepare_video_latents` (single batch). Checkpoint-gated (needs the z16 VAE).
pub fn prepare_video_latents(
    vae: &WanVae,
    video: &Array,
    mask: Option<&Array>,
    references: &[Array],
) -> Result<Array> {
    let mut latents = match mask {
        None => encode_clip(vae, video)?, // 16 ch
        Some(m) => {
            let m = binarize_mask(m)?;
            let inactive = encode_clip(vae, &multiply(video, &subtract(scalar(1.0), &m)?)?)?;
            let reactive = encode_clip(vae, &multiply(video, &m)?)?;
            concatenate_axis(&[&inactive, &reactive], 0)? // 32 ch
        }
    };
    // Reference images: encode → [z,1,h,w] → cat([ref, zeros]) → 2z ch → prepend along frames.
    for reference in references {
        let rsh = reference.shape(); // [3, H, W]
        let ref_clip = reference.reshape(&[rsh[0], 1, rsh[1], rsh[2]])?; // [3,1,H,W]
        let ref_lat = encode_clip(vae, &ref_clip)?; // [z,1,h,w]
        let zeros = Array::zeros::<f32>(ref_lat.shape())?;
        let ref_lat = concatenate_axis(&[&ref_lat, &zeros], 0)?; // [2z,1,h,w]
        latents = concatenate_axis(&[&ref_lat, &latents], 1)?; // prepend along frames
    }
    Ok(latents)
}

/// Assemble the 96-ch `control_hidden_states = cat([video_latents(32), mask_latents(64)], channels)`
/// (diffusers `__call__`'s `conditioning_latents = cat([conditioning_latents, mask], dim=1)`).
pub fn build_vace_control(video_latents: &Array, mask_latents: &Array) -> Result<Array> {
    Ok(concatenate_axis(&[video_latents, mask_latents], 0)?)
}

/// VACE CFG denoise loop (sc-3436) — mirrors the validated base Wan [`crate::pipeline::denoise`]
/// (same `make_scheduler` + per-step `eval`), but each step runs [`WanVaceTransformer::forward_vace`]
/// with the constant 96-ch `control` + per-vace-layer `scales`, classifier-free-guided against the
/// (optional) unconditional context. The control latent is constant across steps (built once by the
/// caller from [`prepare_video_latents`] + [`prepare_masks`] + [`build_vace_control`]).
///
/// `init_noise`: `[out_dim, T, h, w]` (f32). `ctx_cond` / `ctx_uncond`: raw text embeddings
/// `[1, L, text_dim]` (the transformer projects them via its `text_embedder`). CFG:
/// `pred = uncond + guidance·(cond − uncond)` (diffusers `WanVACEPipeline`). Returns the denoised
/// `[out_dim, T, h, w]`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_vace(
    transformer: &WanVaceTransformer,
    control: &Array,
    scales: &[f32],
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        let cond = transformer.forward_vace(&latents, control, t, ctx_cond, scales)?;
        let pred = match ctx_uncond {
            Some(uncond_ctx) => {
                let uncond = transformer.forward_vace(&latents, control, t, uncond_ctx, scales)?;
                add(
                    &uncond,
                    &multiply(&subtract(&cond, &uncond)?, scalar(guidance))?,
                )?
            }
            None => cond,
        };
        latents = sched.step(&pred, &latents)?;
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}
