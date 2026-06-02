//! Z-Image S3-DiT transformer with a VACE-style ControlNet branch (sc-2349 / sc-2257). Port of the
//! fork's `ZImageControlTransformer` (`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`, v2.1
//! config: `add_control_noise_refiner=True` / `add_control_noise_refiner_correctly=True`).
//!
//! On top of the parity-proven base [`ZImageTransformer`] (composed, not re-derived — its
//! submodules and `patchify`/`unpatchify` are reused via `pub(crate)`), this adds:
//!   - `control_all_x_embedder`: a `33·patch²·f_patch → dim` patch embedder for the VAE-encoded
//!     control context (control latent 16ch + mask 1ch + inpaint latent 16ch).
//!   - `control_noise_refiner` (2 blocks): a parallel control refiner whose hints inject into the
//!     base `noise_refiner` (image-length stage).
//!   - `control_layers` (15 blocks at base places 0,2,…,28): the main control stack whose hints
//!     inject into the matching base `layers` (unified image+caption stage).
//!
//! The control branch reuses the base image position ids / RoPE / padding (the control context
//! shares the image's spatial dims), so no separate alignment is needed. With `control_context =
//! None` the forward delegates to the base transformer verbatim; with `control_context_scale = 0`
//! the hints contribute zero — both reproduce the base output exactly (the parity self-check).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use crate::control_transformer_block::ZImageControlBlock;
use crate::transformer::{apply_pad, row_indices, ZImageTransformer};
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Channel count of the VAE-encoded control context (16 control latent + 1 mask + 16 inpaint).
pub const CONTROL_IN_DIM: i32 = 33;
/// Base `layers` indices the 15 control layers inject into (the fork's `CONTROL_LAYERS_PLACES`).
const CONTROL_LAYERS_PLACES: [usize; 15] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
/// Base `noise_refiner` indices the 2 control refiner blocks inject into.
const CONTROL_REFINER_PLACES: [usize; 2] = [0, 1];

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

pub struct ZImageControlTransformer {
    base: ZImageTransformer,
    /// `control_all_x_embedder.{patch_size}-{f_patch_size}` — `33·p²·pf → dim`. Kept dense under
    /// Q8 (its in-features, e.g. 132, is not divisible by the group size 64).
    control_x_embedder: AdaptableLinear,
    /// The 15-block main control stack (injects into `base.layers` at `CONTROL_LAYERS_PLACES`).
    control_layers: Vec<ZImageControlBlock>,
    /// The 2-block control refiner (injects into `base.noise_refiner` at `CONTROL_REFINER_PLACES`).
    control_noise_refiner: Vec<ZImageControlBlock>,
}

impl ZImageControlTransformer {
    /// Build from an already-loaded base transformer + the Fun-Controlnet-Union checkpoint
    /// (`control` Weights). The control keys (`control_all_x_embedder.*`, `control_layers.*`,
    /// `control_noise_refiner.*`) map 1:1 onto this tree. `prefix` is empty for a real checkpoint
    /// (un-prefixed keys) and e.g. `"w"` for the synthetic parity fixture.
    pub fn from_weights(base: ZImageTransformer, control: &Weights, prefix: &str) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let key = format!("{}-{}", base.cfg.patch_size, base.cfg.f_patch_size);
        let bcfg = crate::transformer_block::ZImageBlockConfig {
            dim: base.cfg.dim,
            n_heads: base.cfg.n_heads,
            norm_eps: base.cfg.norm_eps,
        };

        let control_x_embedder = AdaptableLinear::dense(
            control
                .require(&p(&format!("control_all_x_embedder.{key}.weight")))?
                .clone(),
            Some(
                control
                    .require(&p(&format!("control_all_x_embedder.{key}.bias")))?
                    .clone(),
            ),
        );

        let control_layers = (0..CONTROL_LAYERS_PLACES.len())
            .map(|i| {
                ZImageControlBlock::from_weights(
                    control,
                    &p(&format!("control_layers.{i}")),
                    bcfg,
                    i == 0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let control_noise_refiner = (0..CONTROL_REFINER_PLACES.len())
            .map(|i| {
                ZImageControlBlock::from_weights(
                    control,
                    &p(&format!("control_noise_refiner.{i}")),
                    bcfg,
                    i == 0,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            base,
            control_x_embedder,
            control_layers,
            control_noise_refiner,
        })
    }

    /// Quantize to Q4/Q8 (group_size 64) — the base transformer plus every control block, but
    /// **not** the control patch embedder: its in-features (`33·p²·pf`, e.g. 132) is not divisible
    /// by 64, so `nn.quantize` leaves it dense (the fork's `d32454c` predicate). The fork applies
    /// base + control weights at full precision first, *then* quantizes the whole transformer;
    /// here the base load + this overlay are both dense, and this quantizes both together.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.base.quantize(bits)?;
        for block in &mut self.control_layers {
            block.quantize(bits)?;
        }
        for block in &mut self.control_noise_refiner {
            block.quantize(bits)?;
        }
        // control_x_embedder intentionally left dense (in-features not divisible by 64).
        Ok(())
    }

    /// Dual-injection forward (port of `ZImageControlTransformer.__call__`).
    ///
    /// `x`: latent `(C, F, H, W)`; `cap_feats`: `(cap_len, cap_feat_dim)`; `timestep` in `[0,1]`;
    /// `control_context`: the `(33, F, H/8?, W/8?)` VAE-encoded control context (same spatial dims
    /// as the latent), or `None` to run the base transformer verbatim; `control_context_scale`
    /// weights every control hint. Returns the latent-shaped velocity `(C, F, H, W)`.
    pub fn forward(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        control_context: Option<&Array>,
        control_context_scale: f32,
    ) -> Result<Array> {
        // No control context → identical to the base transformer (the fork's `control_context is
        // None` short-circuit). Delegating keeps the base path byte-for-byte the parity-proven one.
        match control_context {
            None => self.base.forward(x, timestep, cap_feats),
            Some(cc) => {
                self.forward_control(x, timestep, cap_feats, cc, control_context_scale, None)
            }
        }
    }

    /// Like [`forward`](Self::forward) (control path) but also returns the named per-stage
    /// intermediates, for stage-by-stage parity bisection vs the fork (`control_q8_bisect`). The
    /// returned output is identical to `forward`'s — both call the same `forward_control`.
    pub fn forward_capture(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        control_context: &Array,
        control_context_scale: f32,
    ) -> Result<(Array, Vec<(&'static str, Array)>)> {
        let mut stages = Vec::new();
        let v = self.forward_control(
            x,
            timestep,
            cap_feats,
            control_context,
            control_context_scale,
            Some(&mut stages),
        )?;
        Ok((v, stages))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_control(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        cc: &Array,
        control_context_scale: f32,
        mut cap: Option<&mut Vec<(&'static str, Array)>>,
    ) -> Result<Array> {
        macro_rules! record {
            ($name:expr, $a:expr) => {
                if let Some(c) = cap.as_deref_mut() {
                    c.push(($name, $a.clone()));
                }
            };
        }

        let cfg = &self.base.cfg;
        let t = Array::from_slice(&[timestep * cfg.t_scale], &[1]);
        let t_emb = self.base.t_embedder.forward(&t)?;
        record!("t_emb", t_emb);

        let patched = self.base.patchify(x, cap_feats)?;
        record!("x_tokens", patched.x_tokens); // pre-embed patchify output (the x-embedder input)

        // Image stream: embed → set padded positions to x_pad_token → (control refiner) → noise refiner.
        let mut x_emb = self.base.x_embedder.forward(&patched.x_tokens)?;
        x_emb = apply_pad(&x_emb, &patched.x_keep, &self.base.x_pad_token)?;
        let x_freqs = self.base.rope.forward(&patched.x_pos_ids)?;
        let mut x_emb = x_emb.expand_dims(0)?;
        record!("x_emb", x_emb);

        // Control refiner pass: build the control context embedding (reusing the image's pad mask /
        // RoPE), run the parallel control refiner, collect the per-block hints + threaded state.
        let c_tokens = patchify_control(cc, cfg.patch_size, cfg.f_patch_size)?;
        // The control patch embedder is `K = 33·p²·pf` (e.g. 132): K ≤ 512 with large M, which is
        // the garbage zone of the pmetal NAX 16-bit dense GEMM (see `bf16_matmul_sweep` — a bf16
        // control forward returns ~0.49 peak-rel garbage there, vs 1.5e-3 in f32). Run *only* this
        // GEMM in f32 (`matmul(f32, bf16-weight)` promotes to the correct f32 kernel), then return
        // to the image stream's dtype, so the rest of the control branch stays bf16 (its block
        // GEMMs are K=3840, safe; adaLN is M=1 gemv, safe). The base `x_embedder` (K=64) is
        // empirically fine in bf16 (the base txt2img path is parity-proven), so it is left alone.
        let stream_dtype = x_emb.dtype();
        let c_tokens = c_tokens.as_dtype(Dtype::Float32)?;
        let mut c_emb = self
            .control_x_embedder
            .forward(&c_tokens)?
            .as_dtype(stream_dtype)?;
        c_emb = apply_pad(&c_emb, &patched.x_keep, &self.base.x_pad_token)?;
        let c_emb = c_emb.expand_dims(0)?;
        record!("c_emb", c_emb);
        let (refiner_hints, threaded_control) =
            self.run_control_blocks(&self.control_noise_refiner, c_emb, &x_emb, &x_freqs, &t_emb)?;
        record!("refiner_hint0", refiner_hints[0]);
        record!("refiner_hint1", refiner_hints[1]);
        record!("threaded", threaded_control);

        // Noise refiner (with control hints).
        for (i, layer) in self.base.noise_refiner.iter().enumerate() {
            x_emb = layer.forward(&x_emb, &x_freqs, &t_emb)?;
            if let Some(n) = hint_index(&CONTROL_REFINER_PLACES, i) {
                x_emb = add_hint(&x_emb, &refiner_hints[n], control_context_scale)?;
            }
        }
        record!("x_refined", x_emb);

        // Caption stream: RMSNorm → linear → set padded to cap_pad_token → context refiner.
        let cap_normed = rms_norm(&patched.cap_tokens, &self.base.cap_norm_w, cfg.norm_eps)?;
        let mut cap_emb = self.base.cap_linear.forward(&cap_normed)?;
        cap_emb = apply_pad(&cap_emb, &patched.cap_keep, &self.base.cap_pad_token)?;
        let cap_freqs = self.base.rope.forward(&patched.cap_pos_ids)?;
        let mut cap_emb = cap_emb.expand_dims(0)?;
        for layer in &self.base.context_refiner {
            cap_emb = layer.forward(&cap_emb, &cap_freqs)?;
        }
        record!("cap_refined", cap_emb);

        // Unify image + caption.
        let x_len = x_emb.shape()[1];
        let mut unified = concatenate_axis(&[&x_emb, &cap_emb], 1)?;
        let unified_freqs = concatenate_axis(&[&x_freqs, &cap_freqs], 0)?;

        // Main control pass: thread the (refined) control state + caption through the 15 control
        // layers to produce the hints for the unified main loop.
        let control_unified = concatenate_axis(&[&threaded_control, &cap_emb], 1)?;
        let (main_hints, _) = self.run_control_blocks(
            &self.control_layers,
            control_unified,
            &unified,
            &unified_freqs,
            &t_emb,
        )?;
        record!("main_hint0", main_hints[0]);
        record!("main_hint_last", main_hints[main_hints.len() - 1]);

        // Main layers (with control hints).
        for (i, layer) in self.base.layers.iter().enumerate() {
            unified = layer.forward(&unified, &unified_freqs, &t_emb)?;
            if let Some(n) = hint_index(&CONTROL_LAYERS_PLACES, i) {
                unified = add_hint(&unified, &main_hints[n], control_context_scale)?;
            }
        }
        record!("unified_main", unified);

        // Final layer + unpatchify (only the real image tokens survive), negate the velocity.
        let unified = self.base.final_layer.forward(&unified, &t_emb)?;
        let embed_dim = unified.shape()[2];
        let head = unified
            .reshape(&[unified.shape()[1], embed_dim])?
            .take_axis(row_indices(x_len), 0)?;
        let out = self.base.unpatchify(&head, patched.x_size)?;
        Ok(out.multiply(scalar(-1.0))?)
    }

    /// Run a parallel control stack, returning `(per-block hints, threaded control state)`.
    /// Mirrors the VACE threading: block 0 seeds the branch via `before_proj(c) + x_base`; each
    /// block runs the base-block forward and emits `after_proj(c)` as its hint, passing the running
    /// control state `c` to the next block.
    fn run_control_blocks(
        &self,
        blocks: &[ZImageControlBlock],
        c: Array,
        x_base: &Array,
        freqs_cis: &Array,
        t_emb: &Array,
    ) -> Result<(Vec<Array>, Array)> {
        let mut c = c;
        let mut hints = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            if i == 0 {
                let bp = block
                    .before_proj()
                    .expect("control block 0 carries before_proj");
                c = add(&bp.forward(&c)?, x_base)?;
            }
            c = block.base.forward(&c, freqs_cis, t_emb)?;
            hints.push(block.after_proj().forward(&c)?);
        }
        Ok((hints, c))
    }
}

impl AdaptableHost for ZImageControlTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Adapters target the base DiT — the fork trains LoRA/LoKr on the base transformer, not the
        // control branch (`control_x_embedder` / `control_layers` / `control_noise_refiner` are
        // ControlNet weights, never adapter targets). Delegate the whole path to the composed base.
        self.base.adaptable_mut(path)
    }
}

/// `x + hint · scale`, broadcasting the scalar scale over `[1, seq, dim]`.
fn add_hint(x: &Array, hint: &Array, scale: f32) -> Result<Array> {
    Ok(add(x, &multiply(hint, scalar(scale))?)?)
}

/// The control-stack hint index for base-layer `i` (the fork's `*_mapping[i]`), or `None` when no
/// control block injects there.
fn hint_index(places: &[usize], i: usize) -> Option<usize> {
    places.iter().position(|&p| p == i)
}

/// Patchify the `(C=33, F, H, W)` control context into `(seq, p²·pf·33)` tokens, padded to a
/// multiple of 32 exactly like the base image patchify so the control sequence aligns 1:1 with the
/// image tokens (shared RoPE / pad mask). Padded rows are zeroed (the embedder output for them is
/// overwritten by `x_pad_token` in the forward, matching the base patchify's zero-pad convention).
fn patchify_control(cc: &Array, patch_size: i32, f_patch_size: i32) -> Result<Array> {
    let (pf, ph, pw) = (f_patch_size, patch_size, patch_size);
    let sh = cc.shape();
    let (c, f, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let (ft, ht, wt) = (f / pf, h / ph, w / pw);
    let tokens = cc
        .reshape(&[c, ft, pf, ht, ph, wt, pw])?
        .transpose_axes(&[1, 3, 5, 2, 4, 6, 0])?
        .reshape(&[ft * ht * wt, pf * ph * pw * c])?;
    let ori = ft * ht * wt;
    let pad = (-(ori as i64)).rem_euclid(32) as i32;
    crate::transformer::pad_rows(&tokens, pad)
}
