//! The top-level Lens DiT (`LensTransformer2DModel`): multi-layer text front-end → `img_in` +
//! timestep embedding → 48 dual-stream blocks → `AdaLayerNormContinuous` + `proj_out` back to patch
//! space. Image-stream output only (the text stream is discarded after the last block).

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::fast::{layer_norm, rms_norm};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::nn::silu;
use mlx_gen::train::lora::LoraParams;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::rope::LensRope3d;
use super::{join, load_weight, LensTransformerBlock, Linear};

/// The Lens-Turbo / Lens `transformer/config.json` values.
#[derive(Clone, Copy, Debug)]
pub struct LensDitConfig {
    pub patch_size: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub head_dim: i32,
    pub inner_dim: i32,
    pub enc_hidden_dim: i32,
    pub axes_dims_rope: [i32; 3],
    pub num_text_layers: usize,
}

impl LensDitConfig {
    pub fn lens() -> Self {
        Self {
            patch_size: 2,
            in_channels: 128,
            out_channels: 32,
            num_layers: 48,
            num_heads: 24,
            head_dim: 64,
            inner_dim: 1536,
            enc_hidden_dim: 2880,
            axes_dims_rope: [8, 28, 28],
            num_text_layers: 4, // selected_layer_index = (5, 11, 17, 23)
        }
    }
}

/// Sinusoidal timestep projection (`Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0,
/// scale=1000)`): `[B] → [B, 256]` as `[cos | sin]`.
fn timestep_proj(timesteps: &Array) -> Result<Array> {
    let (proj_dim, scale, max_period) = (256usize, 1000f32, 10000f32);
    let half = proj_dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|k| (-(max_period.ln()) * k as f32 / half as f32).exp())
        .collect();
    let freq = Array::from_slice(&freqs, &[1, half as i32]);
    let b = timesteps.shape()[0];
    let emb = multiply(&timesteps.reshape(&[b, 1])?, &freq)?;
    let emb = multiply(&emb, Array::from_slice(&[scale], &[1]))?;
    Ok(concatenate_axis(&[&emb.cos()?, &emb.sin()?], 1)?) // flip_sin_to_cos → [cos, sin]
}

/// `AdaLayerNormContinuous`: affine-less LayerNorm scaled/shifted by `linear(silu(temb))` (the Lens
/// checkpoint's `norm_out.linear` **carries a bias** the reference uses). `[scale | shift]` →
/// `norm(x)·(1+scale) + shift`.
struct AdaLayerNormContinuous {
    linear: Linear,
}

impl AdaLayerNormContinuous {
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(w, &join(prefix, "linear"), true, dtype)?,
        })
    }

    fn forward(&self, x: &Array, temb: &Array) -> Result<Array> {
        let mod_params = self.linear.forward(&silu(temb)?)?; // [B, 2·H]
        let parts = split(&mod_params, 2, 1)?; // scale, shift
        let one = Array::from_slice(&[1.0f32], &[1]);
        let scale = add(&parts[0], &one)?.expand_dims(1)?; // [B, 1, H]
        let shift = parts[1].expand_dims(1)?;
        let normed = layer_norm(x, None, None, 1e-6)?;
        Ok(add(&multiply(&normed, &scale)?, &shift)?)
    }
}

/// Load a biased diffusers `[out, in]` projection as a quantizable [`AdaptableLinear`] (sc-3175).
fn load_biased_adaptable(w: &Weights, prefix: &str, dtype: Dtype) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?;
    let bias = w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?;
    Ok(AdaptableLinear::dense(weight, Some(bias)))
}

/// The Lens denoising DiT.
pub struct LensTransformer {
    img_in: AdaptableLinear,
    txt_norm: Vec<Array>, // per-layer RMSNorm weights (eps 1e-5)
    txt_in: AdaptableLinear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    rope: LensRope3d,
    blocks: Vec<LensTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: AdaptableLinear,
    // `pub(crate)` so the trainer can read `num_layers` to size the per-block checkpoint-target map
    // (sc-5170, mirrors z-image's `pub(crate) cfg`).
    pub(crate) cfg: LensDitConfig,
    dtype: Dtype,
}

impl LensTransformer {
    /// Load from a diffusers `transformer/` weight set at `dtype` (bf16 production / f32 gate).
    pub fn from_weights(w: &Weights, cfg: &LensDitConfig, dtype: Dtype) -> Result<Self> {
        let mut txt_norm = Vec::with_capacity(cfg.num_text_layers);
        for i in 0..cfg.num_text_layers {
            txt_norm.push(load_weight(w, &format!("txt_norm.{i}"), dtype)?);
        }
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(LensTransformerBlock::from_weights(
                w,
                &format!("transformer_blocks.{i}"),
                cfg.num_heads,
                cfg.head_dim,
                dtype,
            )?);
        }
        Ok(Self {
            img_in: load_biased_adaptable(w, "img_in", dtype)?,
            txt_norm,
            txt_in: load_biased_adaptable(w, "txt_in", dtype)?,
            time_linear_1: Linear::load(
                w,
                "time_text_embed.timestep_embedder.linear_1",
                true,
                dtype,
            )?,
            time_linear_2: Linear::load(
                w,
                "time_text_embed.timestep_embedder.linear_2",
                true,
                dtype,
            )?,
            rope: LensRope3d::new(10000.0, cfg.axes_dims_rope),
            blocks,
            norm_out: AdaLayerNormContinuous::from_weights(w, "norm_out", dtype)?,
            proj_out: load_biased_adaptable(w, "proj_out", dtype)?,
            cfg: *cfg,
            dtype,
        })
    }

    /// Quantize the DiT's compute-heavy linears to Q4/Q8 (sc-3175): `img_in`, `txt_in`, `proj_out`,
    /// and every block's attention projections and SwiGLU MLPs. The timestep embedder, the AdaLN
    /// modulations, `norm_out`, and all RMSNorm weights stay full precision. Call **after** any adapter
    /// merge (the `apply_adapters` → `quantize_dit` order in the registry) — adapters are forward-time
    /// residuals over the now-quantized base.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_in.quantize(bits, None)?;
        self.txt_in.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        Ok(())
    }

    /// Toggle SDPA-segment gradient checkpointing across all 48 dual-stream blocks (sc-5170).
    /// Training-only knob: the trainer enables it unconditionally (it is numerically identical to the
    /// dense backward and bounds the seq² attention retention to one block's transient); when
    /// whole-block checkpointing is on, the trainer turns this OFF (the block recompute already covers
    /// attention, and nesting would recompute it twice for no memory win). Inference leaves it off.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
    }

    /// `temb = linear_2(silu(linear_1(proj(t))))`, `[B] → [B, inner]`.
    fn time_embed(&self, timestep: &Array) -> Result<Array> {
        let proj = timestep_proj(timestep)?.as_dtype(self.dtype)?;
        let x = silu(&self.time_linear_1.forward(&proj)?)?;
        self.time_linear_2.forward(&x)
    }

    /// Forward.
    ///
    /// - `hidden_states`: `[B, img_len, in_channels]` patchified image latents (`img_len = frame·h·w`).
    /// - `text_feats`: the `num_text_layers` captured gpt-oss layers, each `[B, txt_len, enc_hidden_dim]`.
    /// - `text_valid`: optional `[B, txt_len]` (1 = valid) → additive joint attention mask; `None` =
    ///   all text valid (no padding), the single-prompt path.
    /// - `timestep`: `[B]` in `[0, 1]`.
    /// - `(frame, h, w)`: the latent grid shape (`img_len = frame·h·w`).
    ///
    /// Returns `[B, img_len, patch²·out_channels]` (= 128) patch-space velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        text_feats: &[Array],
        text_valid: Option<&Array>,
        timestep: &Array,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<Array> {
        assert_eq!(
            text_feats.len(),
            self.cfg.num_text_layers,
            "expected {} text-feature layers, got {}",
            self.cfg.num_text_layers,
            text_feats.len()
        );
        let (b, img_len) = (hidden_states.shape()[0], hidden_states.shape()[1]);
        let txt_len = text_feats[0].shape()[1];

        let mut hidden = self.img_in.forward(hidden_states)?;

        // Multi-layer text front-end: per-layer RMSNorm (eps 1e-5) → channel-concat → txt_in.
        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(rms_norm(feat, &self.txt_norm[i], 1e-5)?);
        }
        let normed_refs: Vec<&Array> = normed.iter().collect();
        let mut enc = self.txt_in.forward(&concatenate_axis(&normed_refs, -1)?)?;

        let temb = self.time_embed(&timestep.as_dtype(self.dtype)?)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward(frame, h, w, txt_len as usize)?;

        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, b, self.dtype)?),
            None => None,
        };

        for block in &self.blocks {
            let (e, hs) = block.forward(
                &hidden,
                &enc,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
            )?;
            enc = e;
            hidden = hs;
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }

    /// Training forward with **per-block gradient checkpointing** (sc-5170). Identical compute to
    /// [`forward`](Self::forward), but each of the 48 dual-stream blocks runs inside an
    /// `mlx::checkpoint` segment whose explicit inputs are the block's two hidden states (image
    /// `hidden`, text `enc`) plus that block's trainable LoRA factors — so the reverse pass
    /// recomputes the block instead of retaining its activations (bounding the first-step working
    /// set), while gradients still flow to the LoRA params. The pre-block front-ends (img_in / text
    /// stack / time embed / RoPE / mask) and the trailing `norm_out` + `proj_out` run normally (any
    /// LoRA there is installed on `self` by the caller and trains through ordinary autograd); the long
    /// dual-stream stack is where the activation memory concentrates, so that is what is checkpointed.
    ///
    /// `params` is the live trainable factor map; `block_local_targets[i]` lists the adapter-routable
    /// LOCAL paths (e.g. `"attn.img_qkv"`) trained on block `i`, in the order their factors are
    /// threaded as checkpoint inputs. Blocks with no trained targets still run checkpointed
    /// (hidden-only inputs) so the whole stack is uniformly recompute-on-backward.
    ///
    /// LoRA-only: the explicit-input re-injection mirrors `install_training_lora`'s `(transpose,
    /// alpha/rank fold, scale = 1)` so the checkpointed block forward is numerically identical to the
    /// installed-adapter path. LoKr's delta is a captured-param reconstruction with no clean
    /// thread-as-input form, so the trainer keeps LoKr on the dense path (caught by the preflight
    /// guard) — mirroring the z-image scope split.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_main_checkpointed(
        &self,
        hidden_states: &Array,
        text_feats: &[Array],
        text_valid: Option<&Array>,
        timestep: &Array,
        frame: usize,
        h: usize,
        w: usize,
        params: &LoraParams,
        block_local_targets: &[Vec<String>],
        alpha: f32,
    ) -> Result<Array> {
        assert_eq!(
            text_feats.len(),
            self.cfg.num_text_layers,
            "expected {} text-feature layers, got {}",
            self.cfg.num_text_layers,
            text_feats.len()
        );
        let (b, img_len) = (hidden_states.shape()[0], hidden_states.shape()[1]);
        let txt_len = text_feats[0].shape()[1];

        // Pre-block front-end — identical to `forward` (not checkpointed).
        let mut hidden = self.img_in.forward(hidden_states)?;
        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(rms_norm(feat, &self.txt_norm[i], 1e-5)?);
        }
        let normed_refs: Vec<&Array> = normed.iter().collect();
        let mut enc = self.txt_in.forward(&concatenate_axis(&normed_refs, -1)?)?;
        let temb = self.time_embed(&timestep.as_dtype(self.dtype)?)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward(frame, h, w, txt_len as usize)?;
        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, b, self.dtype)?),
            None => None,
        };

        // Dual-stream stack — each block checkpointed with its LoRA factors as explicit inputs.
        for (i, block) in self.blocks.iter().enumerate() {
            // Cheap clone (Arrays are refcounted): the closure must OWN its state because the
            // backward recompute runs after this method's frame is gone; a borrow of `self` would
            // dangle. `set_adapters` inside the closure replaces whatever the clone carried with the
            // explicit-input LoRA, so any adapters the caller installed on `self.blocks` are moot.
            let mut blk = block.clone();
            let locals = block_local_targets.get(i).cloned().unwrap_or_default();
            let (te, ic, is, tc, ts) = (
                temb.clone(),
                img_cos.clone(),
                img_sin.clone(),
                txt_cos.clone(),
                txt_sin.clone(),
            );
            let m = mask.clone();

            // Threaded inputs: [hidden, enc, a_0, b_0, a_1, b_1, ...] (raw `[r,in]`/`[out,r]`
            // factors). The two hidden states carry the trainable graph and must be inputs (so grads
            // route through the recompute); the per-block constants (temb/RoPE/mask) are captured.
            let mut inputs: Vec<Array> = Vec::with_capacity(2 + 2 * locals.len());
            inputs.push(hidden.clone());
            inputs.push(enc.clone());
            for local in &locals {
                let ak = format!("transformer_blocks.{i}.{local}.lora_a");
                let bk = format!("transformer_blocks.{i}.{local}.lora_b");
                inputs.push(
                    params
                        .get(ak.as_str())
                        .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {ak}")))?
                        .clone(),
                );
                inputs.push(
                    params
                        .get(bk.as_str())
                        .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {bk}")))?
                        .clone(),
                );
            }

            let alpha_c = alpha;
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                // Reinstall the explicit-input factors with the SAME `(transpose, alpha/rank fold,
                // scale = 1)` `install_training_lora` applies — so the checkpointed block forward is
                // numerically identical to the installed-adapter path, and grads route to `inp`.
                // Dtype-following on the image hidden state (sc-4887 lesson): under the bf16 training
                // cast the f32 factors must join the bf16 stream or every adapted Linear re-promotes
                // the block to f32. No-op in f32 mode; grads flow back f32 through the astype VJP.
                let dt = inp[0].dtype();
                for (j, local) in locals.iter().enumerate() {
                    let a = inp[2 + 2 * j].t().as_dtype(dt)?; // [r,in] -> [in,r]
                    let rank = a.shape()[1] as f32;
                    let b = inp[3 + 2 * j]
                        .t() // [out,r] -> [r,out]
                        .multiply(Array::from_slice(&[alpha_c / rank], &[1]))?
                        .as_dtype(dt)?;
                    let segs: Vec<&str> = local.split('.').collect();
                    blk.adaptable_mut(&segs)
                        .ok_or_else(|| {
                            Exception::custom(format!("checkpoint LoRA target not found: {local}"))
                        })?
                        .set_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
                }
                // The block returns `(enc_out, hidden_out)`; emit `[hidden_out, enc_out]`.
                let (e, hs) = blk
                    .forward(&inp[0], &inp[1], &te, &ic, &is, &tc, &ts, m.as_ref())
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(vec![hs, e])
            });
            let mut out = seg(&inputs)?;
            enc = out.pop().expect("enc output"); // [hidden_out, enc_out] → enc_out
            hidden = out.pop().expect("hidden output"); // → hidden_out
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

impl AdaptableHost for LensTransformer {
    /// Route trained-file (diffusers/peft) paths into the per-block joint-attention adapter targets
    /// (sc-3174): `transformer_blocks.{i}.attn.{img_qkv,txt_qkv,to_out.0,to_add_out}`. Only the
    /// attention projections are adapter targets (the Lens trainer's `DEFAULT_LORA_TARGET_MODULES`);
    /// any other key surfaces as unmatched (loud), never silently dropped.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(mlx_gen::adapters::prefixed_paths(
                &format!("transformer_blocks.{i}"),
                b,
            ));
        }
        out
    }
}

/// Additive joint attention mask `[B, 1, 1, img_len + txt_len]`: image tokens always valid; text
/// positions follow `text_valid` (1 = valid). Padded positions get a large-negative additive term so
/// SDPA's softmax masks them out (`(valid − 1)·BIG`, valid → 0).
fn build_joint_mask(text_valid: &Array, img_len: i32, b: i32, dtype: Dtype) -> Result<Array> {
    let txt_len = text_valid.shape()[1];
    let img_ones = mlx_rs::ops::ones::<f32>(&[b, img_len])?;
    let valid = concatenate_axis(&[&img_ones, &text_valid.as_dtype(Dtype::Float32)?], 1)?;
    let one = Array::from_slice(&[1.0f32], &[1]);
    let big = Array::from_slice(&[1e9f32], &[1]);
    let additive = multiply(&subtract(&valid, &one)?, &big)?; // valid→0, invalid→ -1e9
    Ok(additive
        .reshape(&[b, 1, 1, img_len + txt_len])?
        .as_dtype(dtype)?)
}
