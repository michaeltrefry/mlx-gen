//! FLUX.1 MMDiT transformer. Ports the fork's `flux_transformer` modules:
//! dual-stream joint blocks, single-stream blocks, FLUX RoPE, time/text/guidance embedding, and
//! output AdaLayerNorm.

use std::sync::atomic::{AtomicBool, Ordering};

use mlx_gen::adapters::loader::{BflTarget, LoraRowSlice};
use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{gelu_tanh, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::error::Exception;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::nn::gelu;
use mlx_rs::ops::{add, concatenate_axis, divide, multiply, power, split, subtract, tanh};
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};

use crate::config::FluxVariant;

const DIM: i32 = 3072;
pub(crate) const HEADS: i32 = 24;
pub(crate) const HEAD_DIM: i32 = 128;
const LN_EPS: f32 = 1e-6;
// QK-norm epsilon. The fork builds these as `nn.RMSNorm(128)` with MLX's *default* eps (1e-5) —
// NOT 1e-6 (which is the AdaLayerNorm's explicit LayerNorm eps). A 1e-6 here is a small uniform
// bias in every attention block that compounds across the 19 joint + 38 single blocks.
const RMS_EPS: f32 = 1e-5;

/// sc-2963 (rollout of the Wan sc-2957 template): when on, the FLUX.1 MMDiT's fusable elementwise
/// *glue* — adaLN affine (`norm·(1+scale)+shift`), gated residual (`x+gate·y`), the tanh-GELU FFN
/// activation, and the complex RoPE rotation — runs through `mx.compile` so MLX fuses each chain into
/// a single Metal kernel (vs one kernel per primitive op when eager). The big GEMMs / SDPA / `mx.fast`
/// norms stay eager, and the image-FFN exact `gelu` is **already** internally `mx.compile`'d by mlx-rs
/// (`compiled_gelu`). **Bit-exact** to the eager form (`compile_parity.rs` gates `max|Δ|=0`).
/// **Enabled by the production denoise loop** ([`crate::model`]); **off by default**.
static COMPILE_GLUE: AtomicBool = AtomicBool::new(false);

/// Enable/disable compiled elementwise glue (sc-2963). Process-global; set before the denoise loop.
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.store(on, Ordering::Relaxed);
}

pub(crate) fn compile_glue() -> bool {
    COMPILE_GLUE.load(Ordering::Relaxed)
}

/// adaLN affine `normed·(1+scale)+shift` (`scale`/`shift` pre-broadcast to `[B,1,D]`). One fused
/// kernel when the sc-2963 glue toggle is on. The `1` is cast to `scale`'s dtype before the add — the
/// fork's weak python `1` adopts the (bf16) modulation dtype so `1+scale` rounds in bf16 (coarse near
/// 1.0, spacing ~2⁻⁷); a strong f32 `1` would promote the sum to f32 and break bf16 parity (sc-2787).
fn modulate(normed: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    let f = |(n, s, sh): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        let one = scalar(1.0).as_dtype(s.dtype())?;
        add(&multiply(n, &add(s, &one)?)?, sh)
    };
    if compile_glue() {
        Ok(compile(f, true)((normed, scale, shift))?)
    } else {
        Ok(f((normed, scale, shift))?)
    }
}

/// Gated residual `x + gate·y` (`gate` pre-broadcast to `[B,1,D]`) — one fused kernel when the
/// sc-2963 glue toggle is on; bit-identical to the eager `add(x, gate·y)`.
fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    let f = |(x, g, y): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(x, &multiply(g, y)?)
    };
    if compile_glue() {
        Ok(compile(f, true)((x, gate, y))?)
    } else {
        Ok(f((x, gate, y))?)
    }
}

/// The tanh-GELU FFN activation. Body mirrors [`mlx_gen::nn::gelu_tanh`] exactly (dtype-preserving,
/// f64-host `√(2/π)` — NOT mlx-rs's 1-ULP f32 const, sc-2779); when the sc-2963 glue toggle is on,
/// MLX fuses its ~8 elementwise ops into one kernel. Off ⇒ defers to the core `gelu_tanh`.
fn gelu_ffn(x: &Array) -> Result<Array> {
    if !compile_glue() {
        return gelu_tanh(x);
    }
    let f = |x_: &Array| -> std::result::Result<Array, Exception> {
        let dt = x_.dtype();
        let s = |v: f32| -> std::result::Result<Array, Exception> { scalar(v).as_dtype(dt) };
        let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
        let x3 = power(x_, Array::from_int(3))?;
        let inner = multiply(&add(x_, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
        let gate = add(&tanh(&inner)?, &s(1.0)?)?;
        multiply(&multiply(x_, &s(0.5)?)?, &gate)
    };
    Ok(compile(f, true)(x)?)
}

/// The complex RoPE rotation `(real + imag·i)·(cos + sin·i)` → `(out_real, out_imag)`, in f32. Fused
/// into one kernel when the sc-2963 glue toggle is on (vs 6 eager ops, applied to q and k / block).
fn rope_rotate(real: &Array, imag: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let f = |inp: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
        let (r, i, c, s) = (&inp[0], &inp[1], &inp[2], &inp[3]);
        let out0 = subtract(&multiply(r, c)?, &multiply(i, s)?)?;
        let out1 = add(&multiply(i, c)?, &multiply(r, s)?)?;
        Ok(vec![out0, out1])
    };
    let args = [real.clone(), imag.clone(), cos.clone(), sin.clone()];
    let mut out = if compile_glue() {
        compile(f, true)(&args)?
    } else {
        f(&args)?
    };
    let out1 = out.pop().unwrap();
    let out0 = out.pop().unwrap();
    Ok((out0, out1))
}

pub struct FluxTransformerConfig {
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub supports_guidance: bool,
}

impl FluxTransformerConfig {
    pub fn for_variant(variant: FluxVariant) -> Self {
        Self {
            num_layers: 19,
            num_single_layers: 38,
            supports_guidance: variant.supports_guidance(),
        }
    }
}

/// A per-block additive residual injector for the FLUX DiT **image** stream — the generic seam the
/// PuLID-FLUX id cross-attn (sc-3072, `mlx-gen-pulid`) plugs into. Kept model-agnostic so this crate
/// carries no PuLID-specific code: the DiT just asks "is there a residual to add to the image tokens
/// after block N?" and the injector (which owns the id_embedding + cross-attn modules) answers.
///
/// `after_double` sees the image hidden stream directly; `after_single` sees the image-token tail of
/// the joint stream (the DiT slices it out and writes the residual back). Returning `None` (e.g. at a
/// non-injection block index, or when the id weight is 0) leaves the stream untouched.
pub trait DitImageInjector {
    /// Residual to add to the image stream after double block `block_idx`, or `None`.
    fn after_double(&self, block_idx: usize, img_hidden: &Array) -> Result<Option<Array>>;
    /// Cheap gate so the DiT skips the image-token slice on single blocks with no injection.
    fn injects_after_single(&self, block_idx: usize) -> bool;
    /// Residual to add to the image-token tail after single block `block_idx`, or `None`.
    fn after_single(&self, block_idx: usize, img_tokens: &Array) -> Result<Option<Array>>;

    /// Decoupled IP-Adapter cross-attention residual added to the image **attention output** of
    /// double block `block_idx` (before the block's `gate_msa`/FF), given the block's RMS-normed,
    /// **pre-RoPE** per-head image query `img_q` `[B, HEADS, img_seq, HEAD_DIM]`. Returns the residual
    /// `[B, img_seq, DIM]`, or `None`. Default `None`: this seam is the XLabs FLUX IP-Adapter
    /// (sc-3623); injectors like PuLID-FLUX that use only the post-block residuals above don't
    /// implement it, so the DiT skips the mid-attention image-query slice for them.
    fn double_block_ip(&self, _block_idx: usize, _img_q: &Array) -> Result<Option<Array>> {
        Ok(None)
    }
}

pub struct FluxTransformer {
    x_embedder: AdaptableLinear,
    context_embedder: AdaptableLinear,
    time_text_embed: TimeTextEmbed,
    blocks: Vec<JointBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: AdaptableLinear,
    pos_embed: FluxRope,
}

impl FluxTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &FluxTransformerConfig) -> Result<Self> {
        let p = |s: &str| join(prefix, s);
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(JointBlock::from_weights(
                w,
                &p(&format!("transformer_blocks.{i}")),
            )?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            single_blocks.push(SingleBlock::from_weights(
                w,
                &p(&format!("single_transformer_blocks.{i}")),
            )?);
        }
        Ok(Self {
            x_embedder: linear_from(w, &p("x_embedder"), true)?,
            context_embedder: linear_from(w, &p("context_embedder"), true)?,
            time_text_embed: TimeTextEmbed::from_weights(
                w,
                &p("time_text_embed"),
                cfg.supports_guidance,
            )?,
            blocks,
            single_blocks,
            norm_out: AdaLayerNormContinuous::from_weights(w, &p("norm_out"))?,
            proj_out: linear_from(w, &p("proj_out"), true)?,
            pos_embed: FluxRope::new(),
        })
    }

    /// Quantize every transformer Linear to Q4/Q8 in place (group_size 64), matching the
    /// quantizable FLUX transformer leaves hit by the fork's generic `nn.quantize` predicate.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.x_embedder.quantize(bits, None)?;
        self.context_embedder.quantize(bits, None)?;
        self.time_text_embed.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        for block in &mut self.single_blocks {
            block.quantize(bits)?;
        }
        self.norm_out.quantize(bits)?;
        self.proj_out.quantize(bits, None)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        sigma: f32,
        guidance: f32,
        width: u32,
        height: u32,
    ) -> Result<Array> {
        self.forward_injected(
            hidden_states,
            prompt_embeds,
            pooled_prompt_embeds,
            sigma,
            guidance,
            width,
            height,
            None,
        )
    }

    /// As [`forward`], but with an optional per-block image-stream residual injector — the seam the
    /// PuLID-FLUX id cross-attn (sc-3072) hooks into. `injector = None` is byte-identical to
    /// [`forward`] (it IS the same path), so the plain FLUX render carries zero overhead.
    ///
    /// The injector is consulted after every double block (the image stream `hidden`) and after the
    /// single blocks it opts into (the image-token tail of `joint = cat(encoder, hidden)`), matching
    /// the reference `flux/model.py` PuLID injection points (every 2nd double, every 4th single).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_injected(
        &self,
        hidden_states: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        sigma: f32,
        guidance: f32,
        width: u32,
        height: u32,
        injector: Option<&dyn DitImageInjector>,
    ) -> Result<Array> {
        let mut hidden = self.x_embedder.forward(hidden_states)?;
        let mut encoder = self.context_embedder.forward(prompt_embeds)?;
        let text_embeddings = self.time_text_embed.forward(
            sigma * 1000.0,
            pooled_prompt_embeds,
            guidance * 1000.0,
        )?;
        let rope = self.pos_embed.forward(
            prompt_embeds.shape()[1] as usize,
            (height / 16) as usize,
            (width / 16) as usize,
        )?;

        for (i, block) in self.blocks.iter().enumerate() {
            // The image-query (XLabs IP-Adapter) seam is consulted inside the block's attention; the
            // post-block residual seam (PuLID) is consulted just below. With `injector = None` both
            // are inert and this is byte-identical to the plain path.
            let (e, h) = block.forward_with_ip(
                &hidden,
                &encoder,
                &text_embeddings,
                &rope,
                injector.map(|inj| (inj, i)),
            )?;
            encoder = e;
            hidden = h;
            if let Some(inj) = injector {
                if let Some(r) = inj.after_double(i, &hidden)? {
                    hidden = add(&hidden, &r)?;
                }
            }
        }

        let txt_seq = encoder.shape()[1];
        let img_seq = hidden.shape()[1];
        let img_idx = Array::from_slice(
            &(txt_seq..txt_seq + img_seq).collect::<Vec<i32>>(),
            &[img_seq],
        );
        let mut joint = concatenate_axis(&[&encoder, &hidden], 1)?;
        for (i, block) in self.single_blocks.iter().enumerate() {
            joint = block.forward(&joint, &text_embeddings, &rope)?;
            if let Some(inj) = injector {
                if inj.injects_after_single(i) {
                    let img = joint.take_axis(&img_idx, 1)?;
                    if let Some(r) = inj.after_single(i, &img)? {
                        let txt_idx =
                            Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
                        let txt = joint.take_axis(&txt_idx, 1)?;
                        joint = concatenate_axis(&[&txt, &add(&img, &r)?], 1)?;
                    }
                }
            }
        }
        let hidden = joint.take_axis(&img_idx, 1)?;
        let hidden = self.norm_out.forward(&hidden, &text_embeddings)?;
        self.proj_out.forward(&hidden)
    }

    /// Bisection helper (sc-2345 parity): capture the embedding-stage and first-joint-block
    /// intermediates so the Rust transformer can be diffed against the fork golden stage-by-stage.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_capture(
        &self,
        hidden_states: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        sigma: f32,
        guidance: f32,
        width: u32,
        height: u32,
    ) -> Result<Vec<(String, Array)>> {
        let mut out = Vec::new();
        let hidden = self.x_embedder.forward(hidden_states)?;
        let encoder = self.context_embedder.forward(prompt_embeds)?;
        let text_embeddings = self.time_text_embed.forward(
            sigma * 1000.0,
            pooled_prompt_embeds,
            guidance * 1000.0,
        )?;
        out.push(("hidden0".into(), hidden.clone()));
        out.push(("encoder0".into(), encoder.clone()));
        out.push(("text_embeddings0".into(), text_embeddings.clone()));
        let rope = self.pos_embed.forward(
            prompt_embeds.shape()[1] as usize,
            (height / 16) as usize,
            (width / 16) as usize,
        )?;
        let (e0, h0) = self.blocks[0].forward(&hidden, &encoder, &text_embeddings, &rope)?;
        out.push(("block0_encoder".into(), e0));
        out.push(("block0_hidden".into(), h0));

        let mut hidden = hidden;
        let mut encoder = encoder;
        for block in &self.blocks {
            let (e, h) = block.forward(&hidden, &encoder, &text_embeddings, &rope)?;
            encoder = e;
            hidden = h;
        }
        out.push(("joint_hidden".into(), hidden.clone()));
        out.push(("encoder_joint".into(), encoder.clone()));

        let txt_seq = encoder.shape()[1];
        let img_seq = hidden.shape()[1];
        let mut joint = concatenate_axis(&[&encoder, &hidden], 1)?;
        for block in &self.single_blocks {
            joint = block.forward(&joint, &text_embeddings, &rope)?;
        }
        let idx = Array::from_slice(
            &(txt_seq..txt_seq + img_seq).collect::<Vec<i32>>(),
            &[img_seq],
        );
        out.push(("single_img".into(), joint.take_axis(&idx, 1)?));
        Ok(out)
    }

    /// Stage-injection (sc-2345): run ONLY the single-block stack on externally supplied
    /// post-joint tensors, isolating it from upstream accumulation. Returns the img-token slice.
    #[doc(hidden)]
    pub fn debug_single_stack(
        &self,
        encoder: &Array,
        hidden: &Array,
        text_embeddings: &Array,
        latent_h: usize,
        latent_w: usize,
        num_blocks: usize,
    ) -> Result<Array> {
        let txt_seq = encoder.shape()[1];
        let img_seq = hidden.shape()[1];
        let rope = self
            .pos_embed
            .forward(txt_seq as usize, latent_h, latent_w)?;
        let mut joint = concatenate_axis(&[encoder, hidden], 1)?;
        let n = if num_blocks == 0 {
            self.single_blocks.len()
        } else {
            num_blocks
        };
        for block in self.single_blocks.iter().take(n) {
            joint = block.forward(&joint, text_embeddings, &rope)?;
        }
        let idx = Array::from_slice(
            &(txt_seq..txt_seq + img_seq).collect::<Vec<i32>>(),
            &[img_seq],
        );
        Ok(joint.take_axis(&idx, 1)?)
    }

    /// Expose the RoPE cos/sin table for a given (txt_seq, latent_h, latent_w) — to diff against
    /// the fork's `EmbedND` output.
    #[doc(hidden)]
    pub fn debug_rope(
        &self,
        txt_seq: usize,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<(Array, Array)> {
        let r = self.pos_embed.forward(txt_seq, latent_h, latent_w)?;
        Ok((r.cos, r.sin))
    }

    /// Decompose single block 0 into its sub-ops (norm / attn / ff) on injected joint output.
    #[doc(hidden)]
    pub fn debug_single_block0(
        &self,
        encoder: &Array,
        hidden: &Array,
        text_embeddings: &Array,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Vec<(String, Array)>> {
        let txt_seq = encoder.shape()[1];
        let rope = self
            .pos_embed
            .forward(txt_seq as usize, latent_h, latent_w)?;
        let joint = concatenate_axis(&[encoder, hidden], 1)?;
        let b = &self.single_blocks[0];
        let (normed, _gate) = b.norm.forward_three(&joint, text_embeddings)?;
        let attn = b.attn.forward(&normed, &rope)?;
        let ff = gelu_tanh(&b.proj_mlp.forward(&normed)?)?;
        Ok(vec![
            ("sb0_norm".into(), normed),
            ("sb0_attn".into(), attn),
            ("sb0_ff".into(), ff),
        ])
    }
}

struct TimeTextEmbed {
    timestep: MlpEmbedder,
    text: MlpEmbedder,
    guidance: Option<MlpEmbedder>,
}

impl TimeTextEmbed {
    fn from_weights(w: &Weights, prefix: &str, supports_guidance: bool) -> Result<Self> {
        Ok(Self {
            timestep: MlpEmbedder::from_weights(w, &join(prefix, "timestep_embedder"))?,
            text: MlpEmbedder::from_weights(w, &join(prefix, "text_embedder"))?,
            guidance: if supports_guidance {
                Some(MlpEmbedder::from_weights(
                    w,
                    &join(prefix, "guidance_embedder"),
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, sigma_step: f32, pooled: &Array, guidance: f32) -> Result<Array> {
        // Match the fork's bf16 conditioning path (sc-2787). `time_step`/`guidance` are cast to the
        // model precision (bf16) BEFORE the sinusoidal `time_proj` (so they carry bf16 rounding —
        // `Transformer.compute_text_embeddings` does `.astype(config.precision)`); the pooled CLIP
        // embedding enters as bf16; and the summed conditioning is cast back to bf16
        // (`conditioning.astype(ModelConfig.precision)`). `time_proj` then upcasts to f32 for sin/cos,
        // exactly like the fork. The transformer's main residual stream stays f32 (fork latents +
        // prompt_embeds are f32), so only the modulation path is bf16.
        let bf16 = Dtype::Bfloat16;
        let t = Array::from_slice(&[sigma_step], &[1]).as_dtype(bf16)?;
        let mut out = self.timestep.forward(&time_proj(&t)?)?;
        if let Some(g) = &self.guidance {
            let gstep = Array::from_slice(&[guidance], &[1]).as_dtype(bf16)?;
            out = add(&out, &g.forward(&time_proj(&gstep)?)?)?;
        }
        out = add(&out, &self.text.forward(&pooled.as_dtype(bf16)?)?)?;
        Ok(out.as_dtype(bf16)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.timestep.quantize(bits)?;
        self.text.quantize(bits)?;
        if let Some(guidance) = self.guidance.as_mut() {
            guidance.quantize(bits)?;
        }
        Ok(())
    }
}

struct MlpEmbedder {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
}

impl MlpEmbedder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_1: linear_from(w, &join(prefix, "linear_1"), true)?,
            linear_2: linear_from(w, &join(prefix, "linear_2"), true)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.linear_2.forward(&silu(&self.linear_1.forward(x)?)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear_1.quantize(bits, None)?;
        self.linear_2.quantize(bits, None)?;
        Ok(())
    }
}

struct JointBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: JointAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl JointBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::from_weights(w, &join(prefix, "norm1"), 6)?,
            norm1_context: AdaLayerNormZero::from_weights(w, &join(prefix, "norm1_context"), 6)?,
            attn: JointAttention::from_weights(w, &join(prefix, "attn"))?,
            ff: FeedForward::from_weights(
                w,
                &join(prefix, "ff"),
                "net.0.proj",
                "net.2",
                Activation::Gelu,
            )?,
            ff_context: FeedForward::from_weights(
                w,
                &join(prefix, "ff_context"),
                "net.0.proj",
                "net.2",
                Activation::GeluApprox,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        emb: &Array,
        rope: &RopeTable,
    ) -> Result<(Array, Array)> {
        self.forward_with_ip(hidden, encoder, emb, rope, None)
    }

    /// As [`forward`], but consulting the XLabs IP-Adapter image-query seam
    /// ([`DitImageInjector::double_block_ip`]) inside the joint attention. `ip = None` is identical
    /// to [`forward`].
    fn forward_with_ip(
        &self,
        hidden: &Array,
        encoder: &Array,
        emb: &Array,
        rope: &RopeTable,
        ip: Option<(&dyn DitImageInjector, usize)>,
    ) -> Result<(Array, Array)> {
        let (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.norm1.forward_six(hidden, emb)?;
        let (norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
            self.norm1_context.forward_six(encoder, emb)?;
        let (attn_hidden, attn_context, ip_residual) =
            self.attn
                .forward_with_ip(&norm_hidden, &norm_encoder, rope, ip)?;
        let hidden = apply_norm_ff(
            hidden,
            &attn_hidden,
            &gate_msa,
            &shift_mlp,
            &scale_mlp,
            &gate_mlp,
            &self.ff,
        )?;
        // XLabs IP-Adapter: add the decoupled-cross-attention residual RAW (ungated) to the final
        // block output, after the FF residual — diffusers `hidden_states = hidden_states +
        // ip_attn_output` (transformer_flux.py:477). It bypasses `gate_msa` and the FF input.
        let hidden = match ip_residual {
            Some(r) => add(&hidden, &r.as_dtype(hidden.dtype())?)?,
            None => hidden,
        };
        let encoder = apply_norm_ff(
            encoder,
            &attn_context,
            &c_gate_msa,
            &c_shift_mlp,
            &c_scale_mlp,
            &c_gate_mlp,
            &self.ff_context,
        )?;
        Ok((encoder, hidden))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.norm1.quantize(bits)?;
        self.norm1_context.quantize(bits)?;
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.ff_context.quantize(bits)?;
        Ok(())
    }
}

struct SingleBlock {
    norm: AdaLayerNormZero,
    attn: SingleAttention,
    proj_mlp: AdaptableLinear,
    proj_out: AdaptableLinear,
}

impl SingleBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: AdaLayerNormZero::from_weights(w, &join(prefix, "norm"), 3)?,
            attn: SingleAttention::from_weights(w, &join(prefix, "attn"))?,
            proj_mlp: linear_from(w, &join(prefix, "proj_mlp"), true)?,
            proj_out: linear_from(w, &join(prefix, "proj_out"), true)?,
        })
    }

    fn forward(&self, hidden: &Array, emb: &Array, rope: &RopeTable) -> Result<Array> {
        let residual = hidden;
        let (normed, gate) = self.norm.forward_three(hidden, emb)?;
        let attn = self.attn.forward(&normed, rope)?;
        let ff = gelu_ffn(&self.proj_mlp.forward(&normed)?)?;
        let out = concatenate_axis(&[&attn, &ff], 2)?;
        let proj = self.proj_out.forward(&out)?;
        gated(residual, &gate.expand_dims(1)?, &proj)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.norm.quantize(bits)?;
        self.attn.quantize(bits)?;
        self.proj_mlp.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        Ok(())
    }
}

struct JointAttention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    add_q: AdaptableLinear,
    add_k: AdaptableLinear,
    add_v: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
}

impl JointAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            to_q: linear_from(w, &join(prefix, "to_q"), true)?,
            to_k: linear_from(w, &join(prefix, "to_k"), true)?,
            to_v: linear_from(w, &join(prefix, "to_v"), true)?,
            to_out: linear_from(w, &join(prefix, "to_out.0"), true)?,
            add_q: linear_from(w, &join(prefix, "add_q_proj"), true)?,
            add_k: linear_from(w, &join(prefix, "add_k_proj"), true)?,
            add_v: linear_from(w, &join(prefix, "add_v_proj"), true)?,
            to_add_out: linear_from(w, &join(prefix, "to_add_out"), true)?,
            norm_q: w.require(&join(prefix, "norm_q.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "norm_k.weight"))?.clone(),
            norm_added_q: w.require(&join(prefix, "norm_added_q.weight"))?.clone(),
            norm_added_k: w.require(&join(prefix, "norm_added_k.weight"))?.clone(),
        })
    }

    /// When `ip = Some((injector, block_idx))` computes the XLabs IP-Adapter
    /// decoupled-cross-attention residual for the image stream and returns it as the third tuple
    /// element (it is **not** folded into the image attention output here). The IP query is the
    /// image stream's **RMS-normed, pre-RoPE** query (`norm_q(to_q(img))`, before the text concat +
    /// RoPE) — matching diffusers' `FluxIPAdapterAttnProcessor` (`ip_query = query` captured
    /// *before* `apply_rotary_emb`; the position-less IP keys are attended by the un-rotated query).
    /// The caller adds the residual **raw (ungated), after the FF residual**, to the block output —
    /// diffusers' `hidden_states = hidden_states + ip_attn_output` (the IP term bypasses `gate_msa`
    /// and the FF input entirely). `ip = None` returns `None` and is byte-identical to the plain
    /// joint attention.
    fn forward_with_ip(
        &self,
        hidden: &Array,
        encoder: &Array,
        rope: &RopeTable,
        ip: Option<(&dyn DitImageInjector, usize)>,
    ) -> Result<(Array, Array, Option<Array>)> {
        let (q, k, v) = process_qkv(
            hidden,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
        )?;
        // Pre-RoPE, pre-concat image query — diffusers' `ip_query` (captured before RoPE). Cloned
        // only when an IP injector is present (otherwise zero overhead).
        let ip_img_q = ip.map(|_| q.clone());
        let (eq, ek, ev) = process_qkv(
            encoder,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
        )?;
        let q = concatenate_axis(&[&eq, &q], 2)?;
        let k = concatenate_axis(&[&ek, &k], 2)?;
        let v = concatenate_axis(&[&ev, &v], 2)?;
        let (q, k) = apply_rope(&q, &k, rope)?;
        let out = attention(&q, &k, &v)?;
        let txt_seq = encoder.shape()[1];
        let img_seq = hidden.shape()[1];
        let txt_idx = Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
        let img_idx = Array::from_slice(
            &(txt_seq..txt_seq + img_seq).collect::<Vec<i32>>(),
            &[img_seq],
        );
        let attn_img = self.to_out.forward(&out.take_axis(&img_idx, 1)?)?;
        // The IP residual is computed here but returned separately — the block adds it raw to the
        // final output (after gate_msa + FF), per diffusers. Folding it into `attn_img` (which is
        // then gated by `gate_msa` and fed into the FF input) would both suppress it where the gate
        // is small and distort the velocity, breaking resemblance + saturating true_cfg.
        let ip_residual = match ip {
            Some((inj, block_idx)) => inj.double_block_ip(
                block_idx,
                ip_img_q
                    .as_ref()
                    .expect("ip_img_q captured when ip present"),
            )?,
            None => None,
        };
        Ok((
            attn_img,
            self.to_add_out.forward(&out.take_axis(&txt_idx, 1)?)?,
            ip_residual,
        ))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_q.quantize(bits, None)?;
        self.to_k.quantize(bits, None)?;
        self.to_v.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        self.add_q.quantize(bits, None)?;
        self.add_k.quantize(bits, None)?;
        self.add_v.quantize(bits, None)?;
        self.to_add_out.quantize(bits, None)?;
        Ok(())
    }
}

struct SingleAttention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
}

impl SingleAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            to_q: linear_from(w, &join(prefix, "to_q"), true)?,
            to_k: linear_from(w, &join(prefix, "to_k"), true)?,
            to_v: linear_from(w, &join(prefix, "to_v"), true)?,
            norm_q: w.require(&join(prefix, "norm_q.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "norm_k.weight"))?.clone(),
        })
    }

    fn forward(&self, hidden: &Array, rope: &RopeTable) -> Result<Array> {
        let (q, k, v) = process_qkv(
            hidden,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
        )?;
        let (q, k) = apply_rope(&q, &k, rope)?;
        attention(&q, &k, &v)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_q.quantize(bits, None)?;
        self.to_k.quantize(bits, None)?;
        self.to_v.quantize(bits, None)?;
        Ok(())
    }
}

struct AdaLayerNormZero {
    linear: AdaptableLinear,
    chunks: usize,
}

impl AdaLayerNormZero {
    fn from_weights(w: &Weights, prefix: &str, chunks: usize) -> Result<Self> {
        Ok(Self {
            linear: linear_from(w, &join(prefix, "linear"), true)?,
            chunks,
        })
    }

    fn forward_six(
        &self,
        hidden: &Array,
        emb: &Array,
    ) -> Result<(Array, Array, Array, Array, Array)> {
        debug_assert_eq!(self.chunks, 6);
        let p = split(&self.linear.forward(&silu(emb)?)?, 6, 1)?;
        let normed = layer_norm(hidden, None, None, LN_EPS)?;
        let normed = modulate(&normed, &p[1].expand_dims(1)?, &p[0].expand_dims(1)?)?;
        Ok((
            normed,
            p[2].clone(),
            p[3].clone(),
            p[4].clone(),
            p[5].clone(),
        ))
    }

    fn forward_three(&self, hidden: &Array, emb: &Array) -> Result<(Array, Array)> {
        debug_assert_eq!(self.chunks, 3);
        let p = split(&self.linear.forward(&silu(emb)?)?, 3, 1)?;
        let normed = layer_norm(hidden, None, None, LN_EPS)?;
        let normed = modulate(&normed, &p[1].expand_dims(1)?, &p[0].expand_dims(1)?)?;
        Ok((normed, p[2].clone()))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }
}

struct AdaLayerNormContinuous {
    linear: AdaptableLinear,
}

impl AdaLayerNormContinuous {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: linear_from(w, &join(prefix, "linear"), false)?,
        })
    }

    fn forward(&self, x: &Array, emb: &Array) -> Result<Array> {
        let p = split(&self.linear.forward(&silu(emb)?)?, 2, 1)?;
        let normed = layer_norm(x, None, None, LN_EPS)?;
        modulate(&normed, &p[0].expand_dims(1)?, &p[1].expand_dims(1)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }
}

enum Activation {
    Gelu,
    GeluApprox,
}

struct FeedForward {
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
    activation: Activation,
}

impl FeedForward {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        in_name: &str,
        out_name: &str,
        activation: Activation,
    ) -> Result<Self> {
        Ok(Self {
            linear1: linear_from(w, &join(prefix, in_name), true)?,
            linear2: linear_from(w, &join(prefix, out_name), true)?,
            activation,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.linear1.forward(x)?;
        let x = match self.activation {
            // mlx-rs's exact `gelu` is itself `mx.compile`'d internally (`compiled_gelu`) — already
            // a single fused kernel, so the sc-2963 glue toggle leaves it alone.
            Activation::Gelu => gelu(x)?,
            // Dtype-preserving, golden-bit-exact tanh-GELU (sc-2779), replacing
            // `mlx_rs::nn::gelu_approximate` (1-ULP f32 `√(2/π)` + bf16→f32 promotion). sc-2963
            // fuses its ~8 ops into one kernel when the glue toggle is on.
            Activation::GeluApprox => gelu_ffn(&x)?,
        };
        self.linear2.forward(&x)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear1.quantize(bits, None)?;
        self.linear2.quantize(bits, None)?;
        Ok(())
    }
}

struct FluxRope {
    theta: f32,
    axes_dim: [i32; 3],
}

struct RopeTable {
    cos: Array,
    sin: Array,
}

impl FluxRope {
    fn new() -> Self {
        Self {
            theta: 10000.0,
            axes_dim: [16, 56, 56],
        }
    }

    fn forward(&self, txt_seq: usize, latent_h: usize, latent_w: usize) -> Result<RopeTable> {
        let total = (txt_seq + latent_h * latent_w) as i32;
        // Per-axis positions (the fork's `ids[..., i]`): axis 0 is all-zero; axis 1 = latent row (h),
        // axis 2 = latent col (w); text tokens sit at position 0 on every axis.
        let mut pos1 = vec![0f32; total as usize];
        let mut pos2 = vec![0f32; total as usize];
        for h in 0..latent_h {
            for w in 0..latent_w {
                let row = txt_seq + h * latent_w + w;
                pos1[row] = h as f32;
                pos2[row] = w as f32;
            }
        }
        // Build each axis's cos/sin with MLX ops (sc-2787), bit-matching the fork's `EmbedND`:
        // `omega = 1/(theta**scale)`, `out = pos·omega`, then `mx.cos`/`mx.sin`. The host libm trig +
        // `powf` differ from MLX by ~4e-7, which the chaotic 57-block stack amplifies into the only
        // remaining transformer parity gap (every kernel op is otherwise bit-identical at 0.31.2).
        let axis = |dim: i32, pos: &[f32]| -> Result<(Array, Array)> {
            let half = dim / 2;
            let scale: Vec<f32> = (0..half).map(|k| (2 * k) as f32 / dim as f32).collect();
            let omega = divide(
                scalar(1.0),
                power(scalar(self.theta), Array::from_slice(&scale, &[1, half]))?,
            )?;
            let out = multiply(Array::from_slice(pos, &[total, 1]), &omega)?;
            Ok((out.cos()?, out.sin()?))
        };
        let zeros = vec![0f32; total as usize];
        let (c0, s0) = axis(self.axes_dim[0], &zeros)?;
        let (c1, s1) = axis(self.axes_dim[1], &pos1)?;
        let (c2, s2) = axis(self.axes_dim[2], &pos2)?;
        Ok(RopeTable {
            cos: concatenate_axis(&[&c0, &c1, &c2], 1)?,
            sin: concatenate_axis(&[&s0, &s1, &s2], 1)?,
        })
    }
}

fn process_qkv(
    x: &Array,
    q: &AdaptableLinear,
    k: &AdaptableLinear,
    v: &AdaptableLinear,
    norm_q: &Array,
    norm_k: &Array,
) -> Result<(Array, Array, Array)> {
    let b = x.shape()[0];
    let s = x.shape()[1];
    let q = q
        .forward(x)?
        .reshape(&[b, s, HEADS, HEAD_DIM])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = k
        .forward(x)?
        .reshape(&[b, s, HEADS, HEAD_DIM])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = v
        .forward(x)?
        .reshape(&[b, s, HEADS, HEAD_DIM])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let q = rms_norm(&q.as_dtype(Dtype::Float32)?, norm_q, RMS_EPS)?.as_dtype(q_dtype)?;
    let k = rms_norm(&k.as_dtype(Dtype::Float32)?, norm_k, RMS_EPS)?.as_dtype(k_dtype)?;
    Ok((q, k, v))
}

fn attention(q: &Array, k: &Array, v: &Array) -> Result<Array> {
    let b = q.shape()[0];
    let y = scaled_dot_product_attention(q, k, v, (HEAD_DIM as f32).powf(-0.5), None, None)?;
    Ok(y.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, -1, DIM])?)
}

fn apply_rope(q: &Array, k: &Array, rope: &RopeTable) -> Result<(Array, Array)> {
    Ok((apply_rope_one(q, rope)?, apply_rope_one(k, rope)?))
}

fn apply_rope_one(x: &Array, rope: &RopeTable) -> Result<Array> {
    let sh = x.shape();
    let (b, heads, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;
    let x5 = x
        .as_dtype(Dtype::Float32)?
        .reshape(&[b, heads, seq, half, 2])?;
    let p = split(&x5, 2, 4)?;
    let real = p[0].reshape(&[b, heads, seq, half])?;
    let imag = p[1].reshape(&[b, heads, seq, half])?;
    let cos = rope.cos.reshape(&[1, 1, seq, half])?;
    let sin = rope.sin.reshape(&[1, 1, seq, half])?;
    let (out0, out1) = rope_rotate(&real, &imag, &cos, &sin)?;
    Ok(
        concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
            .reshape(&[b, heads, seq, hd])?
            .as_dtype(Dtype::Float32)?,
    )
}

fn apply_norm_ff(
    hidden: &Array,
    attn: &Array,
    gate_msa: &Array,
    shift_mlp: &Array,
    scale_mlp: &Array,
    gate_mlp: &Array,
    ff: &FeedForward,
) -> Result<Array> {
    let hidden = gated(hidden, &gate_msa.expand_dims(1)?, attn)?;
    let norm = layer_norm(&hidden, None, None, LN_EPS)?;
    let norm = modulate(
        &norm,
        &scale_mlp.expand_dims(1)?,
        &shift_mlp.expand_dims(1)?,
    )?;
    let ff_out = ff.forward(&norm)?;
    gated(&hidden, &gate_mlp.expand_dims(1)?, &ff_out)
}

fn time_proj(time_steps: &Array) -> Result<Array> {
    let half = 128i32;
    let max_period = 10000f64;
    // Build the sinusoidal freqs with MLX ops (sc-2787) to bit-match the fork's `_time_proj`:
    // `exp(-log(max_period) * arange(half) / half)`. Host `exp`/`arange` differ from MLX by ~1e-7,
    // which flips one element of the bf16 conditioning by a ULP and seeds the joint stack. `-log` is
    // taken in f64 then cast to f32 (the fork's `math.log` is f64, weak-cast at the MLX multiply).
    let neg_log = -(max_period.ln()) as f32;
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let exponent = divide(
        multiply(Array::from_slice(&arange, &[1, half]), scalar(neg_log))?,
        scalar(half as f32),
    )?;
    let f = exponent.exp()?;
    let emb = multiply(
        &time_steps
            .reshape(&[time_steps.shape()[0], 1])?
            .as_dtype(Dtype::Float32)?,
        &f,
    )?;
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    Ok(concatenate_axis(&[&cos, &sin], 1)?)
}

fn linear_from(w: &Weights, prefix: &str, has_bias: bool) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let bias = if has_bias {
        Some(w.require(&format!("{prefix}.bias"))?.clone())
    } else {
        None
    };
    Ok(AdaptableLinear::dense(weight, bias))
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

// ---- LoRA/LoKr adapter routing (sc-2657) ------------------------------------------------------
//
// The Rust analog of the fork's `FluxLoRAMapping`: map trained-file module paths to the crate's
// `AdaptableLinear` fields. Joint (`transformer_blocks`) and single (`single_transformer_blocks`) block
// linears — INCLUDING the adaLN modulation linears (`norm1.linear`, `norm1_context.linear`, single-block
// `norm.linear`), which `FluxLoRAMapping` targets and a real kohya FLUX LoRA carries
// (`*_mod_lin`/`modulation_lin`) — PLUS the top-level global projections: `x_embedder`,
// `context_embedder`, `proj_out`, `norm_out.linear`, and the three `time_text_embed.*_embedder.linear_{1,2}`.
// The fork's `FluxLoRAMapping` omits those globals, but the production few-step acceleration LoRAs (e.g.
// ByteDance Hyper-FLUX, PEFT/diffusers format) DO train them — so covering them makes the Rust strictly
// more capable than the fork, the same correctness-over-parity call as SDXL Complete coverage (sc-2671);
// under the strict no-silent-drop policy a Hyper-FLUX file would otherwise error on the unmatched global
// keys (sc-2908). The VAE + T5/CLIP text encoders are not adapter targets. `adaptable_mut` accepts the
// diffusers checkpoint spelling AND the fork's renamed `model_path` spelling where they differ
// (`ff.net.0.proj` ≡ `ff.linear1`, `to_out.0` ≡ `to_out`), since the fork's `possible_*_patterns` list
// both. Per-file LoKr/LoRA dispatch, prefix detection, kohya flattening, BFL fused→split, stacking, and
// the strict no-silent-drop policy are the shared core seam (sc-2534/2618/2743), exactly as Z-Image
// (sc-2602), Qwen (sc-2528), and FLUX.2 (sc-2646) use it.

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            // diffusers checkpoint naming (`ff.net.0.proj`/`ff.net.2`) AND the fork's renamed
            // `linear1`/`linear2` `model_path` spelling both address these Linears (the fork lists both).
            ["net", "0", "proj"] | ["linear1"] => Some(&mut self.linear1),
            ["net", "2"] | ["linear2"] => Some(&mut self.linear2),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        // The diffusers spelling — what a kohya file flattening a diffusers checkpoint contains.
        ["net.0.proj", "net.2"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for JointAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Image stream `to_q/k/v`/`to_out.0`; text stream `add_{q,k,v}_proj` → `add_{q,k,v}` and
        // `to_add_out`. The fork accepts both the bare `to_out` and the HF `to_out.0` (diffusers wraps
        // the output projection in a `Sequential[Linear, Dropout]`); both address this Linear.
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out"] | ["to_out", "0"] => Some(&mut self.to_out),
            ["add_q_proj"] => Some(&mut self.add_q),
            ["add_k_proj"] => Some(&mut self.add_k),
            ["add_v_proj"] => Some(&mut self.add_v),
            ["to_add_out"] => Some(&mut self.to_add_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

impl AdaptableHost for SingleAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["to_q", "to_k", "to_v"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for JointBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            ["ff_context", rest @ ..] => self.ff_context.adaptable_mut(rest),
            ["norm1", "linear"] => Some(&mut self.norm1.linear),
            ["norm1_context", "linear"] => Some(&mut self.norm1_context.linear),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.extend(prefixed_paths("ff", &self.ff));
        out.extend(prefixed_paths("ff_context", &self.ff_context));
        out.push("norm1.linear".into());
        out.push("norm1_context.linear".into());
        out
    }
}

impl AdaptableHost for SingleBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["proj_mlp"] => Some(&mut self.proj_mlp),
            ["proj_out"] => Some(&mut self.proj_out),
            ["norm", "linear"] => Some(&mut self.norm.linear),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.push("proj_mlp".into());
        out.push("proj_out".into());
        out.push("norm.linear".into());
        out
    }
}

impl AdaptableHost for MlpEmbedder {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["linear_1"] => Some(&mut self.linear_1),
            ["linear_2"] => Some(&mut self.linear_2),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["linear_1", "linear_2"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for TimeTextEmbed {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["timestep_embedder", rest @ ..] => self.timestep.adaptable_mut(rest),
            ["text_embedder", rest @ ..] => self.text.adaptable_mut(rest),
            // `guidance_embedder` exists only on dev (Hyper-FLUX is a dev LoRA); on schnell it is
            // absent, so a guidance_embedder key correctly fails to resolve.
            ["guidance_embedder", rest @ ..] => self.guidance.as_mut()?.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("timestep_embedder", &self.timestep);
        out.extend(prefixed_paths("text_embedder", &self.text));
        if let Some(g) = &self.guidance {
            out.extend(prefixed_paths("guidance_embedder", g));
        }
        out
    }
}

impl AdaptableHost for FluxTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["single_transformer_blocks", n, rest @ ..] => self
                .single_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            // Top-level global projections — fork-omitted but trained by PEFT acceleration LoRAs
            // (Hyper-FLUX), sc-2908. `norm_out` is the final adaLN-continuous modulation linear.
            ["x_embedder"] => Some(&mut self.x_embedder),
            ["context_embedder"] => Some(&mut self.context_embedder),
            ["proj_out"] => Some(&mut self.proj_out),
            ["norm_out", "linear"] => Some(&mut self.norm_out.linear),
            ["time_text_embed", rest @ ..] => self.time_text_embed.adaptable_mut(rest),
            _ => None,
        }
    }

    /// kohya-reachable targets (sc-2618): the diffusers-named joint + single block linears (incl. the
    /// adaLN modulation linears) plus the top-level global projections (sc-2908). The fork's
    /// `FluxLoRAMapping` itself lists no diffusers-flattened
    /// `lora_unet_transformer_blocks_*` pattern (its kohya `lora_unet_` patterns are all the BFL fused
    /// form below), so this is the core's cross-family diffusers-flattened-kohya superset — proven
    /// generically by the core `kohya_equiv_to_peft_bit_exact` gate + the drift/collision tests, and
    /// consistent with Z-Image/Qwen/FLUX.2. A real FLUX kohya/ComfyUI file uses the BFL naming and is
    /// dispatched to [`bfl_targets`](Self::bfl_targets) first (precise), never here.
    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("transformer_blocks.{i}"), b));
        }
        for (i, b) in self.single_blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("single_transformer_blocks.{i}"), b));
        }
        // Top-level global projections (sc-2908): fork-omitted, trained by PEFT acceleration LoRAs.
        out.push("x_embedder".into());
        out.push("context_embedder".into());
        out.push("proj_out".into());
        out.push("norm_out.linear".into());
        out.extend(prefixed_paths("time_text_embed", &self.time_text_embed));
        out
    }

    /// BFL / ComfyUI fused→split targets (sc-2743), the Rust analog of the fork's
    /// `FluxLoRAMapping._get_bfl_{transformer,single_transformer}_block_targets`. Unlike FLUX.2, the
    /// fork's FLUX.1 BFL patterns use ONLY the kohya `lora_unet_<flat>` spelling (no `diffusion_model.`/
    /// `base_model.model.` BFL variants). Two fused→split shapes:
    /// - joint `double_blocks.{n}.{img,txt}_attn.qkv` → 3-way EQUAL split into `attn.to_q/to_k/to_v`
    ///   (img) / `add_{q,k,v}_proj` (txt), via `Chunk{n:3}` (up) / `ChunkIfDivisible{n:3}` (down).
    /// - single `single_blocks.{n}.linear1` → **4-way** split into `attn.to_q/to_k/to_v` + `proj_mlp`,
    ///   via `Dims{[q,k,v,mlp]}` (up) / `ChunkIfDivisible{n:4}` (down). The dims are config-derived
    ///   (`[DIM, DIM, DIM, 4·DIM]` = `[3072,3072,3072,12288]`), matching the fork `_split_qkv_mlp_*`.
    ///
    /// Everything else is a plain rename (no slice), including the adaLN modulation linears
    /// (`*_mod_lin`/`modulation_lin` → `norm1.linear`/`norm1_context.linear`/`norm.linear`). Byte-faithful
    /// to the fork `LoraTransforms` (guarded by tests). Full surface = 19×14 + 38×6 = 494 targets.
    fn bfl_targets(&self) -> Vec<BflTarget> {
        let mut out = Vec::new();
        // FLUX.1 single-block `linear1` = q/k/v (each `DIM`) + mlp (`4·DIM`) fused along dim 0.
        let single_dims = vec![DIM, DIM, DIM, 4 * DIM];

        // Joint (double) blocks.
        for i in 0..self.blocks.len() {
            // Fused qkv → 3-way split: img → to_{q,k,v}; txt → add_{q,k,v}_proj.
            for (stream, dst) in [
                ("img", ["to_q", "to_k", "to_v"]),
                ("txt", ["add_q_proj", "add_k_proj", "add_v_proj"]),
            ] {
                let flat = format!("double_blocks_{i}_{stream}_attn_qkv");
                for idx in 0..3i32 {
                    out.push(bfl_split(
                        &format!("transformer_blocks.{i}.attn.{}", dst[idx as usize]),
                        &flat,
                        LoraRowSlice::Chunk { n: 3, index: idx },
                        LoraRowSlice::ChunkIfDivisible { n: 3, index: idx },
                    ));
                }
            }
            // attn output proj (rename): img.proj → to_out.0; txt.proj → to_add_out.
            out.push(bfl_rename(
                &format!("transformer_blocks.{i}.attn.to_out.0"),
                &format!("double_blocks_{i}_img_attn_proj"),
            ));
            out.push(bfl_rename(
                &format!("transformer_blocks.{i}.attn.to_add_out"),
                &format!("double_blocks_{i}_txt_attn_proj"),
            ));
            // MLP (rename): img_mlp.{0,2} → ff.linear{1,2}; txt_mlp.{0,2} → ff_context.linear{1,2}.
            for (stream, ff) in [("img", "ff"), ("txt", "ff_context")] {
                out.push(bfl_rename(
                    &format!("transformer_blocks.{i}.{ff}.linear1"),
                    &format!("double_blocks_{i}_{stream}_mlp_0"),
                ));
                out.push(bfl_rename(
                    &format!("transformer_blocks.{i}.{ff}.linear2"),
                    &format!("double_blocks_{i}_{stream}_mlp_2"),
                ));
            }
            // adaLN modulation (rename): img_mod_lin → norm1.linear; txt_mod_lin → norm1_context.linear.
            out.push(bfl_rename(
                &format!("transformer_blocks.{i}.norm1.linear"),
                &format!("double_blocks_{i}_img_mod_lin"),
            ));
            out.push(bfl_rename(
                &format!("transformer_blocks.{i}.norm1_context.linear"),
                &format!("double_blocks_{i}_txt_mod_lin"),
            ));
        }

        // Single blocks.
        for i in 0..self.single_blocks.len() {
            // Fused linear1 → 4-way split: attn.to_{q,k,v} + proj_mlp.
            let flat = format!("single_blocks_{i}_linear1");
            let dst = ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"];
            for idx in 0..4i32 {
                out.push(bfl_split(
                    &format!("single_transformer_blocks.{i}.{}", dst[idx as usize]),
                    &flat,
                    LoraRowSlice::Dims {
                        dims: single_dims.clone(),
                        index: idx,
                    },
                    LoraRowSlice::ChunkIfDivisible { n: 4, index: idx },
                ));
            }
            // linear2 → proj_out (rename); modulation_lin → norm.linear (rename).
            out.push(bfl_rename(
                &format!("single_transformer_blocks.{i}.proj_out"),
                &format!("single_blocks_{i}_linear2"),
            ));
            out.push(bfl_rename(
                &format!("single_transformer_blocks.{i}.norm.linear"),
                &format!("single_blocks_{i}_modulation_lin"),
            ));
        }

        out
    }
}

/// Every BFL/ComfyUI source-key spelling for one FLUX.1 block linear. Unlike FLUX.2, the fork's
/// `FluxLoRAMapping` BFL patterns use ONLY the kohya `lora_unet_<flat>` spelling, so each role has a
/// single key. Returns `(up, down, alpha)` — `lora_up`≡`lora_B`, `lora_down`≡`lora_A`.
fn bfl_keys(flat: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    (
        vec![format!("lora_unet_{flat}.lora_up.weight")],
        vec![format!("lora_unet_{flat}.lora_down.weight")],
        vec![format!("lora_unet_{flat}.alpha")],
    )
}

/// A plain BFL rename (no row-slice): the source factors map straight to `target_path`.
fn bfl_rename(target_path: &str, flat: &str) -> BflTarget {
    let (up_keys, down_keys, alpha_keys) = bfl_keys(flat);
    BflTarget {
        target_path: target_path.to_string(),
        up_keys,
        down_keys,
        alpha_keys,
        up_slice: None,
        down_slice: None,
    }
}

/// A BFL fused→split target: the source up/down factors are row-sliced into `target_path`.
fn bfl_split(
    target_path: &str,
    flat: &str,
    up_slice: LoraRowSlice,
    down_slice: LoraRowSlice,
) -> BflTarget {
    let (up_keys, down_keys, alpha_keys) = bfl_keys(flat);
    BflTarget {
        target_path: target_path.to_string(),
        up_keys,
        down_keys,
        alpha_keys,
        up_slice: Some(up_slice),
        down_slice: Some(down_slice),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::adapters::{install_adapter, Adapter};
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};
    use mlx_rs::ops::array_eq;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn flux_rope_shapes_include_text_and_image_tokens() {
        let r = FluxRope::new().forward(5, 4, 3).unwrap();
        assert_eq!(r.cos.shape(), &[17, 64]);
        assert_eq!(r.sin.shape(), &[17, 64]);
    }

    // ---- sc-2963: the compiled elementwise glue is bit-identical to the eager glue ---------------
    //
    // FLUX.1's transformer has a fixed inner dim (no tiny config) and no committed forward fixture,
    // so the compiled-vs-eager `max|Δ|=0` invariant is gated directly on the private glue helpers the
    // forward composes (`modulate` / `gated` / `gelu_ffn` / `rope_rotate`) at the production dtypes
    // (f32 main stream, bf16 modulation/gate conditioning, sc-2787). `mx.compile` must not perturb the
    // result. Run: `cargo test -p mlx-gen-flux --lib compiled_glue`.
    fn rnd(shape: &[i32], dt: mlx_rs::Dtype) -> Array {
        let k = mlx_rs::random::key(0).unwrap();
        let x = mlx_rs::random::normal::<f32>(shape, None, None, Some(&k)).unwrap();
        let x = if dt == mlx_rs::Dtype::Float32 {
            x
        } else {
            x.as_dtype(dt).unwrap()
        };
        mlx_rs::transforms::eval([&x]).unwrap();
        x
    }

    fn max_abs(a: &Array, b: &Array) -> f32 {
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
        mlx_rs::ops::max(&d, None)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .item::<f32>()
    }

    #[test]
    fn compiled_glue_bit_identical_to_eager() {
        use mlx_rs::Dtype::{Bfloat16, Float32};
        let d = 256i32; // narrower than the real DIM — bit-exactness is dim-independent
        let (b, s) = (2i32, 16i32);

        // modulate: f32 normed × bf16 (1+scale) + bf16 shift (the bf16-conditioning path).
        let normed = rnd(&[b, s, d], Float32);
        let scale = rnd(&[b, 1, d], Bfloat16);
        let shift = rnd(&[b, 1, d], Bfloat16);
        set_compile_glue(false);
        let e = modulate(&normed, &scale, &shift).unwrap();
        set_compile_glue(true);
        let c = modulate(&normed, &scale, &shift).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&c, &e), 0.0, "modulate compiled vs eager");

        // gated: f32 x + (bf16 gate · f32 y).
        let x = rnd(&[b, s, d], Float32);
        let gate = rnd(&[b, 1, d], Bfloat16);
        let y = rnd(&[b, s, d], Float32);
        set_compile_glue(false);
        let e = gated(&x, &gate, &y).unwrap();
        set_compile_glue(true);
        let c = gated(&x, &gate, &y).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&c, &e), 0.0, "gated compiled vs eager");

        // gelu_ffn: f32 FFN activation (the FLUX main-stream dtype). Eager branch defers to the core
        // `gelu_tanh`, so this also proves the compiled body matches the core op exactly.
        let h = rnd(&[b, s, 4 * d], Float32);
        set_compile_glue(false);
        let e = gelu_ffn(&h).unwrap();
        set_compile_glue(true);
        let c = gelu_ffn(&h).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&c, &e), 0.0, "gelu_ffn compiled vs eager");

        // rope_rotate: f32 complex rotation (RoPE runs in f32).
        let half = 64i32;
        let real = rnd(&[b, HEADS, s, half], Float32);
        let imag = rnd(&[b, HEADS, s, half], Float32);
        let cos = rnd(&[1, 1, s, half], Float32);
        let sin = rnd(&[1, 1, s, half], Float32);
        set_compile_glue(false);
        let (e0, e1) = rope_rotate(&real, &imag, &cos, &sin).unwrap();
        set_compile_glue(true);
        let (c0, c1) = rope_rotate(&real, &imag, &cos, &sin).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&c0, &e0), 0.0, "rope_rotate real compiled vs eager");
        assert_eq!(max_abs(&c1, &e1), 0.0, "rope_rotate imag compiled vs eager");
    }

    // ---- sc-2657 adapter routing (FluxLoRAMapping → field translation) ------------------------

    fn dummy_lin() -> AdaptableLinear {
        AdaptableLinear::dense(Array::from_slice(&[0.0f32], &[1, 1]), None)
    }
    fn dummy_arr() -> Array {
        Array::from_slice(&[1.0f32], &[1])
    }
    fn ada_zero(chunks: usize) -> AdaLayerNormZero {
        AdaLayerNormZero {
            linear: dummy_lin(),
            chunks,
        }
    }
    fn joint_attn() -> JointAttention {
        JointAttention {
            to_q: dummy_lin(),
            to_k: dummy_lin(),
            to_v: dummy_lin(),
            to_out: dummy_lin(),
            add_q: dummy_lin(),
            add_k: dummy_lin(),
            add_v: dummy_lin(),
            to_add_out: dummy_lin(),
            norm_q: dummy_arr(),
            norm_k: dummy_arr(),
            norm_added_q: dummy_arr(),
            norm_added_k: dummy_arr(),
        }
    }
    fn feed_forward() -> FeedForward {
        FeedForward {
            linear1: dummy_lin(),
            linear2: dummy_lin(),
            activation: Activation::Gelu,
        }
    }
    fn joint_block() -> JointBlock {
        JointBlock {
            norm1: ada_zero(6),
            norm1_context: ada_zero(6),
            attn: joint_attn(),
            ff: feed_forward(),
            ff_context: feed_forward(),
        }
    }
    fn single_block() -> SingleBlock {
        SingleBlock {
            norm: ada_zero(3),
            attn: SingleAttention {
                to_q: dummy_lin(),
                to_k: dummy_lin(),
                to_v: dummy_lin(),
                norm_q: dummy_arr(),
                norm_k: dummy_arr(),
            },
            proj_mlp: dummy_lin(),
            proj_out: dummy_lin(),
        }
    }
    fn mlp_embedder() -> MlpEmbedder {
        MlpEmbedder {
            linear_1: dummy_lin(),
            linear_2: dummy_lin(),
        }
    }
    /// A FLUX.1 transformer of dummy 1×1 linears with `n_double` joint + `n_single` single blocks —
    /// enough to exercise routing/BFL/kohya enumeration without weights (forward is never called).
    fn test_transformer(n_double: usize, n_single: usize) -> FluxTransformer {
        FluxTransformer {
            x_embedder: dummy_lin(),
            context_embedder: dummy_lin(),
            time_text_embed: TimeTextEmbed {
                timestep: mlp_embedder(),
                text: mlp_embedder(),
                guidance: Some(mlp_embedder()),
            },
            blocks: (0..n_double).map(|_| joint_block()).collect(),
            single_blocks: (0..n_single).map(|_| single_block()).collect(),
            norm_out: AdaLayerNormContinuous {
                linear: dummy_lin(),
            },
            proj_out: dummy_lin(),
            pos_embed: FluxRope::new(),
        }
    }

    fn resolves(host: &mut impl AdaptableHost, path: &str) -> bool {
        let segs: Vec<&str> = path.split('.').collect();
        host.adaptable_mut(&segs).is_some()
    }

    /// The full fork `FluxLoRAMapping` surface — joint + single block linears INCLUDING the adaLN
    /// modulation linears (the diffusers AND fork-renamed spellings) — resolves; off-surface rejects.
    #[test]
    fn routing_covers_full_fork_surface() {
        let mut t = test_transformer(19, 38);
        let double = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj", // diffusers spelling
            "ff.net.2",
            "ff.linear1", // fork model_path spelling (both resolve)
            "ff.linear2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "norm1.linear",
            "norm1_context.linear",
        ];
        for i in [0usize, 9, 18] {
            for tgt in double {
                let p = format!("transformer_blocks.{i}.{tgt}");
                assert!(resolves(&mut t, &p), "expected {p} to resolve");
            }
        }
        let single = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "proj_mlp",
            "proj_out",
            "norm.linear",
        ];
        for i in [0usize, 19, 37] {
            for tgt in single {
                let p = format!("single_transformer_blocks.{i}.{tgt}");
                assert!(resolves(&mut t, &p), "expected {p} to resolve");
            }
        }
        // sc-2908: the top-level global projections now resolve (fork-omitted, trained by PEFT
        // acceleration LoRAs like Hyper-FLUX). The file path uses the `*_embedder` spelling.
        for p in [
            "x_embedder",
            "context_embedder",
            "proj_out",
            "norm_out.linear",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "time_text_embed.text_embedder.linear_1",
            "time_text_embed.text_embedder.linear_2",
            "time_text_embed.guidance_embedder.linear_1",
            "time_text_embed.guidance_embedder.linear_2",
        ] {
            assert!(
                resolves(&mut t, p),
                "expected global {p} to resolve (sc-2908)"
            );
        }
        for p in [
            "time_text_embed.timestep.linear_1", // internal field name, not the file's *_embedder
            "transformer_blocks.19.attn.to_q",   // out of range (19 joint blocks: 0..18)
            "single_transformer_blocks.38.proj_out", // out of range (38 single blocks: 0..37)
            "transformer_blocks.0.attn.add_q",   // internal field, not the file's add_q_proj
            "transformer_blocks.0.attn.qkv",     // fused name — FLUX.1 model is split
            "single_transformer_blocks.0.attn.to_out", // single attn has no separate output proj
        ] {
            assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
        }
    }

    /// `adaptable_paths()` (the kohya-reachable surface) drift guard: every enumerated path resolves,
    /// flattens to a collision-free stem, and the count matches the full surface — 19×14 + 38×6 = 494
    /// block linears + 10 top-level globals (x_embedder, context_embedder, proj_out, norm_out.linear,
    /// and the 3 `time_text_embed.*_embedder.linear_{1,2}` pairs), sc-2908.
    #[test]
    fn adaptable_paths_resolve_and_flatten_uniquely() {
        let t = test_transformer(19, 38);
        let paths = t.adaptable_paths();
        assert_eq!(
            paths.len(),
            19 * 14 + 38 * 6 + 10,
            "full kohya surface count (blocks + globals)"
        );
        let mut probe = test_transformer(19, 38);
        for p in &paths {
            assert!(
                resolves(&mut probe, p),
                "enumerated {p} does not resolve via adaptable_mut"
            );
        }
        let flat: std::collections::BTreeSet<String> =
            paths.iter().map(|p| p.replace('.', "_")).collect();
        assert_eq!(
            flat.len(),
            paths.len(),
            "two enumerated paths collide when flattened to a kohya stem"
        );
    }

    /// The full `bfl_targets()` surface (sc-2743): count = 494, every target resolves on the tree,
    /// target paths are unique, and the single-block `linear1` uses the 4-way `Dims` split while the
    /// double-block qkv uses the 3-way `Chunk`/`ChunkIfDivisible`.
    #[test]
    fn bfl_targets_full_surface() {
        let mut t = test_transformer(19, 38);
        let targets = t.bfl_targets();
        assert_eq!(targets.len(), 19 * 14 + 38 * 6, "full BFL surface count");
        let mut paths = std::collections::BTreeSet::new();
        for tg in &targets {
            assert!(paths.insert(tg.target_path.clone()), "duplicate BFL target");
            assert!(
                resolves(&mut t, &tg.target_path),
                "BFL target {} does not resolve",
                tg.target_path
            );
        }
        // Single-block to_q: 4-way Dims split with FLUX.1's [DIM,DIM,DIM,4·DIM] dims, index 0.
        let sq = targets
            .iter()
            .find(|tg| tg.target_path == "single_transformer_blocks.0.attn.to_q")
            .unwrap();
        match &sq.up_slice {
            Some(LoraRowSlice::Dims { dims, index }) => {
                assert_eq!(dims, &vec![DIM, DIM, DIM, 4 * DIM]);
                assert_eq!(*index, 0);
            }
            other => panic!("single to_q up_slice should be Dims, got {other:?}"),
        }
        // proj_mlp is the 4th (index 3) slice of the same fused linear1.
        let pm = targets
            .iter()
            .find(|tg| tg.target_path == "single_transformer_blocks.0.proj_mlp")
            .unwrap();
        assert!(matches!(
            &pm.up_slice,
            Some(LoraRowSlice::Dims { index: 3, .. })
        ));
        // Double-block img qkv → to_k: 3-way Chunk (up) / ChunkIfDivisible (down), index 1.
        let dk = targets
            .iter()
            .find(|tg| tg.target_path == "transformer_blocks.0.attn.to_k")
            .unwrap();
        assert!(matches!(
            &dk.up_slice,
            Some(LoraRowSlice::Chunk { n: 3, index: 1 })
        ));
        assert!(matches!(
            &dk.down_slice,
            Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: 1 })
        ));
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_flux_adapter_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn lora_ab(adapters: &[Adapter]) -> (Array, Array) {
        match adapters {
            [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
            _ => panic!("expected exactly one LoRA adapter, got {}", adapters.len()),
        }
    }

    /// sc-2743: a BFL *fused* `img_attn.qkv` LoRA reconstructs the BYTE-IDENTICAL `to_q/to_k/to_v`
    /// adapters as the equivalent diffusers split-target file — the 3-way split. No weights needed
    /// (the row-slice operates on the source factors, not the base), so this is a CI gate.
    #[test]
    fn bfl_fused_qkv_matches_diffusers_split() {
        let none = None as Option<&HashMap<String, String>>;
        let (inner, inp, r) = (3072i32, 8i32, 2i32);
        let head = |seed: i32| -> Vec<f32> {
            (0..inner * r)
                .map(|i| (((i + seed) % 17) as f32 - 8.0) * 0.001)
                .collect()
        };
        let (hq, hk, hv) = (head(0), head(5), head(11));
        let mut fused = hq.clone();
        fused.extend_from_slice(&hk);
        fused.extend_from_slice(&hv);
        let up_fused = Array::from_slice(&fused, &[3 * inner, r]);
        let up_q = Array::from_slice(&hq, &[inner, r]);
        let up_k = Array::from_slice(&hk, &[inner, r]);
        let up_v = Array::from_slice(&hv, &[inner, r]);
        // Shared down (rank 2 not divisible by 3 → ChunkIfDivisible returns the whole tensor).
        let down = Array::from_slice(
            &(0..r * inp)
                .map(|i| (i as f32 - 4.0) * 0.002)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let bfl_path = tmp("bfl_qkv.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight",
                    &up_fused,
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                    &down,
                ),
                ("lora_unet_double_blocks_0_img_attn_qkv.alpha", &alpha),
            ],
            none,
            &bfl_path,
        )
        .unwrap();
        let peft_path = tmp("bfl_qkv_split_peft.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
                    &up_q,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
                    &down,
                ),
                ("transformer.transformer_blocks.0.attn.to_q.alpha", &alpha),
                (
                    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
                    &up_k,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",
                    &down,
                ),
                ("transformer.transformer_blocks.0.attn.to_k.alpha", &alpha),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight",
                    &up_v,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
                    &down,
                ),
                ("transformer.transformer_blocks.0.attn.to_v.alpha", &alpha),
            ],
            none,
            &peft_path,
        )
        .unwrap();

        let mut tb = test_transformer(1, 0);
        let rb = crate::adapters::apply_flux_adapters(
            &mut tb,
            &[AdapterSpec {
                path: bfl_path,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(rb.applied, 3, "fused qkv → 3 split targets");
        assert!(rb.unmatched_paths.is_empty());

        let mut tp = test_transformer(1, 0);
        crate::adapters::apply_flux_adapters(
            &mut tp,
            &[AdapterSpec {
                path: peft_path,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();

        for tgt in ["to_q", "to_k", "to_v"] {
            let segs = ["transformer_blocks", "0", "attn", tgt];
            let (ba, bb) = lora_ab(tb.adaptable_mut(&segs).unwrap().adapters());
            let (pa, pb) = lora_ab(tp.adaptable_mut(&segs).unwrap().adapters());
            assert!(
                array_eq(&ba, &pa, false).unwrap().item::<bool>()
                    && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
                "BFL split ≠ diffusers split at {tgt}"
            );
        }
    }

    /// sc-2743: the FLUX.1-specific 4-way `single_blocks.{n}.linear1` → `to_q/to_k/to_v` + `proj_mlp`
    /// split reconstructs byte-identically to the diffusers split-target file (the `Dims` boundaries).
    #[test]
    fn bfl_fused_single_linear1_matches_diffusers_split() {
        let none = None as Option<&HashMap<String, String>>;
        let (r, inp) = (2i32, 8i32);
        let dims = [DIM, DIM, DIM, 4 * DIM]; // q,k,v,mlp
        let block = |rows: i32, seed: i32| -> Vec<f32> {
            (0..rows * r)
                .map(|i| (((i + seed) % 19) as f32 - 9.0) * 0.001)
                .collect()
        };
        let parts: Vec<Vec<f32>> = dims
            .iter()
            .enumerate()
            .map(|(j, &d)| block(d, j as i32 * 7))
            .collect();
        let mut fused = Vec::new();
        for p in &parts {
            fused.extend_from_slice(p);
        }
        let up_fused = Array::from_slice(&fused, &[dims.iter().sum::<i32>(), r]);
        let down = Array::from_slice(
            &(0..r * inp)
                .map(|i| (i as f32 - 4.0) * 0.002)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let alpha = Array::from_slice(&[8.0f32], &[1]);

        let bfl_path = tmp("bfl_single.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_single_blocks_0_linear1.lora_up.weight",
                    &up_fused,
                ),
                ("lora_unet_single_blocks_0_linear1.lora_down.weight", &down),
                ("lora_unet_single_blocks_0_linear1.alpha", &alpha),
            ],
            none,
            &bfl_path,
        )
        .unwrap();

        let split_targets = ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"];
        let mut peft: Vec<(String, Array)> = Vec::new();
        for (j, tgt) in split_targets.iter().enumerate() {
            let up = Array::from_slice(&parts[j], &[dims[j], r]);
            peft.push((
                format!("transformer.single_transformer_blocks.0.{tgt}.lora_B.weight"),
                up,
            ));
            peft.push((
                format!("transformer.single_transformer_blocks.0.{tgt}.lora_A.weight"),
                down.clone(),
            ));
            peft.push((
                format!("transformer.single_transformer_blocks.0.{tgt}.alpha"),
                alpha.clone(),
            ));
        }
        let peft_path = tmp("bfl_single_split_peft.safetensors");
        Array::save_safetensors(
            peft.iter()
                .map(|(k, v)| (k.as_str(), v))
                .collect::<Vec<_>>(),
            none,
            &peft_path,
        )
        .unwrap();

        let mut tb = test_transformer(0, 1);
        let rb = crate::adapters::apply_flux_adapters(
            &mut tb,
            &[AdapterSpec {
                path: bfl_path,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(rb.applied, 4, "fused linear1 → 4 split targets");
        assert!(rb.unmatched_paths.is_empty());

        let mut tp = test_transformer(0, 1);
        crate::adapters::apply_flux_adapters(
            &mut tp,
            &[AdapterSpec {
                path: peft_path,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();

        for tgt in split_targets {
            let path = format!("single_transformer_blocks.0.{tgt}");
            let segs: Vec<&str> = path.split('.').collect();
            let (ba, bb) = lora_ab(tb.adaptable_mut(&segs).unwrap().adapters());
            let (pa, pb) = lora_ab(tp.adaptable_mut(&segs).unwrap().adapters());
            assert!(
                array_eq(&ba, &pa, false).unwrap().item::<bool>()
                    && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
                "BFL 4-way split ≠ diffusers split at {tgt}"
            );
        }
    }

    /// scale-0 application is a bit-exact no-op at a resolved Linear, and a mixed LoRA+LoKr spec list
    /// installs both; an off-surface target errors (strict no-silent-drop).
    #[test]
    fn scale_zero_noop_mixed_stack_and_strict() {
        // scale-0: a no-op adapter installed on a resolved linear leaves its forward unchanged.
        let mut t = test_transformer(1, 0);
        let x = Array::from_slice(&[1.0f32], &[1, 1]);
        let base = t
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
            .unwrap()
            .forward(&x)
            .unwrap();
        install_adapter(
            &mut t,
            "transformer_blocks.0.attn.to_q",
            Adapter::Lora {
                a: Array::from_slice(&[0.0f32], &[1, 1]),
                b: Array::from_slice(&[0.0f32], &[1, 1]),
                scale: 0.0,
            },
        )
        .unwrap();
        let after = t
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(
            array_eq(&base, &after, false).unwrap().item::<bool>(),
            "scale-0 adapter must be a bit-exact no-op"
        );

        // Mixed LoRA + LoKr spec list targeting two distinct FLUX.1 modules applies both.
        let lora_path = tmp("mix_lora.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
                    &Array::from_slice(&[0.1f32], &[1, 1]),
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight",
                    &Array::from_slice(&[0.2f32], &[1, 1]),
                ),
            ],
            None as Option<&HashMap<String, String>>,
            &lora_path,
        )
        .unwrap();
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let lokr_path = tmp("mix_lokr.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "transformer_blocks.0.ff.linear1.lokr_w1",
                    &Array::from_slice(&[0.5f32], &[1, 1]),
                ),
                (
                    "transformer_blocks.0.ff.linear1.lokr_w2",
                    &Array::from_slice(&[0.3f32], &[1, 1]),
                ),
            ],
            Some(&meta),
            &lokr_path,
        )
        .unwrap();
        let mut t2 = test_transformer(1, 0);
        let report = crate::adapters::apply_flux_adapters(
            &mut t2,
            &[
                AdapterSpec {
                    path: lora_path,
                    scale: 1.0,
                    kind: AdapterKind::Lora,
                    pass_scales: None,
                    moe_expert: None,
                },
                AdapterSpec {
                    path: lokr_path,
                    scale: 1.0,
                    kind: AdapterKind::Lokr,
                    pass_scales: None,
                    moe_expert: None,
                },
            ],
        )
        .unwrap();
        assert_eq!(report.applied, 2, "mixed LoRA + LoKr both apply");
        assert!(report.unmatched_paths.is_empty());

        // Strict no-silent-drop: an off-surface target errors.
        let miss = tmp("miss.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "transformer.transformer_blocks.0.nope.lora_A.weight",
                    &Array::from_slice(&[0.1f32], &[1, 1]),
                ),
                (
                    "transformer.transformer_blocks.0.nope.lora_B.weight",
                    &Array::from_slice(&[0.2f32], &[1, 1]),
                ),
            ],
            None as Option<&HashMap<String, String>>,
            &miss,
        )
        .unwrap();
        let mut t3 = test_transformer(1, 0);
        assert!(crate::adapters::apply_flux_adapters(
            &mut t3,
            &[AdapterSpec {
                path: miss,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .is_err());
    }
}
