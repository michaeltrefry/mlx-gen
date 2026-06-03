//! FLUX.1 MMDiT transformer. Ports the fork's `flux_transformer` modules:
//! dual-stream joint blocks, single-stream blocks, FLUX RoPE, time/text/guidance embedding, and
//! output AdaLayerNorm.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{gelu_tanh, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::nn::gelu;
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use crate::config::FluxVariant;

const DIM: i32 = 3072;
const HEADS: i32 = 24;
const HEAD_DIM: i32 = 128;
const LN_EPS: f32 = 1e-6;
// QK-norm epsilon. The fork builds these as `nn.RMSNorm(128)` with MLX's *default* eps (1e-5) —
// NOT 1e-6 (which is the AdaLayerNorm's explicit LayerNorm eps). A 1e-6 here is a small uniform
// bias in every attention block that compounds across the 19 joint + 38 single blocks.
const RMS_EPS: f32 = 1e-5;

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

        for block in &self.blocks {
            let (e, h) = block.forward(&hidden, &encoder, &text_embeddings, &rope)?;
            encoder = e;
            hidden = h;
        }

        let txt_seq = encoder.shape()[1];
        let mut joint = concatenate_axis(&[&encoder, &hidden], 1)?;
        for block in &self.single_blocks {
            joint = block.forward(&joint, &text_embeddings, &rope)?;
        }
        let img_seq = hidden.shape()[1];
        let idx = Array::from_slice(
            &(txt_seq..txt_seq + img_seq).collect::<Vec<i32>>(),
            &[img_seq],
        );
        let hidden = joint.take_axis(&idx, 1)?;
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
        let t = Array::from_slice(&[sigma_step], &[1]);
        let mut out = self.timestep.forward(&time_proj(&t)?)?;
        if let Some(g) = &self.guidance {
            let gstep = Array::from_slice(&[guidance], &[1]);
            out = add(&out, &g.forward(&time_proj(&gstep)?)?)?;
        }
        out = add(&out, &self.text.forward(pooled)?)?;
        // Conditioning runs f32 (the whole transformer is f32 activations — the quality target;
        // the fork's bf16 conditioning is the lossy reference, not the goal).
        Ok(out)
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
        let (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.norm1.forward_six(hidden, emb)?;
        let (norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
            self.norm1_context.forward_six(encoder, emb)?;
        let (attn_hidden, attn_context) = self.attn.forward(&norm_hidden, &norm_encoder, rope)?;
        let hidden = apply_norm_ff(
            hidden,
            &attn_hidden,
            &gate_msa,
            &shift_mlp,
            &scale_mlp,
            &gate_mlp,
            &self.ff,
        )?;
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
        let ff = gelu_tanh(&self.proj_mlp.forward(&normed)?)?;
        let out = concatenate_axis(&[&attn, &ff], 2)?;
        let out = multiply(&gate.expand_dims(1)?, &self.proj_out.forward(&out)?)?;
        Ok(add(residual, &out)?)
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

    fn forward(&self, hidden: &Array, encoder: &Array, rope: &RopeTable) -> Result<(Array, Array)> {
        let (q, k, v) = process_qkv(
            hidden,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
        )?;
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
        Ok((
            self.to_out.forward(&out.take_axis(&img_idx, 1)?)?,
            self.to_add_out.forward(&out.take_axis(&txt_idx, 1)?)?,
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
        let normed = add(
            &multiply(&normed, &add(&p[1], scalar(1.0))?.expand_dims(1)?)?,
            &p[0].expand_dims(1)?,
        )?;
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
        let normed = add(
            &multiply(&normed, &add(&p[1], scalar(1.0))?.expand_dims(1)?)?,
            &p[0].expand_dims(1)?,
        )?;
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
        Ok(add(
            &multiply(&normed, &add(&p[0], scalar(1.0))?.expand_dims(1)?)?,
            &p[1].expand_dims(1)?,
        )?)
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
            Activation::Gelu => gelu(x)?,
            // Dtype-preserving, golden-bit-exact tanh-GELU (sc-2779), replacing
            // `mlx_rs::nn::gelu_approximate` (1-ULP f32 `√(2/π)` + bf16→f32 promotion).
            Activation::GeluApprox => gelu_tanh(&x)?,
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
        let omega = |dim: i32| -> Vec<f32> {
            (0..dim / 2)
                .map(|k| 1.0 / self.theta.powf((2 * k) as f32 / dim as f32))
                .collect()
        };
        let freqs = [
            omega(self.axes_dim[0]),
            omega(self.axes_dim[1]),
            omega(self.axes_dim[2]),
        ];
        let half = freqs.iter().map(Vec::len).sum::<usize>();
        let total = txt_seq + latent_h * latent_w;
        let mut cos = vec![1.0_f32; total * half];
        let mut sin = vec![0.0_f32; total * half];
        for h in 0..latent_h {
            for w in 0..latent_w {
                let row = (txt_seq + h * latent_w + w) * half;
                let mut j = 0;
                for &f in &freqs[0] {
                    let a = 0.0 * f;
                    cos[row + j] = a.cos();
                    sin[row + j] = a.sin();
                    j += 1;
                }
                for &f in &freqs[1] {
                    let a = h as f32 * f;
                    cos[row + j] = a.cos();
                    sin[row + j] = a.sin();
                    j += 1;
                }
                for &f in &freqs[2] {
                    let a = w as f32 * f;
                    cos[row + j] = a.cos();
                    sin[row + j] = a.sin();
                    j += 1;
                }
            }
        }
        Ok(RopeTable {
            cos: Array::from_slice(&cos, &[total as i32, half as i32]),
            sin: Array::from_slice(&sin, &[total as i32, half as i32]),
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
    let out0 = subtract(&multiply(&real, &cos)?, &multiply(&imag, &sin)?)?;
    let out1 = add(&multiply(&imag, &cos)?, &multiply(&real, &sin)?)?;
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
    let hidden = add(hidden, &multiply(&gate_msa.expand_dims(1)?, attn)?)?;
    let norm = layer_norm(&hidden, None, None, LN_EPS)?;
    let norm = add(
        &multiply(&norm, &add(scale_mlp, scalar(1.0))?.expand_dims(1)?)?,
        &shift_mlp.expand_dims(1)?,
    )?;
    Ok(add(
        &hidden,
        &multiply(&gate_mlp.expand_dims(1)?, &ff.forward(&norm)?)?,
    )?)
}

fn time_proj(time_steps: &Array) -> Result<Array> {
    let half = 128usize;
    let max_period = 10000f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(max_period.ln()) * i as f32 / half as f32).exp())
        .collect();
    let f = Array::from_slice(&freqs, &[1, half as i32]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_rope_shapes_include_text_and_image_tokens() {
        let r = FluxRope::new().forward(5, 4, 3).unwrap();
        assert_eq!(r.cos.shape(), &[17, 64]);
        assert_eq!(r.sin.shape(), &[17, 64]);
    }
}
