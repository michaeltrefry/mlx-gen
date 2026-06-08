//! The full Qwen-Image MMDiT. Port of the fork's `QwenTransformer`: project image latents and the
//! (RMSNorm'd) text embeddings into the inner dim, build the timestep conditioning + 3D RoPE, run
//! 60 dual-stream blocks, then `AdaLayerNormContinuous` + `proj_out` back to patch space.
//!
//! Weight keys follow the fork's *internal* module tree (e.g. `transformer_blocks.{i}.img_mod_linear`,
//! `…attn.attn_to_out.0`, `…img_ff.mlp_in`); the on-disk diffusers→internal remapping is applied by
//! the loader (`remap_transformer_keys`). Per-block weights are exercised by the synthetic-weight
//! block parity test; the full 60-layer forward is validated end-to-end against the image golden.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis, multiply, split};
use mlx_rs::Array;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::array::host_i32;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::time_text_embed::TimeTextEmbed;
use super::{linear_from, AdaLayerNormContinuous, QwenRope3d, QwenTransformerBlock};

pub struct QwenTransformerConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub head_dim: i32,
    pub joint_attention_dim: i32,
    pub patch_size: i32,
    pub txt_norm_eps: f32,
    /// Qwen-Image-Edit-2511 `transformer/config.json` sets `zero_cond_t: true` — the conditioning
    /// image latent tokens are modulated as clean (timestep 0) in every block. `false` for T2I /
    /// the superseded 2509 edit weights (then a no-op).
    pub zero_cond_t: bool,
}

impl QwenTransformerConfig {
    pub fn qwen_image() -> Self {
        Self {
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            num_heads: 24,
            head_dim: 128,
            joint_attention_dim: 3584,
            patch_size: 2,
            txt_norm_eps: 1e-6,
            zero_cond_t: false,
        }
    }

    /// The Qwen-Image-Edit-2511 transformer — identical to T2I but with `zero_cond_t` on.
    pub fn qwen_image_edit() -> Self {
        Self {
            zero_cond_t: true,
            ..Self::qwen_image()
        }
    }

    pub fn inner_dim(&self) -> i32 {
        self.num_heads * self.head_dim
    }
}

pub struct QwenTransformer {
    img_in: AdaptableLinear,
    txt_norm_w: Array,
    txt_in: AdaptableLinear,
    time_text_embed: TimeTextEmbed,
    blocks: Vec<QwenTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: AdaptableLinear,
    rope: QwenRope3d,
    eps: f32,
    zero_cond_t: bool,
}

/// The Qwen adapter key→module map — the Rust analog of the fork's `QwenLoRAMapping`. Every fork
/// target is per-block (`transformer_blocks.{i}.{img_mod.1,txt_mod.1,attn.*,img_mlp.*,txt_mlp.*}`);
/// there are no global targets (`img_in`/`txt_in`/`proj_out` are not trained). Adapter files address
/// modules by their trained (diffusers) path, routed here to the block hosts.
impl AdaptableHost for QwenTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    /// kohya-reachable targets (sc-2618): the per-block joint-attention + stream-MLP linears, in
    /// trained-file naming. Qwen's fork mapping has no global targets, so the full kohya surface is
    /// the `transformer_blocks.{i}.*` set.
    fn adaptable_paths(&self) -> Vec<String> {
        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(i, b)| prefixed_paths(&format!("transformer_blocks.{i}"), b))
            .collect()
    }
}

impl QwenTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &QwenTransformerConfig) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(QwenTransformerBlock::from_weights(
                w,
                &p(&format!("transformer_blocks.{i}")),
                cfg.num_heads,
                cfg.head_dim,
            )?);
        }
        Ok(Self {
            img_in: linear_from(w, &p("img_in"), true)?,
            txt_norm_w: w.require(&p("txt_norm.weight"))?.clone(),
            txt_in: linear_from(w, &p("txt_in"), true)?,
            time_text_embed: TimeTextEmbed::from_weights(w, &p("time_text_embed"))?,
            blocks,
            norm_out: AdaLayerNormContinuous::from_weights(w, &p("norm_out"))?,
            proj_out: linear_from(w, &p("proj_out"), true)?,
            rope: QwenRope3d::qwen_image(),
            eps: cfg.txt_norm_eps,
            zero_cond_t: cfg.zero_cond_t,
        })
    }

    /// Quantize every transformer Linear to Q4/Q8 in place (group_size 64), the mlx-rs equivalent
    /// of the fork's `nn.quantize(transformer, bits=…)`. The text encoder + VAE stay dense (they
    /// have no quantizable Linears in the fork's predicate path).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_in.quantize(bits, None)?;
        self.txt_in.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        self.time_text_embed.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        self.norm_out.quantize(bits)?;
        Ok(())
    }

    /// `hidden_states`: packed image latents `[B, img_seq, in_channels]`. For T2I `img_seq =
    /// latent_h·latent_w` and `cond_grids` is empty; for Qwen-Image-Edit the noise latents are
    /// concatenated with the packed reference latents, and `cond_grids` lists each reference's
    /// `(latent_h, latent_w)` so the RoPE covers `[noise] + references` (the dual-latent path).
    /// `encoder_hidden_states`: text features `[B, txt_seq, joint_attention_dim]`. `timestep`: the
    /// scheduler sigma. Returns the velocity over the **full** sequence `[B, img_seq, patch²·out]`
    /// (the caller slices back to the noise prefix for Edit).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        encoder_hidden_states_mask: Option<&Array>,
        timestep: f32,
        latent_h: usize,
        latent_w: usize,
        cond_grids: &[(usize, usize)],
    ) -> Result<Array> {
        self.forward_control(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            timestep,
            latent_h,
            latent_w,
            cond_grids,
            None,
            0.0,
        )
    }

    /// [`forward`](Self::forward) with optional ControlNet residual injection (epic 3401). Identical
    /// to the T2I/Edit forward, plus: after base block `i` the residual
    /// `controlnet_residuals[i / interval]` (scaled by `control_scale`) is added to the image stream,
    /// where `interval = ceil(num_layers / num_residuals)` — the diffusers
    /// `QwenImageTransformer2DModel` `index_block // interval_control` pattern (60 base blocks, 5
    /// control residuals → interval 12). `controlnet_residuals = None` is **byte-identical** to the
    /// plain forward (the T2I/Edit parity path is unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_control(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        encoder_hidden_states_mask: Option<&Array>,
        timestep: f32,
        latent_h: usize,
        latent_w: usize,
        cond_grids: &[(usize, usize)],
        controlnet_residuals: Option<&[Array]>,
        control_scale: f32,
    ) -> Result<Array> {
        let b = hidden_states.shape()[0];
        let img_seq = hidden_states.shape()[1];
        let txt_seq = encoder_hidden_states.shape()[1];

        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = rms_norm(encoder_hidden_states, &self.txt_norm_w, self.eps)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let ts = Array::from_slice(&vec![timestep; b as usize], &[b]);

        // zero_cond_t (Qwen-Image-Edit-2511): double the timestep -> [t, 0] so the conditioning
        // image tokens can be modulated as clean (t 0) in every block, while the noise tokens + text
        // stream keep the real timestep. `modulate_index` (0 = noise, 1 = cond) drives the per-token
        // select inside each block; with no reference (T2I) the flag is a no-op.
        let zero_cond = self.zero_cond_t && !cond_grids.is_empty();
        let (text_emb, modulate_index) = if zero_cond {
            let zeros = Array::from_slice(&vec![0f32; b as usize], &[b]);
            let ts2 = concatenate_axis(&[&ts, &zeros], 0)?;
            let emb = self.time_text_embed.forward(&ts2)?;
            let idx = build_modulate_index(b, latent_h, latent_w, cond_grids);
            (emb, Some(idx))
        } else {
            (self.time_text_embed.forward(&ts)?, None)
        };

        // RoPE over the noise grid followed by each reference grid (empty for T2I).
        let mut shapes = Vec::with_capacity(1 + cond_grids.len());
        shapes.push((latent_h, latent_w));
        shapes.extend_from_slice(cond_grids);
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward_multi(&shapes, txt_seq as usize)?;
        let mask = build_joint_mask(encoder_hidden_states_mask, b, img_seq)?;

        // ControlNet residual injection interval (epic 3401): `ceil(num_layers / num_residuals)`,
        // matching diffusers `int(np.ceil(len(transformer_blocks) / len(controlnet_block_samples)))`.
        let interval = controlnet_residuals.map(|r| {
            let n = self.blocks.len();
            let k = r.len().max(1);
            n.div_ceil(k)
        });
        for (i, block) in self.blocks.iter().enumerate() {
            let (e, h) = block.forward(
                &hidden,
                &encoder,
                &text_emb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
                modulate_index.as_ref(),
            )?;
            encoder = e;
            // After each base block, add the (scaled) control residual for this block's group —
            // diffusers `hidden_states = hidden_states + controlnet_block_samples[i // interval]`.
            hidden = match (controlnet_residuals, interval) {
                (Some(res), Some(interval)) => {
                    let idx = (i / interval).min(res.len() - 1);
                    let scale = Array::from_slice(&[control_scale], &[1]);
                    add(&h, &multiply(&res[idx], &scale)?)?
                }
                _ => h,
            };
        }

        // norm_out uses only the real-timestep half of the (doubled) temb (the fork's `temb[:B]`).
        let norm_emb = match &modulate_index {
            Some(_) => split(&text_emb, 2, 0)?.swap_remove(0),
            None => text_emb.clone(),
        };
        let hidden = self.norm_out.forward(&hidden, &norm_emb)?;
        self.proj_out.forward(&hidden)
    }
}

/// The per-token timestep selector for `zero_cond_t` (Qwen-Image-Edit-2511): `0` for the noise
/// latent tokens (`latent_h·latent_w`), `1` for every conditioning-image token (`Σ h·w` over the
/// reference grids). Shape `[B, img_seq]`. Matches the fork's `_build_modulate_index` /
/// diffusers `[[0]*prod(img_shapes[0]) + [1]*Σ prod(img_shapes[1:])]`.
fn build_modulate_index(
    b: i32,
    latent_h: usize,
    latent_w: usize,
    cond_grids: &[(usize, usize)],
) -> Array {
    let noise_len = latent_h * latent_w;
    let cond_len: usize = cond_grids.iter().map(|(h, w)| h * w).sum();
    let mut row = vec![0i32; noise_len];
    row.extend(std::iter::repeat_n(1i32, cond_len));
    let mut data = Vec::with_capacity(b as usize * row.len());
    for _ in 0..b {
        data.extend_from_slice(&row);
    }
    Array::from_slice(&data, &[b, (noise_len + cond_len) as i32])
}

/// Additive fill for masked-out attention keys — the fork's large-negative joint-mask value
/// (`QwenTransformer`'s `attention_mask` fill), added to logits before softmax.
const MASK_FILL: f32 = -1e9;

/// Additive joint mask `[B, 1, 1, txt+img]` (text keys masked where padded; image keys always
/// attended). Returns `None` when no text token is padded (the fork's all-ones short-circuit).
///
/// **Both shipped `generate` paths always pass `txt_mask = None`** (see `pipeline.rs`): the prompt
/// embeds carry no padding into the transformer, so parity is proven maskless. The construction
/// below is therefore unreached today; it is kept correct for an external caller that supplies a
/// genuinely padded text mask, not as a parity gap.
fn build_joint_mask(txt_mask: Option<&Array>, b: i32, img_seq: i32) -> Result<Option<Array>> {
    let Some(m) = txt_mask else {
        return Ok(None);
    };
    let mvals = host_i32(m)?;
    if mvals.iter().all(|&v| v == 1) {
        return Ok(None);
    }
    let txt_seq = m.shape()[1];
    let joint = txt_seq + img_seq;
    let mut data = vec![0f32; (b * joint) as usize];
    for bi in 0..b {
        for j in 0..joint {
            let valid = j >= txt_seq || mvals[(bi * txt_seq + j) as usize] == 1;
            if !valid {
                data[(bi * joint + j) as usize] = MASK_FILL;
            }
        }
    }
    Ok(Some(Array::from_slice(&data, &[b, 1, 1, joint])))
}
