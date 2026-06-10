//! SDXL T2I sampling pipeline — composes the dual-CLIP conditioning, the seeded prior, the
//! Euler-Ancestral denoise loop with real classifier-free guidance, and the VAE decode. Port of the
//! vendored `StableDiffusionXL.generate_latents` + `_denoising_loop` + `decode`.
//!
//! The U-Net, text encoders, sampler, and CFG run **fp16**, matching the production reference
//! (`StableDiffusionXL(float16=True)`); the VAE runs f32 (it promotes the f16 latents on decode).
//! The RNG is seeded once per image, then the sampler draws the prior + per-step ancestral noise from
//! the global stream — reproducing the reference's exact noise sequence for a seed.

use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, round, subtract};
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{CancelFlag, DiffusionSampler, Error, Image, Progress, Result};

use crate::inpaint::InpaintBlend;
use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::unet::{ControlNet, ControlResiduals, UNet2DConditionModel};
use crate::vae::Autoencoder;

/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;
/// Latent channel count.
pub const LATENT_CHANNELS: i32 = 4;

/// The SDXL micro-conditioning `time_ids`, hardcoded `[512, 512, 0, 0, 512, 512]` per row — the
/// vendored `StableDiffusionXL.generate_latents` quirk (it does NOT pass the real
/// original/target sizes). Reproduced verbatim for parity. `batch` rows.
pub fn text_time_ids(batch: i32) -> Array {
    let row = [512.0f32, 512.0, 0.0, 0.0, 512.0, 512.0];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

/// Run both CLIP encoders over the (CFG) token batch and assemble the SDXL conditioning:
/// `concat(te1.hidden[-2], te2.hidden[-2])` and `te2.pooled`. `tokens` is `[B, N]` (B=2 with CFG).
pub fn encode_conditioning(
    te1: &ClipTextEncoder,
    te2: &ClipTextEncoder,
    tokens: &Array,
) -> Result<(Array, Array)> {
    let o1 = te1.forward(tokens)?;
    let o2 = te2.forward(tokens)?;
    let h1 = &o1.hidden_states[o1.hidden_states.len() - 2];
    let h2 = &o2.hidden_states[o2.hidden_states.len() - 2];
    let conditioning = concatenate_axis(&[h1, h2], -1)?;
    Ok((conditioning, o2.pooled))
}

/// Components needed for one denoise run (borrowed from the loaded model). `sampler` is any
/// [`DiffusionSampler`] — SDXL's production ancestral [`crate::sampler::AncestralEuler`] or a
/// few-step acceleration sampler (`mlx_gen::{LcmSampler, LightningSampler, TcdSampler}`, sc-2769).
pub struct Denoiser<'a> {
    pub unet: &'a UNet2DConditionModel,
    pub sampler: &'a dyn DiffusionSampler,
}

/// ControlNet conditioning for the denoise loop (sc-3058): the loaded branch, the preprocessed
/// control image (NHWC `[B, H, W, 3]` in `[0,1]`, already CFG-batched to match the UNet input), and
/// the `conditioning_scale`. Each step runs the branch on the model input and injects its residuals.
pub struct ControlContext<'a> {
    pub controlnet: &'a ControlNet,
    /// The precomputed conditioning embedding for the fixed control image
    /// ([`ControlNet::embed_cond`]) — step-invariant, so it is computed once at construction rather
    /// than re-run every denoise step (F-069).
    pub cond_embed: Array,
    pub scale: f32,
}

/// Run the denoise loop with CFG, driven entirely by the sampler's own schedule
/// (`sampler.num_steps()` iterations). `latents` is the seeded init `[1, h, w, 4]`;
/// `conditioning`/`pooled`/`time_ids` carry the CFG batch (B = 2 when `cfg > 1`). Returns the final
/// latents; progress per step; `cancel` between steps. Each iteration:
/// `x_in = scale_model_input(latents)` → U-Net eps → (CFG) → `latents = sampler.step(eps, latents)`.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        None,
        &[],
        None,
        None,
    )
}

/// Like [`denoise`] but applies the legacy inpaint **mask-blend** after each step (sc-3057):
/// `latents = (1-mask)·init_noised + mask·latents`. The blend draws no RNG, so the ancestral noise
/// stream is identical to plain img2img (a full-white mask ⇒ bit-identical to [`denoise`]).
#[allow(clippy::too_many_arguments)]
pub fn denoise_inpaint(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    blend: &InpaintBlend,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        Some(blend),
        &[],
        None,
        None,
    )
}

/// Like [`denoise`] but runs a ControlNet branch each step and injects its residuals into the UNet
/// (sc-3058). Works on the txt2img or img2img init (set up by the caller); `scale = 0` ⇒ identical
/// to [`denoise`] (the residuals vanish).
#[allow(clippy::too_many_arguments)]
pub fn denoise_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    control: &ControlContext,
) -> Result<Array> {
    denoise_multi_control(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        std::slice::from_ref(control),
    )
}

/// Like [`denoise_control`] but runs **multiple** ControlNet branches and sums their residuals — the
/// diffusers `MultiControlNetModel` rule (sc-3378). `controls[i]` pairs with the `i`-th branch; all
/// share the text `conditioning` as their cross-attention input. A single-element `controls` is
/// bit-identical to [`denoise_control`]; an empty `controls` reduces to [`denoise`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_multi_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        None,
        controls,
        None,
        None,
    )
}

/// Like [`denoise`] but injects the IP-Adapter image `tokens` (`[B, N, cross_attention_dim]`,
/// CFG-batched with a zeros uncond row) into every cross-attention at `scale` (sc-3059). Works on
/// the txt2img or img2img init; `scale = 0` ⇒ identical to [`denoise`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        None,
        &[],
        Some((tokens, scale)),
        None,
    )
}

/// Like [`denoise`] but runs the **InstantID** dual conditioning each step (sc-3113/3114): the
/// IdentityNet ControlNet (on the kps `control` image, cross-attended to `controlnet_encoder` = the
/// face tokens) injects its residuals, while the face IP `tokens` are injected into the UNet
/// cross-attention at `scale`. `tokens`/`controlnet_encoder` are typically the same CFG-batched
/// `[B, 16, cross_attention_dim]` face tokens. `scale = 0` + a `0`-scale control ⇒ identical to
/// [`denoise`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    control: &ControlContext,
    controlnet_encoder: &Array,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_ip_multi_control(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        std::slice::from_ref(control),
        controlnet_encoder,
        tokens,
        scale,
    )
}

/// Like [`denoise_ip_control`] but runs **multiple** ControlNet branches and sums their residuals
/// before injection — the diffusers `MultiControlNetModel` rule (sc-3378). This is the engine for
/// InstantID pose mode (sc-3117): `controls = [IdentityNet(kps), OpenPose(skeleton)]`, each with its
/// own `conditioning_scale`, all sharing `controlnet_encoder` (the face tokens) as their
/// cross-attention conditioning — exactly as the vendored InstantID pipeline passes the same
/// `prompt_image_emb` to every sub-ControlNet. A single-element `controls` is bit-identical to
/// [`denoise_ip_control`]; an empty `controls` reduces to [`denoise_ip`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_multi_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
    controlnet_encoder: &Array,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        None,
        controls,
        Some((tokens, scale)),
        Some(controlnet_encoder),
    )
}

#[allow(clippy::too_many_arguments)]
fn denoise_core(
    d: &Denoiser,
    mut latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    inpaint: Option<&InpaintBlend>,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
) -> Result<Array> {
    let steps = d.sampler.num_steps();
    // A zero-step denoise (img2img at strength ≤ 1/steps) is a no-op: return the init latents
    // unchanged, matching the reference's `int(num_steps · strength)` loop count. Guards the
    // degenerate schedule and the σ=0 ancestral step that would otherwise NaN.
    if steps == 0 {
        return Ok(latents);
    }
    // sc-2963 (rollout of sc-2957): fuse the UNet's SiLU activations via `mx.compile` — bit-exact in
    // fp16 (`max|Δ|=0`, compile_parity.rs), so it does not move the precision-load-bearing fp16
    // golden. The GELU/GEGLU activations are already compiled (sc-2721). Process-global, idempotent.
    crate::set_compile_glue(true);
    let cfg_on = cfg > 1.0;
    let total = steps as u32;
    for i in 0..steps {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        // Scale the latents into the model's input space: identity for the ancestral sampler (which
        // folds the renormalization into its step → bit-identical to the pre-trait loop), `x/√(σ²+1)`
        // for the Lightning Euler sampler. Acceleration samplers also cast to the U-Net compute dtype.
        let x_in = d.sampler.scale_model_input(&latents, i)?;
        let x_unet = if cfg_on {
            concatenate_axis(&[&x_in, &x_in], 0)?
        } else {
            x_in
        };
        let timestep = d.sampler.timestep(i);
        // ControlNet cross-attn conditioning: `conditioning` (text) for tile-CN; the caller may
        // override it (InstantID feeds the face tokens as the IdentityNet's encoder_hidden_states).
        // The override is shared across branches — matching the InstantID MultiControlNet path,
        // where the vendored pipeline passes the same `prompt_image_emb` to every sub-ControlNet.
        let cn_enc = control_encoder.unwrap_or(conditioning);
        // MultiControlNet (sc-3378): run each branch and sum its (already conditioning_scale'd)
        // residuals — the diffusers `MultiControlNetModel` rule. One branch ⇒ the single
        // residual unchanged (bit-exact regression vs the pre-slice path); zero ⇒ `None`.
        let combined: Option<ControlResiduals> = {
            let mut acc: Option<ControlResiduals> = None;
            for cc in controls {
                let res = cc.controlnet.forward(
                    &x_unet,
                    &cc.cond_embed,
                    timestep,
                    cn_enc,
                    pooled,
                    time_ids,
                    cc.scale,
                )?;
                acc = Some(match acc {
                    None => res,
                    Some(prev) => prev.add(&res)?,
                });
            }
            acc
        };
        let eps = match (ip, combined.as_ref()) {
            (Some((tokens, scale)), Some(res)) => {
                // InstantID (sc-3113/3114/3117): the (possibly multi-branch summed) ControlNet
                // residuals AND the face IP tokens injected into the UNet cross-attention.
                d.unet.forward_with_ip_control(
                    &x_unet,
                    timestep,
                    conditioning,
                    pooled,
                    time_ids,
                    (tokens, scale),
                    res,
                )?
            }
            (Some((tokens, scale)), None) => {
                // IP-Adapter (sc-3059): inject the image tokens into every cross-attention.
                d.unet.forward_with_ip(
                    &x_unet,
                    timestep,
                    conditioning,
                    pooled,
                    time_ids,
                    (tokens, scale),
                )?
            }
            (None, Some(res)) => {
                // ControlNet (sc-3058 / MultiControlNet sc-3378): inject the (summed) residuals.
                d.unet.forward_with_control(
                    &x_unet,
                    timestep,
                    conditioning,
                    pooled,
                    time_ids,
                    res,
                )?
            }
            (None, None) => d
                .unet
                .forward(&x_unet, timestep, conditioning, pooled, time_ids)?,
        };
        let eps = if cfg_on {
            let row = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0);
            let eps_text = row(0)?;
            let eps_neg = row(1)?;
            // `eps_neg + cfg·(eps_text − eps_neg)`. The reference's `cfg_weight` is a python float that
            // weak-casts to the eps dtype, so CFG runs in the compute dtype — cast the scalar to the
            // eps dtype here too. An f32 `cfg` would promote an fp16 eps to f32, and the sampler step
            // (which keys off `eps.dtype()`) would then run f32, silently leaving the latents f32.
            let cfg_s = scalar(cfg).as_dtype(eps_text.dtype())?;
            add(
                &eps_neg,
                &multiply(&subtract(&eps_text, &eps_neg)?, &cfg_s)?,
            )?
        } else {
            eps
        };
        latents = d.sampler.step(&eps, &latents, i)?;
        // Legacy inpaint blend (sc-3057): pin the kept region to the init noised to this step's σ,
        // keep the repaint region freely denoised. No RNG draw → ancestral stream unperturbed.
        if let Some(b) = inpaint {
            latents = b.blend(&latents, i)?;
        }
        // Force evaluation each step (the reference's per-step `mx.eval`). Beyond bounding the lazy
        // graph, this materializes the global-RNG state split between steps so the ancestral noise
        // stream is byte-identical to the reference — leaving it lazy across all steps perturbs the
        // draws and re-introduces the chaotic divergence (sc-2400 S5).
        latents.eval()?;
        on_progress(Progress::Step {
            current: i as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Seed the global RNG and sample the prior latents `[1, height/8, width/8, 4]` (NHWC, f32).
pub fn seeded_prior(sampler: &EulerSampler, seed: u64, width: u32, height: u32) -> Result<Array> {
    random::seed(seed)?;
    sampler.sample_prior(&[
        1,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
        LATENT_CHANNELS,
    ])
}

/// Preprocess an init image for img2img: PIL-LANCZOS resize to the target dims (no-op when already
/// sized), normalize `[0,255] → [-1,1]`, lay out NHWC `[1, H, W, 3]` f32 — the input the VAE encoder
/// expects. Uses the core PIL-exact resampler (`resize_lanczos_u8`).
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "sdxl init image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// Preprocess a ControlNet control image (sc-3058): LANCZOS resize to the target dims, normalize
/// `[0,255] → [0,1]` (the diffusers control image processor uses `do_normalize=False` ⇒ `[0,1]`, NOT
/// the `[-1,1]` of a VAE init), lay out NHWC `[1, H, W, 3]` f32.
pub fn preprocess_control_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "sdxl control image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    let norm: Vec<f32> = resized.iter().map(|&v| v / 255.0).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// img2img init latents: preprocess the image → VAE-encode mean `[1, h, w, 4]` (NHWC). The fork's
/// `generate_latents_from_image` uses the encoder mean (not a sample) as `x_0`.
pub fn encode_init_latents(
    vae: &Autoencoder,
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let nhwc = preprocess_init_image(image, target_width, target_height)?;
    vae.encode_mean(&nhwc)
}

/// Convert a VAE-decoded NHWC tensor `[1, H, W, 3]` (≈`[-1, 1]`) to an RGB8 [`Image`]:
/// `clip(x·0.5 + 0.5, 0, 1) · 255` (the vendored `StableDiffusion.decode` + txt2image recipe).
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let half = scalar(0.5);
    let x = add(&multiply(decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;
    let sh = x.shape();
    let (h, w, c) = (sh[1] as u32, sh[2] as u32, sh[3] as u32);
    let n = (h * w * c) as usize;
    let total: i32 = sh.iter().product();
    let flat = x.reshape(&[total])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w,
        height: h,
        pixels,
    })
}

/// Decode final latents `[1, h, w, 4]` to an RGB8 image.
pub fn decode_image(vae: &Autoencoder, latents: &Array) -> Result<Image> {
    let decoded = vae.decode(latents)?;
    decoded_to_image(&decoded)
}
