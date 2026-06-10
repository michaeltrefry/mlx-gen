//! Z-Image sampling-pipeline helpers: seeded latent creation, latent unpacking, and the
//! decoded-tensor → [`Image`] conversion — ports of the fork's `ZImageLatentCreator` +
//! `ImageUtil`. The denoise loop that composes these with the transformer
//! ([`crate::transformer`]), scheduler ([`mlx_gen::FlowMatchEuler`]) and VAE ([`crate::vae`])
//! lands once `load()` assembles the model from weights (+ the text encoder).

use mlx_gen::array::host_i32;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    CancelFlag, Conditioning, Error, FlowMatchEuler, GenerationRequest, Image, Progress, Result,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{random, Array, Dtype};

use crate::control_transformer::ZImageControlTransformer;
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;
use crate::ZImageTransformer;

/// Z-Image latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;

// The decoded-tensor → Image step is identical across families and now lives in core (F-006);
// re-exported so `crate::pipeline::decoded_to_image` and the crate's public surface are unchanged.
pub use mlx_gen::image::decoded_to_image;

/// Seeded txt2img latent noise — shape `[16, 1, height/8, width/8]`, f32. Port of
/// `ZImageLatentCreator.create_noise` (`mx.random.normal` with `key(seed)`). The fork casts to
/// the model precision (bf16) when the latents enter the loop; this returns the raw f32 sample
/// so seeded-RNG parity can be checked directly.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        LATENT_CHANNELS,
        1,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Port of `ZImageLatentCreator.unpack_latents`: `[C, 1, H, W]` → `[1, C, H, W]` (add a batch
/// axis, drop the singleton temporal axis) before VAE decode.
pub fn unpack_latents(latents: &Array) -> Result<Array> {
    Ok(latents.expand_dims(0)?.squeeze_axes(&[2])?)
}

/// `cap_feats = encoder_out[0, :num_valid, :]` — drop the batch axis and the padded tail. The
/// text encoder returns `[1, seq, hidden]` (padded to max length); the DiT consumes only the
/// valid caption tokens. (mlx-rs has no slice op, so this is a range-gather.)
pub fn slice_valid(encoder_out: &Array, num_valid: i32) -> Result<Array> {
    let sh = encoder_out.shape();
    let (s, h) = (sh[1], sh[2]);
    let flat = encoder_out.reshape(&[s, h])?;
    let idx = Array::from_slice(&(0..num_valid).collect::<Vec<i32>>(), &[num_valid]);
    Ok(flat.take_axis(&idx, 0)?)
}

/// Flow-match Euler denoise loop with progress + cooperative cancellation: each step predicts the
/// velocity with the DiT and takes an Euler step, emitting a [`Progress::Step`] and checking
/// `cancel` between steps. `latents` is the seeded init (see [`create_noise`]); `cap_feats` is the
/// text-encoder conditioning. Returns the final latents (pre-VAE).
///
/// `start_step` is the first schedule index to run — `0` for txt2img, `init_time_step` for img2img
/// (the fork's `range(init_time_step, num_steps)`). Progress is reported over the steps actually
/// run (`total = num_steps - start_step`).
///
/// Mirrors the fork's loop: `timestep = 1 - sigma[t]` (the transformer applies its own
/// `t_scale`), `latents += (sigma[t+1] - sigma[t]) * velocity`.
pub fn denoise_with_progress(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
    start_step: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut latents = latents;
    let total = (scheduler.num_steps() - start_step) as u32;
    for t in start_step..scheduler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let velocity = transformer.forward(&latents, scheduler.timestep(t), cap_feats)?;
        latents = scheduler.step(&latents, &velocity, t)?;
        on_progress(Progress::Step {
            current: (t - start_step) as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// [`denoise_with_progress`] from step 0, with no progress callback and no cancellation — the bare
/// loop used by the stage-wise parity tests. Composes the parity-proven transformer + scheduler;
/// full-weights numeric parity is the real-hardware E2E (sc-2352).
pub fn denoise(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
) -> Result<Array> {
    denoise_with_progress(
        transformer,
        scheduler,
        latents,
        cap_feats,
        0,
        &CancelFlag::default(),
        &mut |_| {},
    )
}

/// [`denoise_with_progress`] for the ControlNet variant: each step predicts the velocity with the
/// [`ZImageControlTransformer`], passing the (constant) `control_context` + `control_context_scale`
/// to every forward (the fork's `ZImageControl._control_predict`). Same Euler step, progress, and
/// cooperative cancellation as the base loop. `start_step` is `0` for txt2img+control and
/// `init_time_step` for img2img+control.
#[allow(clippy::too_many_arguments)]
pub fn denoise_control_with_progress(
    transformer: &ZImageControlTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
    control_context: &Array,
    control_context_scale: f32,
    start_step: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut latents = latents;
    let total = (scheduler.num_steps() - start_step) as u32;
    for t in start_step..scheduler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let velocity = transformer.forward(
            &latents,
            scheduler.timestep(t),
            cap_feats,
            Some(control_context),
            control_context_scale,
        )?;
        latents = scheduler.step(&latents, &velocity, t)?;
        on_progress(Progress::Step {
            current: (t - start_step) as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Resolve the img2img start step (the fork's `Config.init_time_step`): for a reference image with
/// `strength` in `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise `0` (pure txt2img).
/// Higher strength → later start → fewer denoise steps → output stays closer to the init image
/// (the fork's convention).
pub fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// img2img init image → packed clean latents `[16, 1, H/8, W/8]` (f32). Port of the fork's
/// `LatentCreator.encode_image` ∘ `ZImageLatentCreator.pack_latents`: PIL-LANCZOS scale to the
/// target dims, normalize `[0,255] → [-1,1]` as NCHW, VAE-encode (mean → latent space), pack.
pub fn encode_init_latents(
    vae: &Vae,
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, target_width, target_height)?;
    let encoded = vae.encode(&image_nchw)?; // [1, 16, H/8, W/8]
    pack_latents(&encoded)
}

/// Build the 33-channel VAE-encoded **control context** from a control image (e.g. a rendered
/// pose skeleton) — the fork's `ZImageControl._encode_control_context`. VAE-encode to 16ch latents
/// (the exact img2img [`encode_init_latents`] path: LANCZOS → `[-1,1]` NCHW → encode → pack
/// `[16,1,H/8,W/8]`), then concat a zero mask (1ch) and a zero inpaint latent (16ch) → `[33, 1,
/// H/8, W/8]`. Pure-pose control has no init image and no mask, so those two channel groups are
/// zeros; the channel layout (control latent | mask | inpaint) matches the Fun-Controlnet-Union
/// `control_all_x_embedder`'s 33ch input.
pub fn encode_control_context(
    vae: &Vae,
    control_image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let control_latents = encode_init_latents(vae, control_image, target_width, target_height)?;
    let sh = control_latents.shape(); // [16, 1, H/8, W/8]
    let (c, fdim, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let mask = Array::from_slice(&vec![0f32; (fdim * h * w) as usize], &[1, fdim, h, w]);
    let inpaint = Array::from_slice(&vec![0f32; (c * fdim * h * w) as usize], &[c, fdim, h, w]);
    Ok(concatenate_axis(&[&control_latents, &mask, &inpaint], 0)?)
}

/// Scale an RGB8 init image to `target` dims with PIL LANCZOS (the fork's `scale_to_dimensions`,
/// a no-op when already sized), normalize `[0,255] → [-1,1]`, and lay out as NCHW `[1, 3, H, W]`
/// f32 — the input the VAE encoder expects.
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "init image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // PIL LANCZOS on the uint8 image (no-op when already at target size), matching the fork.
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    // /255 then [-1,1], as NHWC, then transpose to NCHW (the fork's `to_array` convention).
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let nhwc = Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]);
    Ok(nhwc.transpose_axes(&[0, 3, 1, 2])?)
}

/// Port of `ZImageLatentCreator.pack_latents`: VAE-encoder latent `[1, C, H/8, W/8]` (or a 5-D
/// `[1, C, 1, H/8, W/8]`) → `[C, 1, H/8, W/8]`, matching the seeded-noise layout so the two can be
/// blended.
pub fn pack_latents(encoded: &Array) -> Result<Array> {
    let sh = encoded.shape();
    let e = if sh.len() == 5 {
        encoded.reshape(&[sh[0], sh[1], sh[3], sh[4]])? // drop temporal axis
    } else {
        encoded.clone()
    };
    Ok(e.expand_dims(2)?.squeeze_axes(&[0])?)
}

/// Port of `LatentCreator.add_noise_by_interpolation`: `(1 - sigma) * clean + sigma * noise`. The
/// img2img blend that seeds the denoise loop at `sigma = sigmas[init_time_step]`.
pub fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    Ok(add(&multiply(clean, one_minus)?, &multiply(noise, s)?)?)
}

/// Prompt → `cap_feats` (f32): tokenize with the Qwen chat template, run the text encoder, slice off
/// the padded tail to the valid caption tokens. Shared by the base + control generators and the
/// trainer (F-035); `id` only labels the empty-prompt error. An empty prompt tokenizes to `[1, 0]`,
/// so guard on shape before any host readback (`host_i32` on a size-0 array would panic) —
/// `validate_request` already rejects it, this is defense-in-depth at the encode boundary.
pub(crate) fn encode_prompt(
    tokenizer: &TextTokenizer,
    text_encoder: &TextEncoder,
    prompt: &str,
    id: &str,
) -> Result<Array> {
    let t = tokenizer.tokenize(prompt)?;
    if t.input_ids.shape()[1] == 0 {
        return Err(Error::Msg(format!("{id}: empty prompt")));
    }
    let num_valid: i32 = host_i32(&t.attention_mask)?.iter().sum();
    if num_valid == 0 {
        return Err(Error::Msg(format!("{id}: empty prompt")));
    }
    let enc = text_encoder.forward(&t.input_ids, &t.attention_mask)?;
    slice_valid(&enc, num_valid)
}

/// Resolve the single img2img init image + its strength from the request's conditioning (F-035). The
/// per-reference strength wins over `req.strength`. Z-Image conditions on exactly one init image, so
/// more than one `Reference` is an error (multi-image is `MultiReference`, unadvertised here).
pub(crate) fn resolve_reference<'a>(
    req: &'a GenerationRequest,
    id: &str,
) -> Result<Option<(&'a Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(Error::Msg(format!(
                    "{id}: multiple reference images are not supported (single img2img init only)"
                )));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
}

/// The shared per-image batch render the two Z-Image generators only differed in the denoise call for
/// (F-035): for each of `req.count` images, build the seeded bf16 noise (blended with the pre-encoded
/// `clean` latents for img2img), run `denoise`, then VAE-decode to an RGB8 [`Image`]. The base vs
/// control branch is the `denoise` closure; the seed convention, the `sigma`-blend, the process-global
/// compile-glue enable, and the decode tail are identical and live here. Bit-identical to the prior
/// inline loops.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_batch(
    vae: &Vae,
    scheduler: &FlowMatchEuler,
    clean: Option<&Array>,
    start_step: usize,
    base_seed: u64,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
    mut denoise: impl FnMut(Array, &mut dyn FnMut(Progress)) -> Result<Array>,
) -> Result<Vec<Image>> {
    let mut images = Vec::with_capacity(req.count as usize);
    for i in 0..req.count {
        // Distinct seed per image in a batch (the fork's `seed + i`). PARITY-BF16 (sc-2609): the noise
        // is bf16 to match the fork's seed→image mapping (f32 is a *different*, higher-precision
        // realization, not just sharper).
        let seed = base_seed.wrapping_add(i as u64);
        let noise = create_noise(seed, req.width, req.height)?.as_dtype(Dtype::Bfloat16)?;
        let latents = match clean {
            // img2img: blend the pre-encoded clean latents with the noise at `sigma = sigmas[start]`.
            Some(clean) => add_noise_by_interpolation(clean, &noise, scheduler.sigmas[start_step])?,
            None => noise,
        };
        // sc-2963 (rollout of sc-2957): run the DiT's fusable elementwise glue through `mx.compile` —
        // bit-exact and a per-step win. Enabled at the production boundary; process-global, idempotent.
        crate::set_compile_glue(true);
        let latents = denoise(latents, on_progress)?;

        on_progress(Progress::Decoding);
        // [16,1,H,W] -> [1,16,H,W] -> [1,16,1,H,W] for VAE decode.
        let unpacked = unpack_latents(&latents)?;
        let sh = unpacked.shape();
        let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]])?;
        let decoded = vae.decode(&latent5)?.as_dtype(Dtype::Float32)?;
        images.push(decoded_to_image(&decoded)?);
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn resolve_reference_handles_count_and_strength() {
        // F-035: the single shared img2img resolution (was duplicated in both generators). No
        // Reference → None.
        let none = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert_eq!(resolve_reference(&none, "z_image_turbo").unwrap(), None);

        // A per-reference strength wins over req.strength.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: img(64, 64),
                strength: Some(0.3),
            }],
            ..Default::default()
        };
        let (_, s) = resolve_reference(&req, "z_image_turbo").unwrap().unwrap();
        assert_eq!(s, Some(0.3));

        // Missing per-reference strength falls back to req.strength.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: img(64, 64),
                strength: None,
            }],
            ..Default::default()
        };
        let (_, s) = resolve_reference(&req, "z_image_turbo").unwrap().unwrap();
        assert_eq!(s, Some(0.6));

        // More than one Reference is an error (single img2img init only).
        let req = GenerationRequest {
            prompt: "a fox".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: img(64, 64),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(64, 64),
                    strength: None,
                },
            ],
            ..Default::default()
        };
        let err = resolve_reference(&req, "z_image_turbo")
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple reference images"), "{err}");
    }
}
