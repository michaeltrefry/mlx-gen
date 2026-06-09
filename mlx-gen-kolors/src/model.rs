//! Kolors T2I pipeline (sc-3094) — composes the ChatGLM3 conditioning, the leading-Euler scheduler,
//! the SDXL U-Net (with the ChatGLM context projection), real CFG, and the SDXL VAE decode.
//!
//! Mirrors diffusers `KolorsPipeline`: tokenize → ChatGLM3 `encode_prompt` (context = `hidden[-2]`,
//! pooled = `hidden[-1]` last token, with the left-padded `position_ids`) for the positive AND
//! negative prompt → CFG-batched U-Net denoise over `EulerDiscreteScheduler(leading)` → VAE decode
//! (latents / 0.13025). `time_ids` = `(H, W, 0, 0, H, W)` (the SDXL `_get_add_time_ids`).
//!
//! The whole pipeline is dtype-parametric; the parity gate (`tests/t2i_parity.rs`) runs f32.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, DiffusionSampler, Image, Result};

use mlx_gen_sdxl::{
    decode_image, denoise, denoise_control, denoise_ip, encode_init_latents,
    load_unet_kolors_dtype, load_vae, preprocess_control_image, Autoencoder, ControlContext,
    ControlNet, Denoiser, IpImageEncoder, UNet2DConditionModel,
};

use crate::chatglm3::{ChatGlmConfig, ChatGlmModel};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;

/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: i32 = 8;

/// diffusers `KolorsImg2ImgPipeline` default `strength` (how much of the schedule to re-noise/denoise).
pub const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.3;

/// A loaded Kolors model: ChatGLM3 text encoder + tokenizer + SDXL-family U-Net (with the ChatGLM
/// context projection) + SDXL VAE.
pub struct Kolors {
    chatglm: ChatGlmModel,
    tokenizer: KolorsTokenizer,
    unet: UNet2DConditionModel,
    vae: Autoencoder,
    dtype: Dtype,
}

/// The SDXL-style micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row (the diffusers
/// `_get_add_time_ids` for `original_size == target_size`, no crop).
fn kolors_time_ids(batch: i32, height: i32, width: i32) -> Array {
    let (h, w) = (height as f32, width as f32);
    let row = [h, w, 0.0, 0.0, h, w];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

impl Kolors {
    /// Load every Kolors component from the `Kwai-Kolors/Kolors-diffusers` snapshot at `dtype`.
    /// `tokenizer/tokenizer.json` must already be materialized (`tools/build_kolors_tokenizer.py`).
    pub fn load(snapshot: &std::path::Path, dtype: Dtype) -> Result<Self> {
        let te_w = Weights::from_dir(snapshot.join("text_encoder"))?;
        let chatglm = ChatGlmModel::from_weights(&te_w, ChatGlmConfig::chatglm3_6b(), None, dtype)?;
        let tokenizer = KolorsTokenizer::from_dir(snapshot.join("tokenizer"))?;
        let unet = load_unet_kolors_dtype(snapshot, dtype)?;
        let vae = load_vae(snapshot)?; // SDXL VAE (sdxl-vae-fp16-fix), f32
        Ok(Self {
            chatglm,
            tokenizer,
            unet,
            vae,
            dtype,
        })
    }

    /// Load every Kolors component, then **load-time quantize** the memory drivers to `bits` (4 or 8)
    /// — the mlx-gen-sdxl sc-2641 path: the dense fp16 snapshot is loaded and packed in-memory (there
    /// is no pre-quantized Kolors snapshot). Quantizes the 6B ChatGLM3 encoder (the dominant footprint)
    /// **and** the SDXL-family U-Net (reusing its own `quantize`); the VAE stays f32 (it overflows in
    /// low precision — the SDXL-family convention). `bits` ∈ {4, 8}.
    pub fn load_quantized(snapshot: &std::path::Path, dtype: Dtype, bits: i32) -> Result<Self> {
        let mut m = Self::load(snapshot, dtype)?;
        m.chatglm.quantize(bits)?;
        m.unet.quantize(bits)?;
        Ok(m)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])`, threading the tokenizer's
    /// left-padded `position_ids` into the ChatGLM3 RoPE (as `KolorsPipeline.encode_prompt` does).
    pub fn encode(&self, prompt: &str) -> Result<(Array, Array)> {
        // Kolors tokenizes the raw prompt (no chat template).
        let t = self.tokenizer.encode(prompt)?;
        self.chatglm
            .encode_prompt(&t.input_ids, &t.attention_mask, Some(&t.position_ids))
    }

    /// Decode latents `[1, h, w, 4]` → an RGB [`Image`] (`vae.decode(latents / 0.13025)`).
    pub fn decode(&self, latents: &Array) -> Result<Image> {
        decode_image(&self.vae, latents)
    }

    /// Run the CFG denoise loop from a (raw, unit-normal) initial-noise tensor `init_noise`
    /// `[1, h, w, 4]` — split out so the parity gate can feed diffusers' exact noise. `pos`/`neg` are
    /// the `(context, pooled)` from [`encode`](Self::encode). Returns the final latents `[1, h, w, 4]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_latents(
        &self,
        init_noise: &Array,
        pos: &(Array, Array),
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        height: i32,
        width: i32,
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        // CFG batch order is [positive, negative] — `mlx_gen_sdxl::denoise` reads row 0 as the text
        // (cond) and row 1 as the uncond.
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        let latents = sampler.scale_initial_noise(init_noise)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        let cancel = CancelFlag::new();
        denoise(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            &cancel,
            &mut |_p| {},
        )
    }

    /// Full T2I: seed the RNG, draw the initial noise, encode the prompt + negative prompt, denoise,
    /// and VAE-decode. `height`/`width` are pixels (multiples of 8). `cfg` ≤ 1 disables guidance.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        let latents =
            self.denoise_latents(&init_noise, &pos, &neg, num_steps, cfg, height, width)?;
        self.decode(&latents)
    }

    /// Run the img2img CFG denoise loop from pre-encoded init latents + a supplied noise tensor —
    /// split out (like [`denoise_latents`](Self::denoise_latents)) so the parity gate can feed
    /// diffusers' exact VAE-encoded init + noise. `init_latents` is the scaled VAE mean
    /// `[1, h, w, 4]`; the sampler is the strength-sliced schedule, the init is seeded via
    /// [`KolorsEulerSampler::add_noise`] (raw `x₀ + noise·σ_start`, no `scale_initial_noise`), and the
    /// loop runs the remaining `int(num_steps·strength)` steps. Returns the final latents.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_img2img_latents(
        &self,
        init_latents: &Array,
        noise: &Array,
        pos: &(Array, Array),
        neg: &(Array, Array),
        num_steps: usize,
        strength: f32,
        cfg: f32,
        height: i32,
        width: i32,
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors_img2img(num_steps, strength, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        // Seed the init: raw `x₀ + noise·σ_start` (diffusers EulerDiscrete add_noise at begin_index).
        let latents = sampler.add_noise(init_latents, noise)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        let cancel = CancelFlag::new();
        denoise(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            &cancel,
            &mut |_p| {},
        )
    }

    /// Full img2img: VAE-encode `image` (resized to `height`×`width`) → seed at the strength-derived
    /// start → encode the prompts → denoise the remaining steps → VAE-decode. Mirrors diffusers
    /// `KolorsImg2ImgPipeline` (using the VAE encoder **mean** as the init, consistent with the rest
    /// of mlx-gen-sdxl's img2img — the production fork convention; the diffusers default samples the
    /// latent dist, which is not reproducible cross-backend). `cfg` ≤ 1 disables guidance.
    #[allow(clippy::too_many_arguments)]
    pub fn img2img(
        &self,
        image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        strength: f32,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        // VAE-encode the init (no RNG: mean, not a sample) so the first global-RNG draw is the
        // add_noise noise — matching the reference's `prepare_latents` order.
        let init_latents = encode_init_latents(&self.vae, image, width as u32, height as u32)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        let latents = self.denoise_img2img_latents(
            &init_latents,
            &noise,
            &pos,
            &neg,
            num_steps,
            strength,
            cfg,
            height,
            width,
        )?;
        self.decode(&latents)
    }

    /// Run the CFG denoise loop with a Kolors **ControlNet** branch injecting residuals each step
    /// (sc-3097) — split out (like [`denoise_latents`](Self::denoise_latents)) so the parity gate can
    /// feed diffusers' exact noise. The `controlnet` is loaded via `mlx_gen_sdxl::load_controlnet`
    /// (the Kolors ControlNet is a standard SDXL `ControlNetModel` whose only deltas — its own
    /// `encoder_hid_proj` 4096→2048 + the 5632 add-embedding — are auto-detected/shape-driven). It is
    /// conditioned with the **same ChatGLM3 context** as the U-Net (the branch projects it with its
    /// own `encoder_hid_proj`). `control_scale = 0` ⇒ the residuals vanish ⇒ identical to plain T2I.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_controlnet_latents(
        &self,
        controlnet: &ControlNet,
        init_noise: &Array,
        control_image: &Image,
        pos: &(Array, Array),
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        control_scale: f32,
        height: i32,
        width: i32,
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        let latents = sampler.scale_initial_noise(init_noise)?;

        // The ControlNet sees the same CFG-batched input as the U-Net (cfg>1 ⇒ [cond, uncond]).
        let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
        let cimg = if cfg > 1.0 {
            concatenate_axis(&[&cimg, &cimg], 0)?
        } else {
            cimg
        };
        let cc = ControlContext {
            controlnet,
            control_image: cimg,
            scale: control_scale,
        };

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        let cancel = CancelFlag::new();
        denoise_control(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            &cancel,
            &mut |_p| {},
            &cc,
        )
    }

    /// Full ControlNet T2I: seed the noise, encode the prompts, denoise with the `controlnet` branch
    /// injecting `control_image`-conditioned residuals (`control_scale`), and VAE-decode. The
    /// `control_image` is preprocessed (LANCZOS resize → `[0,1]` NHWC) by the SDXL primitive.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_controlnet(
        &self,
        controlnet: &ControlNet,
        control_image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        control_scale: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        let latents = self.denoise_controlnet_latents(
            controlnet,
            &init_noise,
            control_image,
            &pos,
            &neg,
            num_steps,
            cfg,
            control_scale,
            height,
            width,
        )?;
        self.decode(&latents)
    }

    /// Install the IP-Adapter decoupled cross-attention K/V pairs (from
    /// [`crate::ip_adapter::load_kolors_ip_adapter`]) into the U-Net's cross-attention layers
    /// (sc-3098). One-time setup; non-destructive to plain T2I (the [`denoise`] path never reads the
    /// IP projections — only [`denoise_ip`] does). 70 pairs for the SDXL-family U-Net.
    pub fn install_ip_adapter(&mut self, pairs: Vec<(Array, Array)>) -> Result<()> {
        self.unet.install_ip_adapter(pairs)
    }

    /// Run the CFG denoise loop with IP-Adapter image tokens injected into every cross-attention at
    /// `ip_scale` (sc-3098) — split out (like [`denoise_latents`](Self::denoise_latents)) for the
    /// parity gate. `ip_tokens` is `[1, N, 2048]` (from [`IpImageEncoder::tokens`]); it is CFG-batched
    /// here with a zeros uncond row. The IP-Adapter pairs must already be installed
    /// ([`install_ip_adapter`](Self::install_ip_adapter)). `ip_scale = 0` ⇒ identical to plain T2I.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_ip_latents(
        &self,
        ip_tokens: &Array,
        init_noise: &Array,
        pos: &(Array, Array),
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        ip_scale: f32,
        height: i32,
        width: i32,
    ) -> Result<Array> {
        use mlx_rs::ops::{concatenate_axis, zeros};
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        let latents = sampler.scale_initial_noise(init_noise)?;

        // CFG batch: [image tokens, zeros] — the uncond row gets no image conditioning.
        let sh = ip_tokens.shape();
        let zero = zeros::<f32>(sh)?.as_dtype(ip_tokens.dtype())?;
        let tokens = concatenate_axis(&[ip_tokens, &zero], 0)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        let cancel = CancelFlag::new();
        denoise_ip(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            &cancel,
            &mut |_p| {},
            &tokens,
            ip_scale,
        )
    }

    /// Full IP-Adapter T2I: encode the `reference_image` → image tokens, seed the noise, encode the
    /// prompts, denoise with the IP tokens injected at `ip_scale`, and VAE-decode. The IP-Adapter
    /// pairs must already be installed via [`install_ip_adapter`](Self::install_ip_adapter).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_ip(
        &self,
        ip_encoder: &IpImageEncoder,
        reference_image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        ip_scale: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        let ip_tokens = ip_encoder.tokens(reference_image)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        let latents = self.denoise_ip_latents(
            &ip_tokens,
            &init_noise,
            &pos,
            &neg,
            num_steps,
            cfg,
            ip_scale,
            height,
            width,
        )?;
        self.decode(&latents)
    }
}
