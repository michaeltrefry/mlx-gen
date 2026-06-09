//! Kolors scheduler (sc-3094) — a faithful port of diffusers `EulerDiscreteScheduler` with the
//! Kolors config: `scaled_linear` betas (β₀=0.00085, β₁=0.014), **num_train_timesteps=1100**,
//! `timestep_spacing="leading"`, `steps_offset=1`, `interpolation_type="linear"`,
//! `final_sigmas_type="zero"`, epsilon prediction, `s_churn=0`.
//!
//! This is the non-ancestral Euler — **no per-step RNG** — so a denoise run is fully deterministic
//! given the initial latents, which is what makes pixel parity vs `KolorsPipeline` achievable. It
//! differs from the core [`mlx_gen::sampler::LightningSampler`] (which is the same scheduler with
//! `timestep_spacing="trailing"`) only in the timestep selection and `init_noise_sigma`:
//!  - **leading**: `timesteps = (arange(0,N)·(1100//N)).round()[::-1] + steps_offset`;
//!  - leading ⇒ `init_noise_sigma = (max_sigma² + 1)^0.5` (trailing/linspace use `max_sigma`).
//!
//! The latents live in diffusers' σ-scaled space; [`DiffusionSampler::scale_model_input`] divides by
//! `√(σ²+1)` before the U-Net, and the Euler step is `x + eps·(σ_next − σ)`.

use mlx_rs::ops::{divide, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::sampler::AlphaSchedule;
use mlx_gen::{DiffusionSampler, Result};

/// Kolors' EulerDiscrete (leading) sampler over the 1100-step `scaled_linear` schedule.
pub struct KolorsEulerSampler {
    /// Interpolated sigmas at the (effective) timesteps, length `num_steps + 1` (trailing `0.0`).
    /// For img2img this is the schedule **sliced** at the strength-derived start (`begin_index`).
    sigmas: Vec<f32>,
    /// The (float) leading timesteps fed to the U-Net, length `num_steps`.
    timesteps: Vec<f32>,
    init_noise_sigma: f32,
    /// The σ at the img2img start (`sigmas[begin_index]`) — what diffusers' `add_noise` scales the
    /// noise by when seeding the init latents. Equals `sigmas[0]` after the img2img slice; for the
    /// full txt2img schedule it is just the max σ and is unused (txt2img seeds via init noise).
    start_sigma: f32,
    model_dtype: Dtype,
}

impl KolorsEulerSampler {
    /// Kolors defaults: `num_train_timesteps=1100`, β₀=0.00085, β₁=0.014, `steps_offset=1`.
    pub fn kolors(num_steps: usize, model_dtype: Dtype) -> Result<Self> {
        Self::new(1100, 0.00085, 0.014, 1, num_steps, model_dtype)
    }

    /// The img2img variant of [`Self::kolors`]: build the full `num_steps` schedule, then slice it at
    /// the strength-derived start exactly as diffusers' `KolorsImg2ImgPipeline.get_timesteps` +
    /// `set_begin_index` do. With `init_timestep = min(int(num_steps·strength), num_steps)` and
    /// `t_start = num_steps − init_timestep`, the run uses `timesteps[t_start..]` / `sigmas[t_start..]`
    /// and seeds the init latents with [`Self::add_noise`] at `σ = sigmas[t_start]`. `strength ≤
    /// 1/num_steps` ⇒ 0 effective steps ⇒ the (un-noised) init is returned unchanged.
    pub fn kolors_img2img(num_steps: usize, strength: f32, model_dtype: Dtype) -> Result<Self> {
        let full = Self::kolors(num_steps, model_dtype)?;
        let init_timestep = ((num_steps as f32 * strength) as usize).min(num_steps);
        let t_start = num_steps - init_timestep;
        // sigmas has length num_steps+1 (trailing 0); slicing from t_start keeps that trailing 0.
        let sigmas = full.sigmas[t_start..].to_vec();
        let timesteps = full.timesteps[t_start..].to_vec();
        let start_sigma = sigmas[0];
        Ok(Self {
            sigmas,
            timesteps,
            init_noise_sigma: full.init_noise_sigma,
            start_sigma,
            model_dtype,
        })
    }

    /// Seed img2img: noise the clean (VAE-encoded, scaled) init latents at the start σ —
    /// diffusers' `EulerDiscreteScheduler.add_noise` with `begin_index = t_start`, which is the
    /// **raw** `x₀ + noise·σ` (the latents stay in un-normalized σ-space; `scale_model_input`
    /// normalizes before each U-Net call). Draws no RNG — the caller supplies `noise`.
    pub fn add_noise(&self, x0: &Array, noise: &Array) -> Result<Array> {
        use mlx_rs::ops::add;
        let x0 = x0.as_dtype(Dtype::Float32)?;
        let noise = noise.as_dtype(Dtype::Float32)?;
        Ok(add(&x0, &multiply(&noise, scalar(self.start_sigma))?)?)
    }

    pub fn new(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        steps_offset: i64,
        num_steps: usize,
        model_dtype: Dtype,
    ) -> Result<Self> {
        let sched = AlphaSchedule::scaled_linear(num_train_timesteps, beta_start, beta_end)?;
        // Per-train-step Karras sigma √((1-ᾱ)/ᾱ) (the `alphas_cumprod` field is public).
        let full: Vec<f64> = sched
            .alphas_cumprod
            .iter()
            .map(|&acp| {
                let a = acp as f64;
                ((1.0 - a) / a).sqrt()
            })
            .collect();

        // leading: timesteps = (arange(0,N)·step_ratio).round()[::-1] + steps_offset.
        let step_ratio = (num_train_timesteps / num_steps) as i64;
        let timesteps: Vec<f32> = (0..num_steps)
            .rev()
            .map(|j| ((j as i64 * step_ratio) + steps_offset) as f32)
            .collect();

        // np.interp(timesteps, arange(0, N), full), then append 0 (final_sigmas_type="zero").
        let interp = |t: f32| -> f32 {
            let tt = (t as f64).clamp(0.0, (num_train_timesteps - 1) as f64);
            let lo = tt.floor() as usize;
            let hi = (lo + 1).min(num_train_timesteps - 1);
            let frac = tt - lo as f64;
            (full[lo] * (1.0 - frac) + full[hi] * frac) as f32
        };
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp(t)).collect();
        let max_sigma = sigmas.iter().copied().fold(0.0_f32, f32::max);
        sigmas.push(0.0);

        Ok(Self {
            sigmas,
            timesteps,
            // leading spacing ⇒ init_noise_sigma = (max_sigma² + 1)^0.5.
            init_noise_sigma: (max_sigma * max_sigma + 1.0).sqrt(),
            // The full txt2img schedule starts at the max σ; the img2img slice overrides this.
            start_sigma: max_sigma,
            model_dtype,
        })
    }
}

impl DiffusionSampler for KolorsEulerSampler {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        let sigma = self.sigmas[i] as f64;
        let scaled = divide(x, scalar(((sigma * sigma + 1.0).sqrt()) as f32))?;
        Ok(scaled.as_dtype(self.model_dtype)?)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        Ok(multiply(
            &noise.as_dtype(Dtype::Float32)?,
            scalar(self.init_noise_sigma),
        )?)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        use mlx_rs::ops::add;
        // Euler, epsilon prediction, gamma=0: prev = x + eps·(σ_next − σ). (diffusers upcasts to f32.)
        let eps = model_output.as_dtype(Dtype::Float32)?;
        let x = x.as_dtype(Dtype::Float32)?;
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        Ok(add(&x, &multiply(&eps, scalar(dt))?)?)
    }
}
