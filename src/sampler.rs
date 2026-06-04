//! Swappable diffusion samplers — the engine-agnostic seam behind the few-step acceleration
//! variants (LCM / SDXL-Lightning / Hyper-SD), sc-2769.
//!
//! A [`DiffusionSampler`] owns a model's **denoise schedule**: the per-step conditioning timestep,
//! the model-input scaling, the initial-noise scaling, and the per-step update. The generic denoise
//! loop drives `&dyn DiffusionSampler` so a model can swap samplers per request without the loop
//! knowing which one is running. Each model family supplies its own impls:
//! - SDXL's production default is the crate-local ancestral Euler sampler (`mlx-gen-sdxl`), which
//!   folds the input scaling into its step → [`DiffusionSampler::scale_model_input`] is identity.
//! - The acceleration samplers here are faithful ports of the **diffusers** schedulers each method
//!   is trained against (`LCMScheduler`, `EulerDiscreteScheduler(timestep_spacing="trailing")`,
//!   `TCDScheduler`). They live in the DDPM `alphas_cumprod` world (NOT the flow-match world of
//!   [`crate::scheduler::FlowMatchEuler`]), so they share an [`AlphaSchedule`] built from the model's
//!   `scaled_linear` betas.
//!
//! The trait is intentionally minimal and family-neutral so FLUX-MLX / Qwen-MLX acceleration (their
//! own flow-match few-step schedules) can implement it later (sc-2908 / sc-2909).

use mlx_rs::ops::{add, divide, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use crate::array::scalar;
use crate::Result;

/// A swappable denoise schedule. The generic loop calls, per step `i`:
/// `x_in = scale_model_input(latents, i)` → `eps = model(x_in, timestep(i))` → (CFG) →
/// `latents = step(eps, latents, i)`. The starting latents are `scale_initial_noise(unit_noise)`.
pub trait DiffusionSampler {
    /// Number of denoise iterations (loop count).
    fn num_steps(&self) -> usize;

    /// The conditioning timestep fed to the model at step `i` (the value the U-Net embeds).
    fn timestep(&self, i: usize) -> f32;

    /// Scale the latents into the model's expected input space at step `i`. The default is identity
    /// (samplers that fold the scaling into [`Self::step`], e.g. the ancestral Euler sampler);
    /// diffusers' Euler divides by `√(σ²+1)`.
    fn scale_model_input(&self, x: &Array, _i: usize) -> Result<Array> {
        Ok(x.clone())
    }

    /// Scale unit-normal noise into the sampler's starting latent space (the txt2img prior).
    fn scale_initial_noise(&self, noise: &Array) -> Result<Array>;

    /// One denoise step: latents at step `i` → latents at step `i+1`, given the (already
    /// CFG-combined) model output. `x` is the **un-scaled** latents (NOT the
    /// [`Self::scale_model_input`] output), matching diffusers' `step(model_output, t, sample)`.
    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array>;
}

/// A discrete DDPM noise schedule: the `alphas_cumprod` table built from `scaled_linear` betas,
/// shared by the diffusers-derived acceleration samplers. Mirrors diffusers'
/// `betas = linspace(√β₀, √β₁, N)²; alphas_cumprod = cumprod(1-betas)` (torch float32).
#[derive(Clone)]
pub struct AlphaSchedule {
    /// `alphas_cumprod[t]`, length `num_train_timesteps`.
    pub alphas_cumprod: Vec<f32>,
}

impl AlphaSchedule {
    /// Build from `scaled_linear` betas (SDXL: `β₀=0.00085`, `β₁=0.012`, `N=1000`). The cumprod runs
    /// through MLX (`Array::cumprod`) so the table matches the reference's torch `cumprod` to f32,
    /// rather than drifting over a 1000-step host accumulation.
    pub fn scaled_linear(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Result<Self> {
        let n = num_train_timesteps as i32;
        // betas = linspace(√β₀, √β₁, N)²  (the √ endpoints taken in f64 like diffusers' Python).
        let (a, b) = ((beta_start as f64).sqrt(), (beta_end as f64).sqrt());
        let lin: Vec<f32> = (0..n)
            .map(|i| {
                let t = if n == 1 {
                    0.0
                } else {
                    i as f32 / (n - 1) as f32
                };
                let v = a + (b - a) * t as f64;
                (v * v) as f32
            })
            .collect();
        let betas = Array::from_slice(&lin, &[n]);
        let alphas = subtract(scalar(1.0), &betas)?;
        let acp = alphas.cumprod(0, false, true)?;
        Ok(Self {
            alphas_cumprod: acp.as_slice::<f32>().to_vec(),
        })
    }

    fn acp(&self, t: usize) -> f64 {
        self.alphas_cumprod[t] as f64
    }

    /// The per-train-step Karras-style sigma `√((1-ᾱ)/ᾱ)` at integer index `t` (diffusers Euler).
    fn sigma_at(&self, t: usize) -> f64 {
        let acp = self.acp(t);
        ((1.0 - acp) / acp).sqrt()
    }
}

/// Select the LCM/TCD inference timesteps (the shared diffusers logic): take `original_steps`
/// linearly-spaced training timesteps `arange(1, original_steps+1)·k − 1` (`k = N/original_steps`),
/// reverse, then pick `num_steps` evenly-spaced-by-index entries. For SDXL `N=1000`,
/// `original_steps=50` → e.g. 4 steps = `[999, 759, 499, 259]`.
fn lcm_style_timesteps(
    num_train_timesteps: usize,
    original_steps: usize,
    num_steps: usize,
) -> Vec<usize> {
    let k = num_train_timesteps / original_steps;
    // lcm_origin_timesteps = arange(1, original_steps+1)·k − 1, then reversed (descending).
    let origin: Vec<i64> = (1..=original_steps as i64)
        .map(|i| i * k as i64 - 1)
        .collect();
    let reversed: Vec<i64> = origin.into_iter().rev().collect();
    // inference_indices = floor(linspace(0, len, num_steps, endpoint=False)).
    let len = reversed.len() as f64;
    (0..num_steps)
        .map(|j| {
            let idx = (len * j as f64 / num_steps as f64).floor() as usize;
            reversed[idx.min(reversed.len() - 1)] as usize
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// LCM — port of diffusers `LCMScheduler` (epsilon prediction; the SDXL world: scaled_linear betas,
// timestep_scaling=10, sigma_data=0.5, set_alpha_to_one=True → final ᾱ = 1).
// ---------------------------------------------------------------------------------------------

/// Latent Consistency Model sampler. Predicts `x₀` from `eps`, applies the consistency boundary
/// scalings `c_skip`/`c_out`, and re-noises between steps. ~2–8 steps; CFG ≈ 1.
pub struct LcmSampler {
    sched: AlphaSchedule,
    timesteps: Vec<usize>,
    timestep_scaling: f32,
    /// The compute dtype the model's forward expects (latents are cast to this in
    /// [`DiffusionSampler::scale_model_input`]); the step math runs f32.
    model_dtype: Dtype,
}

impl LcmSampler {
    /// Build for `num_steps` inference steps. `original_inference_steps` is diffusers' default 50.
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        model_dtype: Dtype,
    ) -> Self {
        Self {
            sched,
            timesteps: lcm_style_timesteps(
                num_train_timesteps,
                original_inference_steps,
                num_steps,
            ),
            timestep_scaling: 10.0,
            model_dtype,
        }
    }
}

impl DiffusionSampler for LcmSampler {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i] as f32
    }

    fn scale_model_input(&self, x: &Array, _i: usize) -> Result<Array> {
        // LCMScheduler.scale_model_input is identity; cast to the model's compute dtype.
        Ok(x.as_dtype(self.model_dtype)?)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // init_noise_sigma = 1.0.
        Ok(noise.as_dtype(Dtype::Float32)?)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        let denoised = self.denoised(model_output, x, i)?;
        if i == self.timesteps.len() - 1 {
            // No re-noise on the final step (also: one-step sampling never re-noises).
            return Ok(denoised);
        }
        // prev = √ᾱ_prev·denoised + √β̄_prev·noise.
        let apt_prev = self.sched.acp(self.timesteps[i + 1]);
        let bpt_prev = 1.0 - apt_prev;
        let noise = random::normal::<f32>(denoised.shape(), None, None, None)?;
        Ok(add(
            &multiply(&denoised, scalar(apt_prev.sqrt() as f32))?,
            &multiply(&noise, scalar(bpt_prev.sqrt() as f32))?,
        )?)
    }
}

impl LcmSampler {
    /// The deterministic consistency prediction at step `i` — diffusers' `denoised` (the second
    /// element of `LCMScheduler.step(return_dict=False)`), before the between-step re-noise. Used by
    /// the scheduler-isolation parity gate (the re-noise draws from a different RNG than torch, so
    /// only the deterministic core is bit-comparable). Epsilon prediction; `clip_sample=False`.
    pub fn denoised(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        let eps = model_output.as_dtype(Dtype::Float32)?;
        let x = x.as_dtype(Dtype::Float32)?;
        let t = self.timesteps[i];
        let apt = self.sched.acp(t);
        let bpt = 1.0 - apt;
        // Boundary-condition scalings (sigma_data=0.5 → sigma_data²=0.25).
        let scaled_t = t as f64 * self.timestep_scaling as f64;
        let c_skip = 0.25 / (scaled_t * scaled_t + 0.25);
        let c_out = scaled_t / (scaled_t * scaled_t + 0.25).sqrt();
        // pred_x0 = (x − √β̄·eps) / √ᾱ.
        let pred_x0 = divide(
            &subtract(&x, &multiply(&eps, scalar(bpt.sqrt() as f32))?)?,
            scalar(apt.sqrt() as f32),
        )?;
        // denoised = c_out·pred_x0 + c_skip·x.
        Ok(add(
            &multiply(&pred_x0, scalar(c_out as f32))?,
            &multiply(&x, scalar(c_skip as f32))?,
        )?)
    }
}

// ---------------------------------------------------------------------------------------------
// SDXL-Lightning — port of diffusers `EulerDiscreteScheduler(timestep_spacing="trailing")`,
// epsilon prediction, no churn (s_churn=0), `final_sigmas_type="zero"`. CFG is disabled for
// Lightning (guidance 1.0). Step counts 2/4/8 must match the loaded acceleration LoRA.
// ---------------------------------------------------------------------------------------------

/// SDXL-Lightning sampler: trailing-spaced Euler. The latents live in diffusers' un-normalized
/// (σ-scaled) space; [`DiffusionSampler::scale_model_input`] divides by `√(σ²+1)` before the U-Net.
pub struct LightningSampler {
    /// Interpolated sigmas at the trailing timesteps, length `num_steps + 1` (trailing `0.0`).
    sigmas: Vec<f32>,
    /// The (float) trailing timesteps fed to the U-Net, length `num_steps`.
    timesteps: Vec<f32>,
    model_dtype: Dtype,
}

impl LightningSampler {
    /// Build for `num_steps` (2/4/8). Timesteps are diffusers' trailing spacing
    /// `round(arange(N, 0, −N/num_steps)) − 1`; sigmas are `√((1-ᾱ)/ᾱ)` linearly interpolated at
    /// those (float) timesteps, with a trailing `0` (`final_sigmas_type="zero"`).
    pub fn new(
        sched: &AlphaSchedule,
        num_train_timesteps: usize,
        num_steps: usize,
        model_dtype: Dtype,
    ) -> Self {
        let step_ratio = num_train_timesteps as f64 / num_steps as f64;
        // arange(N, 0, -step_ratio): N, N-step_ratio, … (num_steps entries), round, then −1.
        let timesteps: Vec<f32> = (0..num_steps)
            .map(|j| {
                let v = num_train_timesteps as f64 - step_ratio * j as f64;
                (v.round() - 1.0) as f32
            })
            .collect();
        // Full per-train-step sigma table for the linear interp.
        let full: Vec<f64> = (0..num_train_timesteps)
            .map(|t| sched.sigma_at(t))
            .collect();
        let interp = |t: f32| -> f32 {
            // np.interp over xp = arange(0, N), fp = full. t is in [0, N-1] here.
            let tt = (t as f64).clamp(0.0, (num_train_timesteps - 1) as f64);
            let lo = tt.floor() as usize;
            let hi = (lo + 1).min(num_train_timesteps - 1);
            let frac = tt - lo as f64;
            (full[lo] * (1.0 - frac) + full[hi] * frac) as f32
        };
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp(t)).collect();
        sigmas.push(0.0); // final_sigmas_type = "zero"
        Self {
            sigmas,
            timesteps,
            model_dtype,
        }
    }

    fn init_noise_sigma(&self) -> f32 {
        // timestep_spacing in {linspace, trailing} → init_noise_sigma = max(sigmas).
        self.sigmas.iter().copied().fold(0.0_f32, f32::max)
    }
}

impl DiffusionSampler for LightningSampler {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        // x / √(σ²+1), then cast to the model's compute dtype.
        let sigma = self.sigmas[i] as f64;
        let scaled = divide(x, scalar(((sigma * sigma + 1.0).sqrt()) as f32))?;
        Ok(scaled.as_dtype(self.model_dtype)?)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // latents = randn · init_noise_sigma.
        Ok(multiply(
            &noise.as_dtype(Dtype::Float32)?,
            scalar(self.init_noise_sigma()),
        )?)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        // Euler step, epsilon prediction, gamma=0: pred_x0 = x − σ·eps; derivative = eps;
        // prev = x + eps·(σ_next − σ). (diffusers upcasts the sample to f32 for the update.)
        let eps = model_output.as_dtype(Dtype::Float32)?;
        let x = x.as_dtype(Dtype::Float32)?;
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        Ok(add(&x, &multiply(&eps, scalar(dt))?)?)
    }
}

// ---------------------------------------------------------------------------------------------
// Hyper-SD — port of diffusers `TCDScheduler` (epsilon prediction). The story allows "TCD or
// default"; TCD covers both the per-step and unified Hyper-SD LoRAs (eta=0 → deterministic
// trajectory-consistency = DDIM-like; eta>0 → stochastic). CFG disabled (guidance 1.0).
// ---------------------------------------------------------------------------------------------

/// Hyper-SD sampler: Trajectory Consistency Distillation. Like LCM but steps to an intermediate
/// noise level `s = ⌊(1−η)·t_prev⌋` and (for `η>0`) re-noises across the `t_prev`/`s` gap.
pub struct TcdSampler {
    sched: AlphaSchedule,
    timesteps: Vec<usize>,
    eta: f32,
    model_dtype: Dtype,
}

impl TcdSampler {
    /// Build for `num_steps`. `original_inference_steps` is diffusers' default 50; `eta` is the
    /// stochasticity (`0.0` = deterministic; ByteDance's unified LoRA recommends ~`0.3`).
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        eta: f32,
        model_dtype: Dtype,
    ) -> Self {
        Self {
            sched,
            timesteps: lcm_style_timesteps(
                num_train_timesteps,
                original_inference_steps,
                num_steps,
            ),
            eta,
            model_dtype,
        }
    }
}

impl DiffusionSampler for TcdSampler {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i] as f32
    }

    fn scale_model_input(&self, x: &Array, _i: usize) -> Result<Array> {
        Ok(x.as_dtype(self.model_dtype)?)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // init_noise_sigma = 1.0.
        Ok(noise.as_dtype(Dtype::Float32)?)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        let pred_noised = self.pred_noised(model_output, x, i)?;
        let last = i == self.timesteps.len() - 1;
        if self.eta > 0.0 && !last {
            // prev = √(ᾱ_prev/ᾱ_s)·pred_noised + √(1 − ᾱ_prev/ᾱ_s)·noise.
            let prev_t = self.timesteps[i + 1];
            let timestep_s = ((1.0 - self.eta as f64) * prev_t as f64).floor() as usize;
            let ratio = self.sched.acp(prev_t) / self.sched.acp(timestep_s);
            let noise = random::normal::<f32>(pred_noised.shape(), None, None, None)?;
            return Ok(add(
                &multiply(&pred_noised, scalar(ratio.sqrt() as f32))?,
                &multiply(&noise, scalar((1.0 - ratio).max(0.0).sqrt() as f32))?,
            )?);
        }
        Ok(pred_noised)
    }
}

impl TcdSampler {
    /// The deterministic noised prediction `x_s` at step `i` — diffusers' `pred_noised_sample` (the
    /// second element of `TCDScheduler.step(return_dict=False)`), before the `η>0` re-noise. Used by
    /// the scheduler-isolation parity gate. Epsilon prediction.
    pub fn pred_noised(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        let eps = model_output.as_dtype(Dtype::Float32)?;
        let x = x.as_dtype(Dtype::Float32)?;
        let t = self.timesteps[i];
        let last = i == self.timesteps.len() - 1;
        // prev_timestep = timesteps[i+1] if it exists else 0; timestep_s = floor((1−η)·prev_t).
        let prev_t = if last { 0 } else { self.timesteps[i + 1] };
        let timestep_s = ((1.0 - self.eta as f64) * prev_t as f64).floor() as usize;
        let apt = self.sched.acp(t);
        let bpt = 1.0 - apt;
        let aps = self.sched.acp(timestep_s);
        let bps = 1.0 - aps;
        // pred_x0 = (x − √β̄·eps)/√ᾱ; pred_noised = √ᾱ_s·pred_x0 + √β̄_s·eps.
        let pred_x0 = divide(
            &subtract(&x, &multiply(&eps, scalar(bpt.sqrt() as f32))?)?,
            scalar(apt.sqrt() as f32),
        )?;
        Ok(add(
            &multiply(&pred_x0, scalar(aps.sqrt() as f32))?,
            &multiply(&eps, scalar(bps.sqrt() as f32))?,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdxl_sched() -> AlphaSchedule {
        AlphaSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap()
    }

    #[test]
    fn alphas_cumprod_is_monotonic_decreasing() {
        let s = sdxl_sched();
        assert_eq!(s.alphas_cumprod.len(), 1000);
        // ᾱ starts near 1 and decreases toward 0.
        assert!(s.alphas_cumprod[0] > 0.99);
        assert!(*s.alphas_cumprod.last().unwrap() < 0.01);
        assert!(s.alphas_cumprod.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn lcm_4step_timesteps_match_diffusers() {
        // diffusers LCMScheduler.set_timesteps(4) on N=1000, original=50.
        let ts = lcm_style_timesteps(1000, 50, 4);
        assert_eq!(ts, vec![999, 759, 499, 259]);
    }

    #[test]
    fn lcm_8step_timesteps_descend_from_999() {
        let ts = lcm_style_timesteps(1000, 50, 8);
        assert_eq!(ts.len(), 8);
        assert_eq!(ts[0], 999);
        assert!(ts.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn lightning_trailing_timesteps_match_diffusers() {
        // diffusers EulerDiscreteScheduler(timestep_spacing="trailing").set_timesteps(4).
        let s = LightningSampler::new(&sdxl_sched(), 1000, 4, Dtype::Float32);
        assert_eq!(s.timesteps, vec![999.0, 749.0, 499.0, 249.0]);
        // sigmas: num_steps + 1 with a trailing 0, strictly decreasing.
        assert_eq!(s.sigmas.len(), 5);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
        // init_noise_sigma = the largest sigma.
        assert_eq!(s.init_noise_sigma(), s.sigmas[0]);
    }

    #[test]
    fn tcd_shares_lcm_timesteps() {
        let ts = lcm_style_timesteps(1000, 50, 4);
        let t = TcdSampler::new(sdxl_sched(), 1000, 50, 4, 0.3, Dtype::Float32);
        assert_eq!(t.timesteps, ts);
    }

    #[test]
    fn samplers_report_step_count() {
        let lcm = LcmSampler::new(sdxl_sched(), 1000, 50, 4, Dtype::Float32);
        assert_eq!(lcm.num_steps(), 4);
        let light = LightningSampler::new(&sdxl_sched(), 1000, 2, Dtype::Float32);
        assert_eq!(light.num_steps(), 2);
        let tcd = TcdSampler::new(sdxl_sched(), 1000, 50, 8, 0.0, Dtype::Float32);
        assert_eq!(tcd.num_steps(), 8);
    }

    fn scalar1(v: f32) -> Array {
        Array::from_slice(&[v], &[1])
    }
    fn val(a: &Array) -> f32 {
        a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>()[0]
    }

    // Inline step-math reference values from diffusers (eps=0.7, x=0.3 at step 0 of the 4-step
    // schedule, t=999), so CI validates the per-step math without the (gitignored) golden file. The
    // full per-step parity sweep is `tests/accel_sampler_parity.rs`. See `dump_sdxl_accel_golden.py`.
    #[test]
    fn lcm_step0_denoised_matches_diffusers() {
        let s = LcmSampler::new(sdxl_sched(), 1000, 50, 4, Dtype::Float32);
        let d = s.denoised(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&d) - (-5.835_607)).abs() < 1e-3, "got {}", val(&d));
    }

    #[test]
    fn lightning_step0_matches_diffusers() {
        let s = LightningSampler::new(&sdxl_sched(), 1000, 4, Dtype::Float32);
        let scaled = s.scale_model_input(&scalar1(0.3), 0).unwrap();
        assert!(
            (val(&scaled) - 0.020_479_47).abs() < 1e-4,
            "scaled {}",
            val(&scaled)
        );
        let prev = s.step(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!(
            (val(&prev) - (-7.073_041)).abs() < 1e-3,
            "prev {}",
            val(&prev)
        );
    }

    #[test]
    fn tcd_eta0_step0_pred_noised_matches_diffusers() {
        let s = TcdSampler::new(sdxl_sched(), 1000, 50, 4, 0.0, Dtype::Float32);
        let pn = s.pred_noised(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&pn) - (-0.651_963_8)).abs() < 1e-4, "got {}", val(&pn));
    }
}
