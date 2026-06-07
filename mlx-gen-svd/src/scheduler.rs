//! SVD EDM scheduler — the `EulerDiscreteScheduler` (`use_karras_sigmas`, `timestep_type="continuous"`,
//! `prediction_type="v_prediction"`) used by `StableVideoDiffusionPipeline`. Port of the diffusers
//! `set_timesteps` + `_convert_to_karras` + `scale_model_input` + v-prediction `step`.
//!
//! Because the config sets `sigma_min`/`sigma_max` explicitly, the sigma schedule is **pure Karras**
//! over those (the betas/alphas path in `set_timesteps` is computed but unused for the sigma *values*),
//! and the model timestep is `0.25·ln(σ)` (the continuous-v_prediction branch). All scalar math is
//! done on the host in f32; the array ops apply the scalars.

use mlx_rs::ops::{add, divide, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

use crate::config::SchedulerConfig;

/// The discrete EDM schedule for a run: `sigmas` (len `n+1`, descending, `sigmas[0]` = `sigma_max`,
/// `sigmas[n]` = 0) and the per-step model `timesteps` (len `n`, `0.25·ln(σ)`).
#[derive(Clone, Debug)]
pub struct EdmSchedule {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl EdmSchedule {
    /// Build the Karras schedule for `num_steps` (diffusers `_convert_to_karras`):
    /// `ramp = linspace(0,1,n)`, `σ_i = (σmax^(1/ρ) + ramp_i·(σmin^(1/ρ) − σmax^(1/ρ)))^ρ`, then
    /// append a final `0` (`final_sigmas_type="zero"`). The model timesteps are `0.25·ln(σ)`.
    pub fn karras(num_steps: usize, cfg: &SchedulerConfig) -> Self {
        assert!(num_steps >= 1, "num_steps must be ≥ 1");
        let rho = cfg.rho as f64;
        let (smin, smax) = (cfg.sigma_min as f64, cfg.sigma_max as f64);
        let min_inv = smin.powf(1.0 / rho);
        let max_inv = smax.powf(1.0 / rho);
        let mut sigmas: Vec<f32> = (0..num_steps)
            .map(|i| {
                // np.linspace(0,1,n): ramp[i] = i/(n-1) (n==1 → 0.0).
                let ramp = if num_steps == 1 {
                    0.0
                } else {
                    i as f64 / (num_steps - 1) as f64
                };
                (max_inv + ramp * (min_inv - max_inv)).powf(rho) as f32
            })
            .collect();
        let timesteps: Vec<f32> = sigmas
            .iter()
            .map(|&s| 0.25 * (s as f64).ln() as f32)
            .collect();
        sigmas.push(0.0); // final_sigmas_type = "zero"
        Self { sigmas, timesteps }
    }

    /// Number of denoise steps (`n`).
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// `init_noise_sigma` for the `leading` spacing SVD uses: `sqrt(max(σ)² + 1)` (the init latents are
    /// `noise · init_noise_sigma`). `sigmas[0]` is the max.
    pub fn init_noise_sigma(&self) -> f32 {
        ((self.sigmas[0] as f64).powi(2) + 1.0).sqrt() as f32
    }
}

/// `scale_model_input`: `x / sqrt(σ² + 1)` (the EDM `c_in`), applied before the UNet each step.
pub fn scale_model_input(x: &Array, sigma: f32) -> Result<Array> {
    let c_in = (1.0 / ((sigma as f64).powi(2) + 1.0).sqrt()) as f32;
    Ok(multiply(x, Array::from_f32(c_in))?)
}

/// v-prediction → predicted clean sample (diffusers `step`):
/// `x̂0 = v·(−σ/sqrt(σ²+1)) + x/(σ²+1)`.
pub fn v_pred_denoised(model_output: &Array, sample: &Array, sigma: f32) -> Result<Array> {
    let s2 = (sigma as f64).powi(2);
    let c_out = (-(sigma as f64) / (s2 + 1.0).sqrt()) as f32;
    let c_skip = (1.0 / (s2 + 1.0)) as f32;
    Ok(add(
        multiply(model_output, Array::from_f32(c_out))?,
        multiply(sample, Array::from_f32(c_skip))?,
    )?)
}

/// Euler step (`s_churn=0` → `sigma_hat=σ`): `x' = x + (x − x̂0)/σ · (σ_next − σ)`.
pub fn euler_step(sample: &Array, denoised: &Array, sigma: f32, sigma_next: f32) -> Result<Array> {
    let derivative = divide(subtract(sample, denoised)?, Array::from_f32(sigma))?;
    let dt = sigma_next - sigma;
    Ok(add(sample, multiply(derivative, Array::from_f32(dt))?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn karras_endpoints_and_zero_tail() {
        let s = EdmSchedule::karras(25, &SchedulerConfig::default());
        assert_eq!(s.sigmas.len(), 26);
        assert_eq!(s.timesteps.len(), 25);
        // σ[0] = sigma_max (700), σ[24] = sigma_min (0.002), σ[25] = 0.
        assert!((s.sigmas[0] - 700.0).abs() < 1e-1, "σ0 {}", s.sigmas[0]);
        assert!((s.sigmas[24] - 0.002).abs() < 1e-4, "σ24 {}", s.sigmas[24]);
        assert_eq!(s.sigmas[25], 0.0);
        // descending.
        for i in 0..25 {
            assert!(s.sigmas[i] > s.sigmas[i + 1], "not descending at {i}");
        }
        // timestep = 0.25·ln(σ).
        assert!((s.timesteps[0] - 0.25 * 700f32.ln()).abs() < 1e-4);
        // init_noise_sigma = sqrt(700²+1).
        assert!((s.init_noise_sigma() - (700.0f64 * 700.0 + 1.0).sqrt() as f32).abs() < 1e-1);
    }

    #[test]
    fn v_pred_and_euler_match_formulas() {
        let v = Array::from_slice(&[0.5f32, -0.5], &[2]);
        let x = Array::from_slice(&[2.0f32, 4.0], &[2]);
        let sigma = 3.0f32;
        let den = v_pred_denoised(&v, &x, sigma).unwrap();
        let s2 = sigma * sigma;
        let want: Vec<f32> = (0..2)
            .map(|i| {
                let (vv, xv) = (v.as_slice::<f32>()[i], x.as_slice::<f32>()[i]);
                vv * (-sigma / (s2 + 1.0).sqrt()) + xv / (s2 + 1.0)
            })
            .collect();
        for (g, w) in den.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "v_pred {g} vs {w}");
        }
        // euler.
        let nxt = euler_step(&x, &den, sigma, 1.5).unwrap();
        let want_e: Vec<f32> = (0..2)
            .map(|i| {
                let (xv, dv) = (x.as_slice::<f32>()[i], den.as_slice::<f32>()[i]);
                xv + (xv - dv) / sigma * (1.5 - sigma)
            })
            .collect();
        for (g, w) in nxt.as_slice::<f32>().iter().zip(&want_e) {
            assert!((g - w).abs() < 1e-5, "euler {g} vs {w}");
        }
    }
}
