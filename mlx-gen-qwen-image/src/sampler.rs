//! Qwen-Image flow-match sampler under the core [`mlx_gen::DiffusionSampler`] seam (sc-2909).
//!
//! Qwen-Image is **flow-match**, so the DDPM `alphas_cumprod`-world acceleration samplers shipped
//! with sc-2769 (`LcmSampler`/`LightningSampler`/`TcdSampler`) do not apply. Instead this wraps the
//! crate's [`FlowMatchEuler`] schedule as a [`DiffusionSampler`], so both the production schedule
//! (`qwen_scheduler`) and the few-step **Lightning** schedule drive the same generic denoise loop.
//!
//! - [`FlowMatchSampler::new`] wraps an arbitrary schedule (the production `qwen_scheduler`, i.e. the
//!   fork's `LinearScheduler`: terminal-shift 0.02, resolution-dependent μ).
//! - [`FlowMatchSampler::lightning`] builds the **official lightx2v Qwen-Image-Lightning** schedule,
//!   reproducing diffusers' `FlowMatchEulerDiscreteScheduler` under that LoRA's model-card config: a
//!   static flow-match shift of `3.0` (`base_shift = max_shift = ln 3`, which collapses dynamic
//!   shifting to a resolution-independent constant) with **no terminal rescale** (`shift_terminal =
//!   None`), over the **full** sigma span `linspace(1, 1/num_train_timesteps, n)` — NOT the mflux
//!   `linspace(1, 1/n, n)` of [`crate::pipeline::qwen_scheduler`]. The matching distillation LoRA
//!   (e.g. `lightx2v/Qwen-Image-Lightning`) must be supplied via `spec.adapters`; the CFG-distilled
//!   LoRAs run CFG-off (a single forward). Validated bit-exact-ish vs diffusers
//!   (`tests/lightning_parity.rs`, `tools/dump_qwen_lightning_golden.py`).
//!
//! Timestep convention: Qwen feeds the **raw sigma** as the model timestep — the transformer's
//! `QwenTimesteps` time-proj scales by ×1000 internally (so `embed(sigma·1000)` matches diffusers'
//! `timesteps = sigmas·1000` fed to a scale-1 embedding). Hence [`DiffusionSampler::timestep`]
//! returns `sigmas[i]`, exactly what the pixel-parity production loop already passes.

use mlx_rs::Array;

use mlx_gen::{DiffusionSampler, FlowMatchEuler, Result};

/// The official lightx2v Qwen-Image-Lightning flow-match shift (`exp(μ)`, μ = `ln 3`). The model
/// card sets `base_shift = max_shift = ln 3`, so the per-resolution dynamic shift collapses to this
/// constant; `shift_terminal = None` means no terminal-sigma rescale (unlike the production
/// `qwen_scheduler`'s 0.02).
pub const LIGHTNING_SHIFT: f32 = 3.0;

/// Flow-match training timesteps (diffusers `num_train_timesteps`) — the Lightning sigma span runs
/// down to `1/LIGHTNING_NUM_TRAIN_TIMESTEPS`, the full diffusers minimum (not the mflux `1/n`).
const LIGHTNING_NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Build the Lightning sigmas, reproducing diffusers' `FlowMatchEulerDiscreteScheduler.set_timesteps`
/// under the official config: exponential time-shift `exp(μ)/(exp(μ) + (1/σ − 1))` with `exp(μ) =
/// 3.0`, applied over `linspace(1.0, 1/num_train_timesteps, n)`, then a trailing `0.0`. The `1/1000`
/// floor (vs the production schedule's `1/n`) is the whole difference — proven bit-exact vs diffusers
/// in `tests/lightning_parity.rs` (e.g. 4-step → `[1.0, 0.857.., 0.601.., 0.00299.., 0.0]`).
fn lightning_sigmas(num_steps: usize) -> Vec<f32> {
    let n = num_steps.max(1);
    let e = LIGHTNING_SHIFT; // exp(μ)
    let sigma_min = 1.0 / LIGHTNING_NUM_TRAIN_TIMESTEPS;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, sigma_min, n)
            let s = if n == 1 {
                1.0
            } else {
                1.0 + (sigma_min - 1.0) * (i as f32) / ((n - 1) as f32)
            };
            e / (e + (1.0 / s - 1.0))
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// A [`FlowMatchEuler`] schedule adapted to the generic [`DiffusionSampler`] seam.
///
/// Flow-match has no model-input scaling and a unit-noise prior, so [`DiffusionSampler::scale_model_input`]
/// (the trait default) and [`DiffusionSampler::scale_initial_noise`] are identity — the Qwen pipeline
/// prepares the f32 noise (and the img2img blend) itself. The per-step update is the schedule's Euler
/// step, identical to the pre-trait production loop, so routing the production path through this
/// wrapper is bit-for-bit unchanged.
pub struct FlowMatchSampler {
    sched: FlowMatchEuler,
}

impl FlowMatchSampler {
    /// Wrap an existing schedule (the production `qwen_scheduler`).
    pub fn new(sched: FlowMatchEuler) -> Self {
        Self { sched }
    }

    /// Build the few-step **Lightning** schedule for `num_steps` (typically 4 or 8, matching the
    /// loaded distillation LoRA): the official diffusers Lightning sigmas (see [`lightning_sigmas`]).
    pub fn lightning(num_steps: usize) -> Self {
        Self::new(FlowMatchEuler {
            sigmas: lightning_sigmas(num_steps),
        })
    }

    /// The schedule sigma at step `i` (length `num_steps + 1`, trailing `0.0`). Used by the img2img
    /// noise blend, which seeds the loop at `sigma(start_step)`.
    pub fn sigma(&self, i: usize) -> f32 {
        self.sched.sigmas[i]
    }
}

impl DiffusionSampler for FlowMatchSampler {
    fn num_steps(&self) -> usize {
        self.sched.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        // Raw sigma — the transformer's time-proj applies the ×1000 scale (see module docs).
        self.sched.sigmas[i]
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // Flow-match prior is unit noise; the pipeline owns noise creation + the img2img blend.
        Ok(noise.clone())
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        // Euler flow-match update: x + (sigma[i+1] - sigma[i]) · velocity.
        self.sched.step(x, model_output, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::qwen_scheduler;

    #[test]
    fn lightning_4step_sigmas_match_diffusers() {
        // The official recipe as realized in diffusers FlowMatchEulerDiscreteScheduler (shift=3.0,
        // shift_terminal=None) over linspace(1, 1/1000, 4): bit-exact values from
        // `tools/dump_qwen_lightning_golden.py` (the tight cross-impl gate is `tests/lightning_parity.rs`).
        let s = FlowMatchSampler::lightning(4);
        assert_eq!(s.num_steps(), 4);
        let expected = [1.0_f32, 0.857_326_5, 0.600_719_4, 0.002_994_012, 0.0];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (s.sigma(i) - want).abs() < 1e-5,
                "lightning sigma[{i}] = {} want {want}",
                s.sigma(i)
            );
        }
        // No terminal rescale: the span runs to the diffusers 1/1000 floor (≈0.003), NOT 0.02.
        assert!(s.sigma(3) < 0.01);
    }

    #[test]
    fn lightning_8step_sigmas_match_diffusers() {
        let s = FlowMatchSampler::lightning(8);
        assert_eq!(s.num_steps(), 8);
        let expected = [
            1.0_f32,
            0.947_426_5,
            0.882_498_3,
            0.800_279_9,
            0.692_804_4,
            0.546_321_5,
            0.334_886_8,
            0.002_994_012,
            0.0,
        ];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (s.sigma(i) - want).abs() < 1e-5,
                "lightning sigma[{i}] = {} want {want}",
                s.sigma(i)
            );
        }
    }

    #[test]
    fn lightning_is_resolution_independent() {
        // base_shift == max_shift ⇒ μ is constant ⇒ the schedule ignores width/height.
        let a = FlowMatchSampler::lightning(8);
        let b = FlowMatchSampler::lightning(8);
        for i in 0..=8 {
            assert_eq!(a.sigma(i), b.sigma(i));
        }
    }

    #[test]
    fn timestep_is_raw_sigma() {
        // The model timestep is the raw sigma (the time-proj scales ×1000), matching the production
        // loop; NOT FlowMatchEuler::timestep's `1 - sigma` (used by other families).
        let s = FlowMatchSampler::lightning(4);
        for i in 0..4 {
            assert_eq!(s.timestep(i), s.sigma(i));
        }
    }

    #[test]
    fn wrapping_production_scheduler_preserves_sigmas() {
        // Routing the production `qwen_scheduler` through the wrapper must expose the identical
        // schedule (the base path stays bit-for-bit unchanged).
        let sched = qwen_scheduler(8, 1024, 1024);
        let sigmas = sched.sigmas.clone();
        let s = FlowMatchSampler::new(sched);
        assert_eq!(s.num_steps(), 8);
        for (i, want) in sigmas.iter().enumerate() {
            assert_eq!(s.sigma(i), *want);
            if i < 8 {
                assert_eq!(s.timestep(i), *want);
            }
        }
    }
}
