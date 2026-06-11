//! Learning-rate schedules for training (sc-3043). A faithful port of the SceneWorks Python
//! kernel's `lr_decay_multiplier` / `lr_schedule_updates` (`training_adapters.py`): a multiplier in
//! `[0,1]` applied to the base LR, with an optional linear *warmup* ramp followed by `constant` /
//! `linear` / `cosine` decay over the remaining updates. mlx-rs has no built-in scheduler, so the
//! trainer mutates `optimizer.lr` between *updates* (post grad-accumulation) using these.
//!
//! This is the *learning-rate* scheduler — distinct from the flow-matching *noise* schedule
//! (timestep sampling), which is a separate concept.

/// The LR-schedule shape (mirrors SceneWorks `SUPPORTED_LR_SCHEDULERS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LrSchedule {
    /// Hold the optimizer LR fixed (after any warmup ramp).
    #[default]
    Constant,
    /// Linearly decay to 0 over the post-warmup updates.
    Linear,
    /// Half-cosine decay to 0 over the post-warmup updates.
    Cosine,
}

impl LrSchedule {
    /// Parse the free-form contract string; unknown / empty → `Constant`.
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "linear" => LrSchedule::Linear,
            "cosine" => LrSchedule::Cosine,
            _ => LrSchedule::Constant,
        }
    }
}

/// The LR multiplier in `[0,1]` for `update` (0-based) of a run of `total` updates with `warmup`
/// warmup updates. Port of `lr_decay_multiplier`:
///   * during warmup (`update < warmup`): linear ramp `(update+1)/(warmup+1)` — starts at
///     `1/(warmup+1)`, never 0, so there are no dead steps;
///   * after warmup: `Constant` → 1; `Linear` → `1-progress`; `Cosine` → `0.5(1+cos(π·progress))`,
///     where `progress = clamp((update-warmup)/(total-warmup), 0, 1)`.
pub fn lr_multiplier(schedule: LrSchedule, update: u32, total: u32, warmup: u32) -> f32 {
    if warmup > 0 && update < warmup {
        return (update as f32 + 1.0) / (warmup as f32 + 1.0);
    }
    if total <= warmup {
        return 1.0;
    }
    let progress =
        ((update as f32 - warmup as f32) / (total as f32 - warmup as f32)).clamp(0.0, 1.0);
    match schedule {
        LrSchedule::Constant => 1.0,
        LrSchedule::Linear => 1.0 - progress,
        LrSchedule::Cosine => 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()),
    }
}

/// Convert micro-step counts to *optimizer-update* counts under gradient accumulation. Port of
/// `lr_schedule_updates`: an optimizer update fires every `accum` micro-steps, so
/// `total_updates = ceil(steps/accum)` and `warmup_updates = ceil(warmup_steps/accum)`, clamped to
/// `total-1`. Returns `(total_updates, warmup_updates)`.
pub fn schedule_updates(steps: u32, grad_accum: u32, warmup_steps: u32) -> (u32, u32) {
    let accum = grad_accum.max(1);
    let total = steps.max(1).div_ceil(accum).max(1);
    let warmup = warmup_steps.div_ceil(accum);
    (total, warmup.min(total.saturating_sub(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_is_flat_and_warmup_ramps() {
        // No warmup → flat 1.0 throughout.
        for u in 0..10 {
            assert_eq!(lr_multiplier(LrSchedule::Constant, u, 10, 0), 1.0);
        }
        // Warmup=4: ramp 1/5,2/5,3/5,4/5 then hold at 1.0.
        assert!((lr_multiplier(LrSchedule::Constant, 0, 100, 4) - 0.2).abs() < 1e-6);
        assert!((lr_multiplier(LrSchedule::Constant, 3, 100, 4) - 0.8).abs() < 1e-6);
        assert_eq!(lr_multiplier(LrSchedule::Constant, 4, 100, 4), 1.0);
    }

    #[test]
    fn linear_decays_to_zero_after_warmup() {
        // total=10, warmup=0: 1.0 at start, 0.0 at the last update.
        assert!((lr_multiplier(LrSchedule::Linear, 0, 10, 0) - 1.0).abs() < 1e-6);
        assert!((lr_multiplier(LrSchedule::Linear, 5, 10, 0) - 0.5).abs() < 1e-6);
        assert!(lr_multiplier(LrSchedule::Linear, 10, 10, 0).abs() < 1e-6);
    }

    #[test]
    fn cosine_half_period() {
        // Cosine: 1.0 at progress 0, 0.5 at the midpoint, 0.0 at the end.
        assert!((lr_multiplier(LrSchedule::Cosine, 0, 8, 0) - 1.0).abs() < 1e-6);
        assert!((lr_multiplier(LrSchedule::Cosine, 4, 8, 0) - 0.5).abs() < 1e-6);
        assert!(lr_multiplier(LrSchedule::Cosine, 8, 8, 0).abs() < 1e-6);
    }

    #[test]
    fn updates_account_for_accumulation() {
        // 100 micro-steps, accum 4 → 25 updates; warmup 8 steps → 2 update-warmups.
        assert_eq!(schedule_updates(100, 4, 8), (25, 2));
        // accum 1 → updates == steps.
        assert_eq!(schedule_updates(50, 1, 5), (50, 5));
        // ceil division (101/4 = 26 updates).
        assert_eq!(schedule_updates(101, 4, 0), (26, 0));
        // warmup never reaches total.
        let (total, warmup) = schedule_updates(10, 1, 100);
        assert_eq!(total, 10);
        assert_eq!(warmup, 9);
    }

    #[test]
    fn parse_is_lenient() {
        assert_eq!(LrSchedule::parse("Cosine"), LrSchedule::Cosine);
        assert_eq!(LrSchedule::parse(" linear "), LrSchedule::Linear);
        assert_eq!(LrSchedule::parse("whatever"), LrSchedule::Constant);
        assert_eq!(LrSchedule::parse(""), LrSchedule::Constant);
    }
}
