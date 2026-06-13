//! sc-5142: the renderer's **ViT-conditioned** guidance-combine modes — the velocity combination half
//! of `sample_one_step` (`wan_diffusion.py` 795-1049, the `BerniniPipeline`-only modes the renderer
//! port left out of scope). These take the per-stream target predictions (each = a DiT forward over a
//! given packed-latent variant + one of the planner's 4 prompt-embed streams, sc-5142 slice B) and
//! combine them into the step's noise prediction.
//!
//! Two delta families:
//!   - **plain** (`vae_txt_vit`, `rv2v_wapg`) — raw `to − from` deltas.
//!   - **`apg_delta`** (`vae_txt_vit_wapg`, `r2v_wapg`) — each delta projected v-space (∥0.2/⊥1.0)
//!     against a per-mode reference (see [`crate::guidance::apg_delta`]).
//!
//! The `v2v_apg` mode is the x-space [`crate::guidance::normalized_guidance`] family (already in the
//! renderer foundation, sc-4706); it is dispatched in the denoise loop (slice C), not here.
//!
//! All predictions are `[1, n_target, C]` (the target-sliced packed-token velocity, batch 1).

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

use crate::guidance::apg_delta;

/// Fixed APG projection scales for the ViT-conditioned modes (`apg_delta` defaults).
const PARALLEL_SCALE: f32 = 0.2;
const ORTHOGONAL_SCALE: f32 = 1.0;

/// `base + Σ ω·delta` accumulator.
fn combine(base: &Array, terms: &[(f32, Array)]) -> Result<Array> {
    let mut acc = base.clone();
    for (w, d) in terms {
        acc = add(&acc, &multiply(d, Array::from_f32(*w))?)?;
    }
    Ok(acc)
}

/// Optionally APG-project a delta against `reference` (the per-mode reference); identity when `apg`
/// is false.
fn maybe_apg(delta: Array, reference: &Array, apg: bool) -> Result<Array> {
    if apg {
        apg_delta(&delta, reference, PARALLEL_SCALE, ORTHOGONAL_SCALE)
    } else {
        Ok(delta)
    }
}

/// `vae_txt_vit` (`apg = false`) / `vae_txt_vit_wapg` (`apg = true`) — the primary full-Bernini mode
/// (t2i / edit). Three cumulative deltas over the VAE-conditioned predictions, the APG reference being
/// the **higher-conditioned** ("to") prediction of each delta:
///
///   `base + ω_img·Δ(img←base) + ω_txt·Δ(txt←img) + ω_tgt·Δ(vit←txt)`
///
/// where `base = wotxt_wovit_wovae`, `img = wotxt_wovit_wvae`, `txt = wtxt_wovit_wvae`,
/// `vit = wtxt_wvit_wvae`.
#[allow(clippy::too_many_arguments)]
pub fn vae_txt_vit(
    base: &Array,
    img: &Array,
    txt: &Array,
    vit: &Array,
    omega_img: f32,
    omega_txt: f32,
    omega_tgt: f32,
    apg: bool,
) -> Result<Array> {
    let d_img = maybe_apg(subtract(img, base)?, img, apg)?;
    let d_txt = maybe_apg(subtract(txt, img)?, txt, apg)?;
    let d_vit = maybe_apg(subtract(vit, txt)?, vit, apg)?;
    combine(
        base,
        &[(omega_img, d_img), (omega_txt, d_txt), (omega_tgt, d_vit)],
    )
}

/// `rv2v_wapg` (`apg = false`) / `r2v_wapg` (`apg = true`) — the 5-prediction reference chain over the
/// video / image / text / ViT conditioning. The APG reference being the **lower-conditioned** ("from")
/// prediction of each delta:
///
///   `base + ω_vid·Δ(V←base) + ω_img·Δ(VI←V) + ω_txt·Δ(VTI←VI) + ω_tgt·Δ(VTIC←VTI)`
///
/// where `base = wotxt_wovit_wovae` and `V/VI/VTI/VTIC` are the progressively-conditioned predictions.
#[allow(clippy::too_many_arguments)]
pub fn rv2v_chain(
    base: &Array,
    eps_v: &Array,
    eps_vi: &Array,
    eps_vti: &Array,
    eps_vtic: &Array,
    omega_vid: f32,
    omega_img: f32,
    omega_txt: f32,
    omega_tgt: f32,
    apg: bool,
) -> Result<Array> {
    let d_vid = maybe_apg(subtract(eps_v, base)?, base, apg)?;
    let d_img = maybe_apg(subtract(eps_vi, eps_v)?, eps_v, apg)?;
    let d_txt = maybe_apg(subtract(eps_vti, eps_vi)?, eps_vi, apg)?;
    let d_vit = maybe_apg(subtract(eps_vtic, eps_vti)?, eps_vti, apg)?;
    combine(
        base,
        &[
            (omega_vid, d_vid),
            (omega_img, d_img),
            (omega_txt, d_txt),
            (omega_tgt, d_vit),
        ],
    )
}
