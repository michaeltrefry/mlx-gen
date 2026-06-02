//! Z-Image adapter consumption (sc-2602) — the model-specific orchestration over the core
//! adapter seam ([`mlx_gen::adapters::loader::apply_adapter_specs`], sc-2534).
//!
//! The core seam dispatches LoRA vs LoKr and installs onto an [`AdaptableHost`]; what is
//! Z-Image-specific is (a) the key→module map (the top-level `AdaptableHost for ZImageTransformer`,
//! the Rust analog of the fork's `ZImageLoRAMapping`, lives in `transformer.rs`) and (b) the
//! per-file LoRA namespace prefix. LoKr files carry bare module-path keys; LoRA files carry a
//! namespace prefix — peft `save_lora_weights` emits `transformer.`, ComfyUI/diffusion exports
//! emit `diffusion_model.`, and some are bare. We detect and strip it per file so the keys land on
//! the host paths. Stacking + mixed LoRA/LoKr fall out of the core loaders (sc-2343).

use mlx_gen::adapters::loader::{apply_adapter_specs, is_lokr, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// LoRA key namespace prefixes Z-Image adapter files use, tried in order; the first that any key
/// begins with is stripped. LoKr files are bare (no prefix). (kohya `lora_unet_…_` underscore
/// files are NOT handled here — their dots are flattened to underscores, so they need the fork's
/// explicit per-target pattern matcher rather than path-splitting; tracked as a follow-on.)
const LORA_PREFIXES: [&str; 2] = ["transformer.", "diffusion_model."];

/// Apply every adapter in `specs` onto a Z-Image transformer `host`, stacking in order. Each file
/// is dispatched (LoKr vs LoRA) and installed via the core seam, with the LoRA namespace prefix
/// detected per file. Returns the merged [`ApplyReport`]. Errors — never silently drops — if
/// nothing matched at all, or if any adapter target resolved to no module.
pub fn apply_z_image_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        // LoKr keys are bare module paths; LoRA files carry a namespace prefix.
        let prefix = if is_lokr(&w) {
            None
        } else {
            detect_lora_prefix(&w)
        };
        let report = apply_adapter_specs(host, std::slice::from_ref(spec), prefix)?;
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }

    if !specs.is_empty() && combined.applied == 0 {
        return Err(Error::Msg(format!(
            "z_image adapters: no target modules matched across {} adapter file(s) — check the \
             adapter format/prefix (expected diffusers/peft LoRA or LoKr keys); kohya `lora_unet_` \
             files are not yet supported",
            specs.len()
        )));
    }
    if !combined.unmatched_paths.is_empty() {
        return Err(Error::Msg(format!(
            "z_image adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            combined.unmatched_paths.len(),
            combined.unmatched_paths
        )));
    }
    Ok(combined)
}

/// The LoRA namespace prefix present in `w`'s keys, if any (see [`LORA_PREFIXES`]).
fn detect_lora_prefix(w: &Weights) -> Option<&'static str> {
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    LORA_PREFIXES
        .iter()
        .find(|p| keys.iter().any(|k| k.starts_with(*p)))
        .copied()
}
