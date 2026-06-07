//! LTX-2.3 LoRA application (sc-2687) — wires the reference `mlx_video/lora/` into the Rust
//! pipeline. The reference *defines* its `lora/` module but never invokes it from `generate_av.py`,
//! so this is the wiring plus the LTX key→module map.
//!
//! **Strategy: forward-time residual** (the reference `lora/apply.py::LoRALinear`,
//! `out + scale·strength·(x·Aᵀ·Bᵀ)`), not a merged weight. The reference also offers
//! `apply_loras_to_model` (dequant Q8 → merge → dense bf16); residual is chosen because the shipped
//! transformer is **Q8-only** at 22B — merging a full attn+ff LoRA would dequantize ~15 GB to bf16,
//! and the net-new per-pass strength would double it — and because residual leaves the bit-exact base
//! forward (sc-2842) untouched. Installed onto the model tree's [`Linear`]s via
//! [`LtxDiT::adaptable_mut`](crate::transformer::LtxDiT::adaptable_mut).
//!
//! **Format.** PEFT `lora_A`/`lora_B` (`.default` infix tolerated) and kohya `lora_down`/`lora_up`,
//! per-module `.alpha` (default = rank); real LTX-2.3 files ship PEFT, bf16, `diffusion_model.`-prefixed.
//! `scale = alpha/rank` (the reference `LoRAWeights.scale`).
//!
//! **LoKr** (sc-2393) is net-new — the reference `lora/` is LoRA-only, so this is parity-PLUS. A LoKr
//! file (`networkType=lokr`, `‹path›.lokr_w1/w2[_a/_b]`) is parsed by the core `parse_lokr`, its
//! per-module `[out,in]` delta reconstructed via `reconstruct_lokr_delta` (`alpha/rank` folded in),
//! mapped through this same LTX key→module table, and installed as a forward-time residual carrying
//! the same per-pass strength as LoRA.
//!
//! **Skips, never errors-on-skip.** Mirrors the reference (`apply_loras_to_weights` counts skipped
//! modules, never raises): audio / `av_ca` / `a2v` targets (the video-only port has no such modules)
//! and the PixArt-spelled adaLN embedder (`linear_1/2` ≠ the checkpoint's `linear1/2`) resolve to no
//! module and are reported, not dropped. We error only if a non-empty spec list matched *nothing*.

use std::collections::BTreeMap;

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::loader::{is_lokr, parse_lokr};
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::transformer::LtxAdaptable;

/// LoRA key namespace prefixes stripped (longest-first), matching the reference
/// `_normalize_ltx_lora_key`. SceneWorks' trained LTX LoRAs use `diffusion_model.`.
const PREFIXES: [&str; 3] = ["model.diffusion_model.", "diffusion_model.", "model."];

/// Outcome of applying the LTX adapter specs: residuals installed and the LoRA module paths that
/// resolved to no target (surfaced, never silently dropped — audio/av_ca/a2v and PixArt-spelled
/// adaLN embedder leaves).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LtxLoraReport {
    pub applied: usize,
    pub skipped: Vec<String>,
}

#[derive(Clone, Copy)]
enum Role {
    Down, // lora_A / lora_down → A [rank, in]
    Up,   // lora_B / lora_up   → B [out, rank]
    Alpha,
}

#[derive(Default)]
struct LoraParts {
    down: Option<Array>,
    up: Option<Array>,
    alpha: Option<f32>,
}

/// Normalize a LoRA module path to the LTX checkpoint's naming (the reference
/// `_normalize_ltx_lora_key`): strip a known prefix, then the diffusers→checkpoint renames
/// `to_out.0`→`to_out`, `ff.net.0.proj`→`ff.proj_in`, `ff.net.2`→`ff.proj_out` (+ audio analogues,
/// which the video-only port then resolves to no module). The leading `.` in each pattern keeps
/// `.ff.net.*` from matching inside `audio_ff.net.*`, exactly as the reference relies on.
pub(crate) fn normalize_ltx_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();
    if let Some(head) = t.strip_suffix(".to_out.0") {
        t = format!("{head}.to_out");
    }
    t = t.replace(".to_out.0.", ".to_out.");
    t = t.replace(".ff.net.0.proj.", ".ff.proj_in.");
    t = t.replace(".ff.net.0.proj", ".ff.proj_in");
    t = t.replace(".ff.net.2.", ".ff.proj_out.");
    t = t.replace(".ff.net.2", ".ff.proj_out");
    t = t.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.");
    t = t.replace(".audio_ff.net.0.proj", ".audio_ff.proj_in");
    t = t.replace(".audio_ff.net.2.", ".audio_ff.proj_out.");
    t = t.replace(".audio_ff.net.2", ".audio_ff.proj_out");
    t
}

/// Suffix → role, longest-first. PEFT `lora_A/B`, kohya `lora_down/up`, the peft-export `.default`
/// infix, and a bare `.alpha`. `lora_A`/`lora_down` are the A (down) factor; `lora_B`/`lora_up` the B.
const SUFFIXES: [(&str, Role); 9] = [
    (".lora_A.default.weight", Role::Down),
    (".lora_B.default.weight", Role::Up),
    (".lora_down.default.weight", Role::Down),
    (".lora_up.default.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// Read a scalar `.alpha` as f32 regardless of on-disk dtype (real files ship it bf16; a direct
/// `as_slice::<f32>()` would panic on a dtype mismatch). A `[]`- or `[1]`-shaped scalar both read.
fn read_alpha(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.as_slice::<f32>()[0])
}

/// Per-pass user strengths for one adapter: `spec.pass_scales` (one per distilled stage, validated
/// to `num_passes`) or `spec.scale` (uniform — a length-1 vec the forward clamps into). The `strength`
/// is the user knob; `alpha/rank` is folded in separately (into B for LoRA via [`pass_scales`], into
/// the delta for LoKr via `reconstruct_lokr_delta`).
fn pass_strengths(spec: &AdapterSpec, num_passes: usize) -> Result<Vec<f32>> {
    match &spec.pass_scales {
        None => Ok(vec![spec.scale]),
        Some(v) => {
            if v.len() != num_passes {
                return Err(Error::Msg(format!(
                    "ltx_2_3 adapter {}: pass_scales has {} entries but the distilled pipeline runs \
                     {num_passes} passes",
                    spec.path.display(),
                    v.len()
                )));
            }
            Ok(v.clone())
        }
    }
}

/// LoRA per-pass effective scales for one resolved module: `(alpha/rank)·strength`. `strength` comes
/// from [`pass_strengths`]; the `alpha/rank·strength` product is computed in f64 then f32, matching
/// the reference's Python-float `scale * strength`. (LoKr bakes `alpha/rank` into the delta, so it
/// uses [`pass_strengths`] directly — no fold here.)
fn pass_scales(spec: &AdapterSpec, alpha: f32, rank: f32, num_passes: usize) -> Result<Vec<f32>> {
    let eff = |strength: f32| ((alpha as f64 / rank as f64) * strength as f64) as f32;
    Ok(pass_strengths(spec, num_passes)?
        .into_iter()
        .map(eff)
        .collect())
}

/// Install one LoRA file's residuals onto `host` at `spec`'s strength, accumulating into `report`.
fn apply_one(
    host: &mut impl LtxAdaptable,
    w: &Weights,
    spec: &AdapterSpec,
    num_passes: usize,
    report: &mut LtxLoraReport,
) -> Result<()> {
    // Group factors by normalized module path.
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a LoRA factor key (base weight / bundled extra) — ignore.
        };
        let path = normalize_ltx_key(stem);
        let parts = groups.entry(path).or_default();
        match role {
            Role::Down => parts.down = Some(w.require(&key)?.clone()),
            Role::Up => parts.up = Some(w.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(w.require(&key)?)?),
        }
    }

    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            // A down/up whose partner targeted a non-LoRA key — skip the orphan, surface the path.
            report.skipped.push(path);
            continue;
        };
        let segs: Vec<&str> = path.split('.').collect();
        let rank = down.shape()[0] as f32;
        let alpha = parts.alpha.unwrap_or(rank);
        let scales = pass_scales(spec, alpha, rank, num_passes)?;
        match host.adaptable_mut(&segs) {
            Some(lin) => {
                // Residual form: a = Aᵀ [in, rank], b = Bᵀ [rank, out]; factors keep their loaded
                // (bf16) dtype so the residual promotes against the activation like the reference.
                lin.push_lora(down.t(), up.t(), scales);
                report.applied += 1;
            }
            None => report.skipped.push(path),
        }
    }
    Ok(())
}

/// Install one LoKr file's residuals onto `host` at `spec`'s per-pass strength (sc-2393 — net-new,
/// the reference `lora/` has no LoKr). Each module's `[out,in]` delta is reconstructed from its
/// Kronecker factors via the core `reconstruct_lokr_delta` (`alpha/rank` baked in), keyed at the
/// target linear's base shape, then installed as a forward-time residual carrying the raw per-pass
/// strengths (no further alpha/rank fold). Skips/surfaces a path that resolves to no module, like
/// the LoRA path (audio/av_ca/a2v on the video-only port).
fn apply_one_lokr(
    host: &mut impl LtxAdaptable,
    w: &Weights,
    spec: &AdapterSpec,
    num_passes: usize,
    report: &mut LtxLoraReport,
) -> Result<()> {
    let file = parse_lokr(w)?;
    let strengths = pass_strengths(spec, num_passes)?;
    for (raw_path, factors) in &file.groups {
        let path = normalize_ltx_key(raw_path);
        let segs: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&segs) {
            Some(lin) => {
                // Residual path keeps the delta bf16 (PARITY-BF16) like the core LoKr install; the
                // forward casts it to the activation dtype.
                let delta = file.delta(factors, &lin.base_shape(), Dtype::Bfloat16)?;
                lin.push_lokr(delta, strengths.clone());
                report.applied += 1;
            }
            None => report.skipped.push(path),
        }
    }
    Ok(())
}

/// Install every adapter in `specs` onto the LTX transformer, stacking in order (sc-2687 LoRA /
/// sc-2393 LoKr). `num_passes` is the distilled pipeline's denoise-pass count (for validating +
/// expanding `pass_scales`). LoRA (PEFT/kohya) and LoKr (`networkType=lokr`) are dispatched by the
/// file's metadata / the spec kind. Errors only if a non-empty spec list matched no target module
/// (a format/prefix misconfiguration); per-key skips are reported, not fatal.
pub fn apply_ltx_adapters(
    host: &mut impl LtxAdaptable,
    specs: &[AdapterSpec],
    num_passes: usize,
) -> Result<LtxLoraReport> {
    let mut report = LtxLoraReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        // The file's metadata is authoritative; the spec kind is an additional hint. A spec that
        // declares Lora but whose file says `networkType=lokr` is a caller error (the LoRA loader
        // would find no `lora_A/B` and apply nothing) — route by the file so it is never mis-applied.
        if spec.kind == AdapterKind::Lokr || is_lokr(&w) {
            apply_one_lokr(host, &w, spec, num_passes, &mut report)?;
        } else {
            apply_one(host, &w, spec, num_passes, &mut report)?;
        }
    }
    if !specs.is_empty() && report.applied == 0 {
        return Err(Error::Msg(format!(
            "ltx_2_3 adapters: no target modules matched across {} file(s) — check the format \
             (expected PEFT `lora_A/B` or kohya `lora_down/up`, or LoKr `lokr_w1/w2`, with \
             `diffusion_model.` / `transformer_blocks.*` naming)",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_prefix_and_renames_to_out_and_ff() {
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.attn1.to_out.0"),
            "transformer_blocks.0.attn1.to_out"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.5.ff.net.0.proj"),
            "transformer_blocks.5.ff.proj_in"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.5.ff.net.2"),
            "transformer_blocks.5.ff.proj_out"
        );
        // gate + q/k/v pass through unchanged (already checkpoint naming).
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.attn2.to_gate_logits"),
            "transformer_blocks.0.attn2.to_gate_logits"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.adaln_single.linear"),
            "adaln_single.linear"
        );
    }

    #[test]
    fn ic_lora_union_control_keys_map_to_av_blocks() {
        // The production LTX-2.3 IC-LoRA (Lightricks LTX-2.3-22b-IC-LoRA-Union-Control, used for
        // extend_clip / video_bridge / replace_person) ships 960 PEFT bf16 tensors named
        // `diffusion_model.transformer_blocks.N.{attn1,attn2}.to_{q,k,v}.lora_{A,B}.weight`,
        // `...to_out.0...`, and `...ff.net.{0.proj,2}...`. Confirmed against the real file: every key
        // strips to a (suffix, role) and normalizes to an AvDiT video-block module path — so the
        // IC-LoRA loads via the existing `apply_ltx_adapters` seam with no new code (epic 3040).
        let cases = [
            (
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
                "transformer_blocks.0.attn1.to_q",
            ),
            (
                "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight",
                "transformer_blocks.0.attn1.to_out",
            ),
            (
                "diffusion_model.transformer_blocks.27.attn2.to_k.lora_A.weight",
                "transformer_blocks.27.attn2.to_k",
            ),
            (
                "diffusion_model.transformer_blocks.27.ff.net.0.proj.lora_A.weight",
                "transformer_blocks.27.ff.proj_in",
            ),
            (
                "diffusion_model.transformer_blocks.27.ff.net.2.lora_B.weight",
                "transformer_blocks.27.ff.proj_out",
            ),
        ];
        for (key, want) in cases {
            let stem = SUFFIXES
                .iter()
                .find_map(|(suf, _)| key.strip_suffix(suf))
                .unwrap_or_else(|| panic!("no LoRA suffix matched {key}"));
            assert_eq!(normalize_ltx_key(stem), want, "key {key}");
        }
    }

    #[test]
    fn normalize_other_prefixes_and_audio_analogues() {
        assert_eq!(
            normalize_ltx_key("model.diffusion_model.transformer_blocks.0.attn1.to_q"),
            "transformer_blocks.0.attn1.to_q"
        );
        // `.ff.net.*` must NOT fire inside `audio_ff.net.*`; the audio rename handles it separately.
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.audio_ff.net.0.proj"),
            "transformer_blocks.0.audio_ff.proj_in"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.audio_ff.net.2"),
            "transformer_blocks.0.audio_ff.proj_out"
        );
    }

    #[test]
    fn pass_scales_uniform_and_per_pass() {
        let mut spec = AdapterSpec::new("x.safetensors".into(), 0.5, AdapterKind::Lora);
        // Uniform: one entry = (alpha/rank)·scale = (16/8)·0.5 = 1.0.
        let u = pass_scales(&spec, 16.0, 8.0, 2).unwrap();
        assert_eq!(u, vec![1.0]);
        // Per-pass: (16/8)·[0.5, 0.25] = [1.0, 0.5].
        spec.pass_scales = Some(vec![0.5, 0.25]);
        assert_eq!(pass_scales(&spec, 16.0, 8.0, 2).unwrap(), vec![1.0, 0.5]);
        // Wrong length errors.
        spec.pass_scales = Some(vec![0.5]);
        assert!(pass_scales(&spec, 16.0, 8.0, 2).is_err());
    }
}
