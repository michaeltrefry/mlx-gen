//! Native weight converter (sc-4813): the `numz/SeedVR2_comfyUI` checkpoints
//! (`seedvr2_ema_{3b,7b}_fp16.safetensors`, `ema_vae_fp16.safetensors`) → the MLX-native key/layout
//! the Rust modules load. Port of the mflux `SeedVR2WeightMapping`; no Python.
//!
//! - **VAE:** keys are unchanged; conv weights are torch `[out,in,kT,kH,kW]` → MLX `[out,kT,kH,kW,in]`
//!   (any 5-D weight). Everything else passes through.
//! - **DiT:** rename the dotted attention/ada submodules (`attn.proj_qkv.vid` → `attn.proj_qkv_vid`,
//!   `ada.vid` → `ada.params_vid`, `attn.rope.rope.freqs` → `attn.rope.freqs`, …). Shared layers store
//!   the attention projections once under `.all`; they are duplicated into both `_vid` and `_txt`
//!   (the attention always uses separate projections). MLP keys pass through.

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

/// Convert the raw VAE checkpoint: transpose every 5-D conv weight to channels-last.
pub fn convert_vae(raw: &Weights) -> Result<Weights> {
    let mut out = Weights::empty();
    let keys: Vec<String> = raw.keys().map(String::from).collect();
    for k in keys {
        let v = raw.require(&k)?;
        let nv = if v.ndim() == 5 {
            v.transpose_axes(&[0, 2, 3, 4, 1])?
        } else {
            v.clone()
        };
        out.insert(k, nv);
    }
    Ok(out)
}

const ATTN_SUBS: [&str; 4] = ["proj_qkv", "proj_out", "norm_q", "norm_k"];

/// Map a raw DiT key to its converted target name(s) (two for a shared-layer attention `.all`).
fn dit_targets(k: &str) -> Vec<String> {
    // output AdaLN scale/shift live under `vid_out_ada.` in the checkpoint, flat in the model.
    if let Some(rest) = k.strip_prefix("vid_out_ada.") {
        return vec![rest.to_string()];
    }
    if k.contains(".attn.rope.rope.freqs") {
        return vec![k.replace(".attn.rope.rope.", ".attn.rope.")];
    }
    for sub in ATTN_SUBS {
        let all = format!(".attn.{sub}.all.");
        if k.contains(&all) {
            return vec![
                k.replace(&all, &format!(".attn.{sub}_vid.")),
                k.replace(&all, &format!(".attn.{sub}_txt.")),
            ];
        }
        for stream in ["vid", "txt"] {
            let from = format!(".attn.{sub}.{stream}.");
            if k.contains(&from) {
                return vec![k.replace(&from, &format!(".attn.{sub}_{stream}."))];
            }
        }
    }
    for stream in ["vid", "txt", "all"] {
        let from = format!(".ada.{stream}.");
        if k.contains(&from) {
            return vec![k.replace(&from, &format!(".ada.params_{stream}."))];
        }
    }
    vec![k.to_string()] // mlp.* and top-level pass through
}

/// Convert the raw DiT checkpoint (key renames; no transposes — all weights are 2-D).
pub fn convert_dit(raw: &Weights) -> Result<Weights> {
    let mut out = Weights::empty();
    let keys: Vec<String> = raw.keys().map(String::from).collect();
    for k in keys {
        let v = raw.require(&k)?;
        for target in dit_targets(&k) {
            out.insert(target, v.clone());
        }
    }
    Ok(out)
}

/// Convert both raw files in `src_dir` and write `vae.safetensors` + `transformer.safetensors`
/// (+ copy `neg_embed.safetensors` if present) into `out_dir`. `dit_file` selects 3B/7B.
pub fn convert_to_dir(
    src_dir: impl AsRef<std::path::Path>,
    dit_file: &str,
    out_dir: impl AsRef<std::path::Path>,
) -> Result<()> {
    let src = src_dir.as_ref();
    let out = out_dir.as_ref();
    std::fs::create_dir_all(out)?;

    let vae = convert_vae(&Weights::from_file(src.join("ema_vae_fp16.safetensors"))?)?;
    save_weights(&vae, &out.join("vae.safetensors"))?;
    let dit = convert_dit(&Weights::from_file(src.join(dit_file))?)?;
    save_weights(&dit, &out.join("transformer.safetensors"))?;
    Ok(())
}

fn save_weights(w: &Weights, path: &std::path::Path) -> Result<()> {
    let map: HashMap<String, Array> = w
        .keys()
        .map(|k| (k.to_string(), w.get(k).unwrap().clone()))
        .collect();
    let arrays: Vec<&Array> = map.values().collect();
    eval(arrays)?;
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}
