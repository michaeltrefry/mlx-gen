//! Native (Rust/MLX) Wan2.2 weight converter (sc-3224). Replaces the Python `mlx_video.convert_wan`.
//!
//! Wan native checkpoints ship the transformer as safetensors but the T5 encoder and VAE as torch
//! `.pth` (zip-of-pickle) — read via [`crate::pth`]. This module ports the reference sanitizers that
//! map the native key layout onto the MLX model layout the Wan loaders consume.
//!
//! **sc-3237: the Wan2.2 VAE path.** [`convert_vae22`] reads `Wan2.2_VAE.pth`, applies
//! [`sanitize_wan22_vae`] (the reference `sanitize_wan22_vae_weights`), and writes
//! `vae.safetensors` in f32 (official Wan runs VAE decode in float32).
//!
//! **sc-3238: the TI2V-5B single-model converter.** [`convert_ti2v_5b`] assembles a full MLX dir —
//! the transformer (native safetensors shards → [`sanitize_wan_transformer`], bf16), the T5
//! (`.pth` → [`sanitize_wan_t5`], bf16), the VAE (f32), and `config.json`.
//!
//! **sc-3239: the I2V-A14B dual-expert converter.** [`convert_i2v_14b`] converts both the
//! `low_noise_model` + `high_noise_model` experts (in_dim 36, optionally Q4/Q8 via
//! [`quantize_wan_transformer`]), the z16 Wan2.1 VAE ([`sanitize_wan_vae_weights`]), the T5, and the
//! I2V-14B `config.json`. ⚠ No golden dir exists for I2V-14B and its native source is not cached
//! (~114 GB), so this path is **structurally** validated (sanitizers + config round-trip + the
//! byte-proven [`sanitize_wan_transformer`]/pickle-reader/`quantize`) but not yet byte-parity'd
//! end-to-end — `tests/convert_i2v_14b_parity.rs` is wired and `#[ignore]`d, ready for when a
//! reference is generated.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Channels-last transpose of a PyTorch conv weight: Conv3d `[O,I,D,H,W]→[O,D,H,W,I]`, Conv2d
/// `[O,I,H,W]→[O,H,W,I]`. Other ranks pass through.
fn conv_channels_last(v: &Array) -> Result<Array> {
    match v.ndim() {
        5 => Ok(v.transpose_axes(&[0, 2, 3, 4, 1])?),
        4 => Ok(v.transpose_axes(&[0, 2, 3, 1])?),
        _ => Ok(v.clone()),
    }
}

/// Drop every size-1 axis (`np.squeeze`) — for the RMS_norm `gamma` tensors `(dim,1,1,1)`/`(dim,1,1)`
/// → `(dim,)`.
fn squeeze_all(v: &Array) -> Result<Array> {
    let new_shape: Vec<i32> = v.shape().iter().copied().filter(|&d| d != 1).collect();
    Ok(v.reshape(&new_shape)?)
}

/// Port of `sanitize_wan22_vae_weights` (mlx_video/models/wan/vae22.py): map the native Wan2.2 VAE
/// key layout (PyTorch `nn.Sequential` indices, channels-first convs, 4-D RMS gammas) onto the MLX
/// `WanVae22` layout. With `include_encoder=false` the encoder + `conv1.*` are dropped (decode-only);
/// TI2V/I2V keep them. Conv weights → channels-last; `gamma` → squeezed.
pub fn sanitize_wan22_vae(
    raw: &HashMap<String, Array>,
    include_encoder: bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for (k, src) in raw {
        if !include_encoder && (k.starts_with("encoder.") || k.starts_with("conv1.")) {
            continue;
        }

        // Sequential index → named layer: residual.{0,2,3,6} and head.{0,2}.
        let mut new = k.clone();
        for idx in ["0", "2", "3", "6"] {
            new = new.replace(
                &format!(".residual.{idx}."),
                &format!(".residual.layer_{idx}."),
            );
        }
        for idx in ["0", "2"] {
            new = new.replace(&format!(".head.{idx}."), &format!(".head.layer_{idx}."));
        }
        // Resample Conv2d + AttentionBlock Conv2d renames (first match wins, mirroring the if/elif).
        if new.contains(".resample.1.weight") {
            new = new.replace(".resample.1.weight", ".resample_weight");
        } else if new.contains(".resample.1.bias") {
            new = new.replace(".resample.1.bias", ".resample_bias");
        }
        if new.contains(".to_qkv.weight") {
            new = new.replace(".to_qkv.weight", ".to_qkv_weight");
        } else if new.contains(".to_qkv.bias") {
            new = new.replace(".to_qkv.bias", ".to_qkv_bias");
        } else if new.contains(".proj.weight") && !new.contains("time_projection") {
            new = new.replace(".proj.weight", ".proj_weight");
        } else if new.contains(".proj.bias") && !new.contains("time_projection") {
            new = new.replace(".proj.bias", ".proj_bias");
        }

        // Conv-weight channels-last (keys ending `.weight` OR the renamed `_weight`).
        let mut value = if new.ends_with(".weight") || new.ends_with("_weight") {
            conv_channels_last(src)?
        } else {
            src.clone()
        };
        // RMS_norm gamma: squeeze trailing singleton dims.
        if new.contains("gamma") {
            value = squeeze_all(&value)?;
        }
        out.insert(new, value);
    }
    Ok(out)
}

/// Convert a Wan2.2 `Wan2.2_VAE.pth` into `out_file` (`vae.safetensors`), f32. `include_encoder` is
/// `true` for TI2V/I2V (encode path needed), `false` for decode-only T2V.
pub fn convert_vae22(
    vae_pth: impl AsRef<Path>,
    out_file: impl AsRef<Path>,
    include_encoder: bool,
) -> Result<()> {
    let vae_pth = vae_pth.as_ref();
    if !vae_pth.is_file() {
        return Err(Error::Msg(format!(
            "Wan VAE .pth not found: {}",
            vae_pth.display()
        )));
    }
    // Load the native .pth as f32 (mirrors torch.load(...).float()), then sanitize.
    let raw = crate::pth::load_pth_f32(vae_pth)?;
    let sanitized = sanitize_wan22_vae(&raw, include_encoder)?;

    let arrays: Vec<&Array> = sanitized.values().collect();
    eval(arrays)?;
    if let Some(parent) = out_file.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        sanitized.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        out_file.as_ref(),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// sc-3238: Wan2.2 TI2V-5B full converter (transformer + T5 + config + orchestration)
// ---------------------------------------------------------------------------

/// Collect a [`Weights`] into a plain key→Array map (lazy clones).
fn weights_to_map(w: &Weights) -> HashMap<String, Array> {
    w.keys()
        .map(|k| {
            (
                k.to_string(),
                w.require(k).expect("key from keys()").clone(),
            )
        })
        .collect()
}

/// Cast every tensor in `map` to `dtype` in place.
fn cast_map(map: &mut HashMap<String, Array>, dtype: Dtype) -> Result<()> {
    for v in map.values_mut() {
        if v.dtype() != dtype {
            *v = v.as_dtype(dtype)?;
        }
    }
    Ok(())
}

/// Materialize + write a key→Array map to `path`.
fn save_map(path: PathBuf, map: &HashMap<String, Array>) -> Result<()> {
    let arrays: Vec<&Array> = map.values().collect();
    eval(arrays)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

fn write_json(path: PathBuf, v: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(v)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(&path, text)?;
    Ok(())
}

/// Port of `sanitize_wan_transformer_weights`: native Wan transformer keys → MLX `WanTransformer`
/// keys. `patch_embedding.weight` `[dim,in,1,2,2]` flattens to `[dim, in·4]` → `patch_embedding_proj`;
/// the `text/time_embedding` Sequentials (`.0`/`.2`) → `_0`/`_1`; `time_projection.1` → bare;
/// `ffn.0`/`ffn.2` → `ffn.fc1`/`ffn.fc2`; the `freqs` buffer is dropped. Everything else (attn, norms,
/// modulation, head) passes through unchanged.
pub fn sanitize_wan_transformer(raw: &HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for (key, value) in raw {
        if key == "patch_embedding.weight" {
            let s = value.shape();
            let cols: i32 = s[1..].iter().product();
            out.insert(
                "patch_embedding_proj.weight".into(),
                value.reshape(&[s[0], cols])?,
            );
            continue;
        }
        if key == "patch_embedding.bias" {
            out.insert("patch_embedding_proj.bias".into(), value.clone());
            continue;
        }
        let renamed_seq = [
            ("text_embedding.0.", "text_embedding_0."),
            ("text_embedding.2.", "text_embedding_1."),
            ("time_embedding.0.", "time_embedding_0."),
            ("time_embedding.2.", "time_embedding_1."),
            ("time_projection.1.", "time_projection."),
        ]
        .iter()
        .find_map(|(p, r)| key.strip_prefix(p).map(|rest| format!("{r}{rest}")));
        if let Some(new) = renamed_seq {
            out.insert(new, value.clone());
            continue;
        }
        if key == "freqs" {
            continue;
        }
        let new = key
            .replace(".ffn.0.", ".ffn.fc1.")
            .replace(".ffn.2.", ".ffn.fc2.");
        out.insert(new, value.clone());
    }
    Ok(out)
}

/// Port of `sanitize_wan_t5_weights`: the sole rename `.ffn.gate.0.` → `.ffn.gate_proj.` (the gate
/// Linear); every other UMT5 key passes through.
pub fn sanitize_wan_t5(raw: &HashMap<String, Array>) -> HashMap<String, Array> {
    raw.iter()
        .map(|(k, v)| (k.replace(".ffn.gate.0.", ".ffn.gate_proj."), v.clone()))
        .collect()
}

/// The `WanModelConfig.wan22_ti2v_5b().to_dict()` config.json (matches the golden semantically).
fn wan22_ti2v_5b_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "ti2v",
        "model_version": "2.2",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 48,
        "dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "num_layers": 30,
        "window_size": [-1, -1],
        "qk_norm": true,
        "cross_attn_norm": true,
        "eps": 1e-6,
        "vae_stride": [4, 16, 16],
        "vae_z_dim": 48,
        "dual_model": false,
        "boundary": 0.0,
        "sample_shift": 5.0,
        "sample_steps": 40,
        "sample_guide_scale": 5.0,
        "num_train_timesteps": 1000,
        "sample_fps": 24,
        "frame_num": 81,
        "sample_neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "max_area": 901120,
        "t5_vocab_size": 256384,
        "t5_dim": 4096,
        "t5_dim_attn": 4096,
        "t5_dim_ffn": 10240,
        "t5_num_heads": 64,
        "t5_num_layers": 24,
        "t5_num_buckets": 32
    })
}

/// Convert a native Wan2.2 **TI2V-5B** checkpoint dir into an MLX model dir at `out_dir`: the
/// transformer (native `diffusion_pytorch_model-*.safetensors` shards) → `model.safetensors` (bf16),
/// the T5 (`models_t5_umt5-xxl-enc-bf16.pth`) → `t5_encoder.safetensors` (bf16), the VAE
/// (`Wan2.2_VAE.pth`) → `vae.safetensors` (f32), plus `config.json`. The UMT5 `tokenizer.json` (at
/// `google/umt5-xxl/tokenizer.json` in the native repo) is copied by the install flow, not emitted
/// here — matching the reference `convert_wan`.
pub fn convert_ti2v_5b(
    checkpoint_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let checkpoint_dir = checkpoint_dir.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    // 1. Transformer — native single-model safetensors (the 3 shards merge in `from_dir`).
    let w = Weights::from_dir(checkpoint_dir)?;
    let map = weights_to_map(&w);
    let mut transformer = sanitize_wan_transformer(&map)?;
    cast_map(&mut transformer, Dtype::Bfloat16)?;
    save_map(out_dir.join("model.safetensors"), &transformer)?;
    drop((w, map, transformer));

    // 2. Config.
    write_json(out_dir.join("config.json"), &wan22_ti2v_5b_config())?;

    // 3. T5 encoder — native `.pth` (pickle) → f32 → sanitize → bf16.
    let t5_pth = checkpoint_dir.join("models_t5_umt5-xxl-enc-bf16.pth");
    let raw_t5 = crate::pth::load_pth_f32(&t5_pth)?;
    let mut t5 = sanitize_wan_t5(&raw_t5);
    cast_map(&mut t5, Dtype::Bfloat16)?;
    save_map(out_dir.join("t5_encoder.safetensors"), &t5)?;
    drop((raw_t5, t5));

    // 4. VAE — TI2V keeps the encoder; saved f32.
    convert_vae22(
        checkpoint_dir.join("Wan2.2_VAE.pth"),
        out_dir.join("vae.safetensors"),
        true,
    )?;

    Ok(out_dir.to_path_buf())
}

// ---------------------------------------------------------------------------
// sc-3239: Wan2.2 I2V-A14B dual-expert converter (in_dim 36, z16 VAE, optional Q4/Q8)
// ---------------------------------------------------------------------------

/// The reference `_quantize_predicate`: a Wan transformer Linear is quantized iff its weight key
/// (minus `.weight`) ends with one of these — attention q/k/v/o (self + cross) and the FFN fc1/fc2.
/// Norms / modulation / embeddings / head stay dense.
const WAN_QUANT_SUFFIXES: &[&str] = &[
    ".self_attn.q",
    ".self_attn.k",
    ".self_attn.v",
    ".self_attn.o",
    ".cross_attn.q",
    ".cross_attn.k",
    ".cross_attn.v",
    ".cross_attn.o",
    ".ffn.fc1",
    ".ffn.fc2",
];

/// Port of `sanitize_wan_vae_weights` (the Wan2.1 z16 VAE — `convert_wan.py`): channels-last conv
/// transposes only (Conv3d/Conv2d weights gated on `"weight" in key`), **no** key renames. Distinct
/// from the bespoke z48 [`sanitize_wan22_vae`].
pub fn sanitize_wan_vae_weights(raw: &HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let value = if k.contains("weight") {
            conv_channels_last(v)?
        } else {
            v.clone()
        };
        out.insert(k.clone(), value);
    }
    Ok(out)
}

/// Selectively Q4/Q8-quantize a (sanitized) Wan transformer expert in place: each
/// [`WAN_QUANT_SUFFIXES`]-matched Linear `{base}.weight` (bf16) becomes the packed triple
/// `{base}.weight` (u32), `{base}.scales`, `{base}.biases` via MLX `quantize` (byte-identical to
/// `nn.quantize`); the bias and all other tensors pass through.
pub fn quantize_wan_transformer(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight");
        let is_q = base.is_some_and(|b| WAN_QUANT_SUFFIXES.iter().any(|s| b.ends_with(s)));
        if let (true, Some(base)) = (is_q, base) {
            let (wq, scales, biases) = quantize(&v, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// The `WanModelConfig.wan22_i2v_14b().to_dict()` config.json (round-trips through
/// `WanModelConfig::from_config_json`; the dual guide scale is a 2-element array).
fn wan22_i2v_14b_config(quantize: Option<(i32, i32)>) -> serde_json::Value {
    let mut cfg = serde_json::json!({
        "model_type": "i2v",
        "model_version": "2.2",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 36,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "window_size": [-1, -1],
        "qk_norm": true,
        "cross_attn_norm": true,
        "eps": 1e-6,
        "vae_stride": [4, 8, 8],
        "vae_z_dim": 16,
        "dual_model": true,
        "boundary": 0.9,
        "sample_shift": 5.0,
        "sample_steps": 40,
        "sample_guide_scale": [3.5, 3.5],
        "num_train_timesteps": 1000,
        "sample_fps": 16,
        "frame_num": 81,
        "sample_neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "max_area": 901120,
        "t5_vocab_size": 256384,
        "t5_dim": 4096,
        "t5_dim_attn": 4096,
        "t5_dim_ffn": 10240,
        "t5_num_heads": 64,
        "t5_num_layers": 24,
        "t5_num_buckets": 32
    });
    if let Some((bits, group_size)) = quantize {
        cfg["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    }
    cfg
}

/// Convert one native transformer expert dir (`low_noise_model` / `high_noise_model`) → a sanitized,
/// bf16 (optionally quantized) component file.
fn convert_expert(
    expert_dir: &Path,
    out_file: PathBuf,
    quantize: Option<(i32, i32)>,
) -> Result<()> {
    let w = Weights::from_dir(expert_dir)?;
    let map = weights_to_map(&w);
    let mut t = sanitize_wan_transformer(&map)?;
    cast_map(&mut t, Dtype::Bfloat16)?;
    let t = match quantize {
        Some((bits, group)) => quantize_wan_transformer(t, bits, group)?,
        None => t,
    };
    save_map(out_file, &t)?;
    Ok(())
}

/// Convert a native Wan2.2 **I2V-A14B** checkpoint dir into an MLX model dir: the two MoE experts
/// (`low_noise_model` / `high_noise_model` → `*.safetensors`, in_dim 36, optionally Q4/Q8), the z16
/// Wan2.1 VAE (`Wan2.1_VAE.pth`, falling back to `Wan2.2_VAE.pth`), the T5, and `config.json`.
/// `quantize = Some((bits, group_size))` enables selective transformer quantization on both experts.
///
/// ⚠ Unlike [`convert_ti2v_5b`] this path has no golden + the native source (~114 GB) is uncached, so
/// it is validated structurally (sanitizers + config round-trip), not yet byte-parity'd end-to-end.
pub fn convert_i2v_14b(
    checkpoint_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
) -> Result<PathBuf> {
    let checkpoint_dir = checkpoint_dir.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    // 1. Dual experts.
    for (sub, out) in [
        ("low_noise_model", "low_noise_model.safetensors"),
        ("high_noise_model", "high_noise_model.safetensors"),
    ] {
        let expert_dir = checkpoint_dir.join(sub);
        if !expert_dir.is_dir() {
            return Err(Error::Msg(format!(
                "missing expert dir: {}",
                expert_dir.display()
            )));
        }
        convert_expert(&expert_dir, out_dir.join(out), quantize)?;
    }

    // 2. Config.
    write_json(out_dir.join("config.json"), &wan22_i2v_14b_config(quantize))?;

    // 3. T5 encoder.
    let t5_pth = checkpoint_dir.join("models_t5_umt5-xxl-enc-bf16.pth");
    let raw_t5 = crate::pth::load_pth_f32(&t5_pth)?;
    let mut t5 = sanitize_wan_t5(&raw_t5);
    cast_map(&mut t5, Dtype::Bfloat16)?;
    save_map(out_dir.join("t5_encoder.safetensors"), &t5)?;
    drop((raw_t5, t5));

    // 4. VAE — prefer the z16 Wan2.1 VAE; fall back to the z48 Wan2.2 VAE (encoder kept for i2v).
    let vae21 = checkpoint_dir.join("Wan2.1_VAE.pth");
    let vae22 = checkpoint_dir.join("Wan2.2_VAE.pth");
    if vae21.is_file() {
        let raw = crate::pth::load_pth_f32(&vae21)?;
        let sanitized = sanitize_wan_vae_weights(&raw)?;
        save_map(out_dir.join("vae.safetensors"), &sanitized)?;
    } else if vae22.is_file() {
        convert_vae22(&vae22, out_dir.join("vae.safetensors"), true)?;
    } else {
        return Err(Error::Msg(format!(
            "no VAE (.pth) found in {} — provide Wan2.1_VAE.pth or Wan2.2_VAE.pth",
            checkpoint_dir.display()
        )));
    }

    Ok(out_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::all_close;

    fn exact_eq(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape() && all_close(a, b, 0.0, 0.0, false).unwrap().item::<bool>()
    }

    fn m(entries: &[(&str, Array)]) -> HashMap<String, Array> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// Key renames: Sequential index → layer_N, resample/to_qkv/proj conv renames.
    #[test]
    fn vae_key_renames() {
        let ones5 = Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(); // conv3d weight
        let s = sanitize_wan22_vae(
            &m(&[
                ("decoder.middle.0.residual.0.weight", ones5.clone()),
                (
                    "decoder.middle.0.residual.6.bias",
                    Array::ones::<f32>(&[2]).unwrap(),
                ),
                (
                    "decoder.head.0.gamma",
                    Array::ones::<f32>(&[4, 1, 1, 1]).unwrap(),
                ),
                ("decoder.head.2.weight", ones5.clone()),
                (
                    "decoder.upsamples.0.upsamples.0.resample.1.weight",
                    Array::ones::<f32>(&[2, 2, 3, 3]).unwrap(),
                ),
                (
                    "decoder.middle.0.to_qkv.weight",
                    Array::ones::<f32>(&[6, 2, 1, 1]).unwrap(),
                ),
                (
                    "decoder.middle.0.proj.bias",
                    Array::ones::<f32>(&[2]).unwrap(),
                ),
            ]),
            true,
        )
        .unwrap();
        let mut keys: Vec<&str> = s.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "decoder.head.layer_0.gamma",
                "decoder.head.layer_2.weight",
                "decoder.middle.0.proj_bias",
                "decoder.middle.0.residual.layer_0.weight",
                "decoder.middle.0.residual.layer_6.bias",
                "decoder.middle.0.to_qkv_weight",
                "decoder.upsamples.0.upsamples.0.resample_weight",
            ]
        );
    }

    /// `include_encoder=false` drops `encoder.*` and `conv1.*`; `true` keeps them.
    #[test]
    fn vae_encoder_gating() {
        let entries = [
            (
                "encoder.conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "conv2.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "decoder.conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
        ];
        let dec_only = sanitize_wan22_vae(&m(&entries), false).unwrap();
        assert!(!dec_only
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("conv1.")));
        assert!(dec_only.contains_key("conv2.weight"));
        assert!(dec_only.contains_key("decoder.conv1.weight")); // not a top-level conv1
        let with_enc = sanitize_wan22_vae(&m(&entries), true).unwrap();
        assert!(with_enc.contains_key("conv1.weight"));
        assert!(with_enc.contains_key("encoder.conv1.weight"));
    }

    /// Conv3d weight → channels-last; gamma squeezed; bias untouched.
    #[test]
    fn vae_transpose_and_squeeze() {
        // Conv3d [O=1,I=2,D=1,H=1,W=2] row-major 0..3 → [O,D,H,W,I]=[1,1,1,2,2] values [0,2,1,3].
        let v = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 1, 1, 2]);
        let s = sanitize_wan22_vae(
            &m(&[
                ("conv2.weight", v),
                (
                    "decoder.middle.0.norm.gamma",
                    Array::ones::<f32>(&[3, 1, 1, 1]).unwrap(),
                ),
            ]),
            true,
        )
        .unwrap();
        assert!(exact_eq(
            &s["conv2.weight"],
            &Array::from_slice(&[0.0f32, 2.0, 1.0, 3.0], &[1, 1, 1, 2, 2])
        ));
        assert_eq!(s["decoder.middle.0.norm.gamma"].shape(), &[3]);
    }

    /// Transformer sanitizer: patch_embedding flatten, Sequential→_0/_1, time_projection.1→bare,
    /// ffn.0/2→fc1/fc2, freqs dropped, attn/modulation pass-through.
    #[test]
    fn transformer_renames() {
        let s = sanitize_wan_transformer(&m(&[
            // [dim=2, in=3, 1, 2, 2] → patch_embedding_proj.weight [2, 12]
            (
                "patch_embedding.weight",
                Array::ones::<f32>(&[2, 3, 1, 2, 2]).unwrap(),
            ),
            ("patch_embedding.bias", Array::ones::<f32>(&[2]).unwrap()),
            (
                "text_embedding.0.weight",
                Array::ones::<f32>(&[2, 4]).unwrap(),
            ),
            ("text_embedding.2.bias", Array::ones::<f32>(&[2]).unwrap()),
            (
                "time_embedding.0.weight",
                Array::ones::<f32>(&[2, 4]).unwrap(),
            ),
            (
                "time_embedding.2.weight",
                Array::ones::<f32>(&[2, 2]).unwrap(),
            ),
            (
                "time_projection.1.weight",
                Array::ones::<f32>(&[12, 2]).unwrap(),
            ),
            (
                "blocks.0.ffn.0.weight",
                Array::ones::<f32>(&[8, 2]).unwrap(),
            ),
            (
                "blocks.0.ffn.2.weight",
                Array::ones::<f32>(&[2, 8]).unwrap(),
            ),
            (
                "blocks.0.self_attn.q.weight",
                Array::ones::<f32>(&[2, 2]).unwrap(),
            ),
            (
                "blocks.0.modulation",
                Array::ones::<f32>(&[1, 6, 2]).unwrap(),
            ),
            ("freqs", Array::ones::<f32>(&[2, 2]).unwrap()),
        ]))
        .unwrap();
        let mut keys: Vec<&str> = s.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "blocks.0.ffn.fc1.weight",
                "blocks.0.ffn.fc2.weight",
                "blocks.0.modulation",
                "blocks.0.self_attn.q.weight",
                "patch_embedding_proj.bias",
                "patch_embedding_proj.weight",
                "text_embedding_0.weight",
                "text_embedding_1.bias",
                "time_embedding_0.weight",
                "time_embedding_1.weight",
                "time_projection.weight",
            ]
        );
        assert_eq!(s["patch_embedding_proj.weight"].shape(), &[2, 12]); // 3·1·2·2 = 12
        assert!(!s.contains_key("freqs"));
    }

    /// T5 sanitizer: only `.ffn.gate.0.` → `.ffn.gate_proj.`; everything else unchanged.
    #[test]
    fn t5_gate_rename() {
        let s = sanitize_wan_t5(&m(&[
            (
                "blocks.0.ffn.gate.0.weight",
                Array::ones::<f32>(&[4, 2]).unwrap(),
            ),
            (
                "blocks.0.ffn.fc1.weight",
                Array::ones::<f32>(&[4, 2]).unwrap(),
            ),
            (
                "blocks.0.attn.q.weight",
                Array::ones::<f32>(&[2, 2]).unwrap(),
            ),
            (
                "token_embedding.weight",
                Array::ones::<f32>(&[5, 2]).unwrap(),
            ),
        ]));
        assert!(s.contains_key("blocks.0.ffn.gate_proj.weight"));
        assert!(!s.keys().any(|k| k.contains("gate.0")));
        assert!(s.contains_key("blocks.0.ffn.fc1.weight"));
        assert!(s.contains_key("blocks.0.attn.q.weight"));
        assert!(s.contains_key("token_embedding.weight"));
    }

    /// z16 VAE sanitizer (`sanitize_wan_vae_weights`): conv transpose only, no key renames.
    #[test]
    fn z16_vae_transpose_only() {
        // Conv2d [O=1,I=2,H=2,W=1] 0..3 → [O,H,W,I]=[1,2,1,2] values [0,2,1,3]; keys unchanged.
        let s = sanitize_wan_vae_weights(&m(&[
            (
                "decoder.conv1.weight",
                Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 2, 1]),
            ),
            (
                "decoder.middle.0.residual.0.bias",
                Array::ones::<f32>(&[3]).unwrap(),
            ),
        ]))
        .unwrap();
        // raw keys preserved (NOT renamed to layer_N like the z48 vae22 sanitizer)
        assert!(s.contains_key("decoder.conv1.weight"));
        assert!(s.contains_key("decoder.middle.0.residual.0.bias"));
        assert!(exact_eq(
            &s["decoder.conv1.weight"],
            &Array::from_slice(&[0.0f32, 2.0, 1.0, 3.0], &[1, 2, 1, 2])
        ));
    }

    /// Wan quant predicate: attn q/k/v/o (self + cross) + ffn fc1/fc2; norms/modulation/head dense.
    #[test]
    fn wan_quant_predicate() {
        let q = |k: &str| {
            k.strip_suffix(".weight")
                .is_some_and(|b| WAN_QUANT_SUFFIXES.iter().any(|s| b.ends_with(s)))
        };
        for k in [
            "blocks.0.self_attn.q.weight",
            "blocks.5.cross_attn.o.weight",
            "blocks.0.ffn.fc1.weight",
            "blocks.0.ffn.fc2.weight",
        ] {
            assert!(q(k), "should quantize: {k}");
        }
        for k in [
            "blocks.0.self_attn.q.bias",
            "blocks.0.self_attn.norm_q.weight",
            "blocks.0.modulation",
            "patch_embedding_proj.weight",
            "head.head.weight",
        ] {
            assert!(!q(k), "should stay dense: {k}");
        }
    }

    /// Quantizing a Wan transformer emits packed weight + scales/biases for matched Linears, keeps
    /// the bias, and leaves norms dense.
    #[test]
    fn quantize_wan_transformer_packs() {
        let bf = |a: Array| a.as_dtype(Dtype::Bfloat16).unwrap();
        let q = quantize_wan_transformer(
            m(&[
                (
                    "blocks.0.self_attn.q.weight",
                    bf(Array::ones::<f32>(&[64, 128]).unwrap()),
                ),
                (
                    "blocks.0.self_attn.q.bias",
                    bf(Array::ones::<f32>(&[64]).unwrap()),
                ),
                (
                    "blocks.0.norm1.weight",
                    bf(Array::ones::<f32>(&[64]).unwrap()),
                ),
            ]),
            4,
            64,
        )
        .unwrap();
        assert!(q.contains_key("blocks.0.self_attn.q.scales"));
        assert!(q.contains_key("blocks.0.self_attn.q.biases"));
        assert!(q.contains_key("blocks.0.self_attn.q.bias")); // bias preserved
        assert_ne!(q["blocks.0.self_attn.q.weight"].dtype(), Dtype::Bfloat16); // packed (u32)
        assert!(q.contains_key("blocks.0.norm1.weight")); // dense
        assert!(!q.contains_key("blocks.0.norm1.scales"));
    }

    /// The I2V-14B config.json round-trips through the loader's parser to the `wan22_i2v_14b` preset
    /// (no golden exists, so this is the validation oracle), with the quant block when requested.
    #[test]
    fn i2v_14b_config_round_trips() {
        use crate::config::WanModelConfig;
        let cfg = WanModelConfig::from_config_json(&wan22_i2v_14b_config(None));
        assert_eq!(cfg, WanModelConfig::wan22_i2v_14b());
        let cfgq = WanModelConfig::from_config_json(&wan22_i2v_14b_config(Some((4, 64))));
        assert!(cfgq.quantization.is_some());
    }
}
