//! Native (Rust/MLX) LTX-2.3 **single-file → split MLX** weight converter (sc-3233 / sc-3224).
//!
//! Community / fine-tuned LTX-2.3 checkpoints such as `TenStrip/LTX2.3-10Eros`
//! (`10Eros_v1_bf16.safetensors`) ship as a single flat `.safetensors` holding the whole stack —
//! transformer (`model.diffusion_model.*`, including the two embeddings connectors), the video VAE
//! (`vae.{encoder,decoder}.*`), the audio VAE (`audio_vae.*`), the vocoder (`vocoder.*`), and the
//! text-embedding projection (`text_embedding_projection.*`). The [`crate::model::load`] path,
//! however, consumes a **split** MLX model dir (one `.safetensors` per component, a
//! `split_model.json` manifest, `embedded_config.json`/`config.json`, and — for the shipped eros
//! checkpoint — a **Q4-quantized** transformer).
//!
//! [`convert_and_assemble`] reproduces, in Rust/MLX, the exact transforms the (slated-for-removal,
//! sc-3242) Python `mlx_video.convert` applied:
//!
//!   * **split by prefix** into per-component weight maps;
//!   * **sanitize keys** per component (the reference `sanitize_{transformer,vae,vae_encoder,
//!     audio_vae,vocoder}_weights`): `model.diffusion_model.` prefix strip, `.to_out.0.`→`.to_out.`,
//!     `.ff.net.0.proj.`→`.ff.proj_in.`, `.ff.net.2.`→`.ff.proj_out.` (+ audio), `.linear_1/2.`→
//!     `.linear1/2.`, and the VAE `per_channel_statistics` key remaps;
//!   * **channels-last conv transposes** — Conv3d `[O,I,D,H,W]→[O,D,H,W,I]`, Conv2d `[O,I,H,W]→
//!     [O,H,W,I]`, vocoder Conv1d `[O,I,K]→[O,K,I]` and `ups` ConvTranspose1d `[I,O,K]→[O,K,I]`;
//!   * **selective Q4 quantization of the transformer** — MLX `quantize` (group 64) over the same
//!     Linear set the reference `_quantize_ltx_predicate` selects (`*.to_q/k/v/out`, `*.ff.proj_in/
//!     out`, `*.audio_ff.proj_in/out`), emitting `.weight`(u32)/`.scales`/`.biases`;
//!   * **merge the latent upsampler(s)** from the base `Lightricks/LTX-2.3` repo (the eros repo
//!     ships none) as raw component copies;
//!   * **emit** `config.json`, `embedded_config.json`, `split_model.json`, `quantize_config.json`.
//!
//! ## The one load-bearing subtlety (validated by tensor count)
//! The reference `sanitize_transformer_weights` sweeps in the connector keys too (they live under
//! `model.diffusion_model.`), but the downstream `model.load_weights(strict=False)` +
//! `tree_flatten(model.parameters())` silently **drops** them from the transformer component — which
//! is why the golden transformer carries no connector keys. We reproduce that by excluding any key
//! containing `embeddings_connector` from the transformer component. This is exact: the golden eros
//! transformer is `4444 (model.diffusion_model.*) − 258 (embeddings_connector) = 4186` sanitized
//! tensors, of which `1344` Linear weights quantize to `.weight`/`.scales`/`.biases` → `4186 + 2·1344
//! = 6874` tensors, matching the golden byte-for-byte.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

/// The reference `_quantize_ltx_predicate`: a transformer Linear is Q4-quantized iff its weight key
/// (minus the `.weight` suffix) ends with one of these. Matches all attention `to_q/k/v/out` (self,
/// cross, and the cross-modal audio↔video attentions) plus the video/audio FFN in/out projections;
/// leaves `q_norm`/`k_norm`/`to_gate_logits`/adaLN/patchify/`proj_out` dense.
const QUANT_SUFFIXES: &[&str] = &[
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out",
    ".ff.proj_in",
    ".ff.proj_out",
    ".audio_ff.proj_in",
    ".audio_ff.proj_out",
];

/// The LTX-2.3 latent-upsampler component specs (component prefix → source filename in the base
/// `Lightricks/LTX-2.3` repo). Each is emitted only if the source file is present (the eros golden
/// carries just `upsampler` + `spatial_upscaler_x2_v1_1`). Raw copies — no key/shape transform.
const UPSCALER_SPECS_23: &[(&str, &str)] = &[
    ("upsampler", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    (
        "spatial_upscaler_x2_v1_1",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    ),
    (
        "spatial_upscaler_x1_5_v1_0",
        "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    ),
    (
        "temporal_upscaler_x2_v1_0",
        "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
    ),
];

/// Conversion knobs. Defaults mirror the shipped `ltx_2_3_eros` recipe (audio + Q4 group-64).
#[derive(Clone, Copy, Debug)]
pub struct LtxConvertOpts {
    /// Include the audio VAE + vocoder components (the AudioVideo path). `false` = video-only.
    pub include_audio: bool,
    /// Q4/Q8-quantize the transformer (the reference `--quantize`).
    pub quantize: bool,
    /// Quantization bits (4 → Q4, 8 → Q8).
    pub bits: i32,
    /// Quantization group size (the affine-quant group width; reference default 64).
    pub group_size: i32,
}

impl Default for LtxConvertOpts {
    fn default() -> Self {
        LtxConvertOpts {
            include_audio: true,
            quantize: false,
            bits: 4,
            group_size: 64,
        }
    }
}

impl LtxConvertOpts {
    /// The `ltx_2_3_eros` recipe: audio + Q4 (group 64) — `python -m mlx_video.convert --quantize
    /// --q-bits 4`.
    pub fn eros_q4() -> Self {
        LtxConvertOpts {
            include_audio: true,
            quantize: true,
            bits: 4,
            group_size: 64,
        }
    }
}

/// Channels-last transpose of a PyTorch conv weight, gated exactly as the reference (`"conv" in
/// key.lower() and "weight" in key`): Conv3d `[O,I,D,H,W]→[O,D,H,W,I]`, Conv2d `[O,I,H,W]→[O,H,W,I]`.
/// Other tensors pass through untouched.
fn conv_channels_last(key: &str, v: &Array, allow_3d: bool, allow_2d: bool) -> Result<Array> {
    if !(key.to_ascii_lowercase().contains("conv") && key.contains("weight")) {
        return Ok(v.clone());
    }
    match v.ndim() {
        5 if allow_3d => Ok(v.transpose_axes(&[0, 2, 3, 4, 1])?),
        4 if allow_2d => Ok(v.transpose_axes(&[0, 2, 3, 1])?),
        _ => Ok(v.clone()),
    }
}

/// `sanitize_transformer_weights`: keep `model.diffusion_model.*` EXCEPT the embeddings connectors
/// (dropped downstream — see module docs), strip the prefix, apply the FFN / `to_out` / adaLN-linear
/// renames. No conv transposes (the transformer has no convs).
fn sanitize_transformer(raw: &Weights) -> HashMap<String, Array> {
    let mut out = HashMap::new();
    for k in raw.keys() {
        let Some(rest) = k.strip_prefix("model.diffusion_model.") else {
            continue;
        };
        if k.contains("embeddings_connector") {
            continue;
        }
        let new = rest
            .replace(".to_out.0.", ".to_out.")
            .replace(".ff.net.0.proj.", ".ff.proj_in.")
            .replace(".ff.net.2.", ".ff.proj_out.")
            .replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
            .replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
            .replace(".linear_1.", ".linear1.")
            .replace(".linear_2.", ".linear2.");
        out.insert(new, raw.require(k).expect("key from keys()").clone());
    }
    out
}

/// `sanitize_vae_weights` (decoder) — `vae.decoder.*` (prefix stripped) + the two
/// `per_channel_statistics` stats (`mean-of-means`→`mean`, `std-of-means`→`std`); conv3d/2d
/// channels-last. `position_ids` and all other `vae.*` keys are dropped.
fn sanitize_vae_decoder(raw: &Weights) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for k in raw.keys() {
        if k.contains("position_ids") || !k.starts_with("vae.") {
            continue;
        }
        let new = if k.starts_with("vae.per_channel_statistics") {
            match k {
                "vae.per_channel_statistics.mean-of-means" => {
                    "per_channel_statistics.mean".to_string()
                }
                "vae.per_channel_statistics.std-of-means" => {
                    "per_channel_statistics.std".to_string()
                }
                _ => continue,
            }
        } else if let Some(rest) = k.strip_prefix("vae.decoder.") {
            rest.to_string()
        } else {
            continue;
        };
        let v = conv_channels_last(&new, raw.require(k)?, true, true)?;
        out.insert(new, v);
    }
    Ok(out)
}

/// `sanitize_vae_encoder_weights` — `vae.encoder.*` (prefix stripped) + the two stats remapped to the
/// encoder's `_mean_of_means`/`_std_of_means`; conv3d/2d channels-last.
fn sanitize_vae_encoder(raw: &Weights) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for k in raw.keys() {
        if k.contains("position_ids") || !k.starts_with("vae.") {
            continue;
        }
        let new = if k.starts_with("vae.per_channel_statistics") {
            match k {
                "vae.per_channel_statistics.mean-of-means" => {
                    "per_channel_statistics._mean_of_means".to_string()
                }
                "vae.per_channel_statistics.std-of-means" => {
                    "per_channel_statistics._std_of_means".to_string()
                }
                _ => continue,
            }
        } else if let Some(rest) = k.strip_prefix("vae.encoder.") {
            rest.to_string()
        } else {
            continue;
        };
        let v = conv_channels_last(&new, raw.require(k)?, true, true)?;
        out.insert(new, v);
    }
    Ok(out)
}

/// `sanitize_audio_vae_weights` — `audio_vae.decoder.*` (prefix stripped) + the two stats remapped to
/// `_mean_of_means`/`_std_of_means`; Conv2d channels-last only (the audio VAE is 2-D). Non-decoder
/// `audio_vae.*` keys (e.g. the encoder) are dropped.
fn sanitize_audio_vae(raw: &Weights) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for k in raw.keys() {
        let new = if let Some(rest) = k.strip_prefix("audio_vae.decoder.") {
            rest.to_string()
        } else if k.starts_with("audio_vae.per_channel_statistics.") {
            if k.contains("mean-of-means") {
                "per_channel_statistics._mean_of_means".to_string()
            } else if k.contains("std-of-means") {
                "per_channel_statistics._std_of_means".to_string()
            } else {
                continue;
            }
        } else {
            continue;
        };
        let v = conv_channels_last(&new, raw.require(k)?, false, true)?;
        out.insert(new, v);
    }
    Ok(out)
}

/// `sanitize_vocoder_weights` — keep every `vocoder.*` key, with **every** `vocoder.` occurrence
/// removed (the reference's `key.replace("vocoder.", "")`, NOT a single prefix strip: the core
/// generator ships double-nested as `vocoder.vocoder.*` → `*`, while `vocoder.bwe_generator.*` →
/// `bwe_generator.*`). Transpose ndim-3 weights (gated on `"weight"`, NOT `"conv"`): `ups`
/// ConvTranspose1d `[I,O,K]→[O,K,I]` via `(1,2,0)`, all other Conv1d `[O,I,K]→[O,K,I]` via `(0,2,1)`.
fn sanitize_vocoder(raw: &Weights) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for k in raw.keys() {
        if !k.starts_with("vocoder.") {
            continue;
        }
        let new = k.replace("vocoder.", "");
        let src = raw.require(k)?;
        let v = if new.contains("weight") && src.ndim() == 3 {
            if new.contains("ups") {
                src.transpose_axes(&[1, 2, 0])?
            } else {
                src.transpose_axes(&[0, 2, 1])?
            }
        } else {
            src.clone()
        };
        out.insert(new, v);
    }
    Ok(out)
}

/// The connector component (`connector.safetensors`): the two embeddings connectors
/// (`model.diffusion_model.{video,audio}_embeddings_connector.*` → prefix stripped) plus the
/// top-level `text_embedding_projection.*` (kept verbatim). No key renames, no conv transposes — a
/// straight prefix swap (matching the reference, which keeps the raw `.to_out.0`/`.ff.net.*` naming).
fn build_connector(raw: &Weights) -> HashMap<String, Array> {
    let mut out = HashMap::new();
    for k in raw.keys() {
        if k.contains("embeddings_connector") {
            if let Some(rest) = k.strip_prefix("model.diffusion_model.") {
                out.insert(
                    rest.to_string(),
                    raw.require(k).expect("key from keys()").clone(),
                );
            }
        } else if k.starts_with("text_embedding_projection.") {
            out.insert(
                k.to_string(),
                raw.require(k).expect("key from keys()").clone(),
            );
        }
    }
    out
}

/// Selectively Q4/Q8-quantize the transformer component in place: each predicate-matched Linear
/// `{base}.weight` (bf16) becomes `{base}.weight` (u32 packed) + `{base}.scales` + `{base}.biases`
/// via MLX `quantize` (same op `nn.quantize` calls — byte-identical on matched MLX). Non-matching
/// tensors (norms, gate logits, adaLN, biases) pass through.
fn quantize_transformer(
    m: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(m.len());
    for (k, v) in m {
        let base = k.strip_suffix(".weight");
        let is_q = base.is_some_and(|b| QUANT_SUFFIXES.iter().any(|s| b.ends_with(s)));
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

/// Materialize then write a component weight map to `dir/<name>.safetensors`.
fn save_component(dir: &Path, name: &str, weights: &HashMap<String, Array>) -> Result<()> {
    let arrays: Vec<&Array> = weights.values().collect();
    eval(arrays)?;
    Array::save_safetensors(
        weights.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        dir.join(format!("{name}.safetensors")),
    )?;
    Ok(())
}

/// `build_output_config(version="2.3", include_audio)` — the runtime `config.json`. The reference
/// sets `caption_channels=3840` then overwrites it with `null` for the 2.3 family; the net value is
/// `null`, written directly here.
fn build_output_config(include_audio: bool) -> serde_json::Value {
    serde_json::json!({
        "model_type": if include_audio { "AudioVideo" } else { "VideoOnly" },
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "caption_channels": serde_json::Value::Null,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20],
        "timestep_scale_multiplier": 1000,
        "av_ca_timestep_scale_multiplier": 1000,
        "norm_eps": 1e-6,
        "audio_sample_rate": 24000,
        "audio_latent_sample_rate": 16000,
        "audio_hop_length": 160,
        "audio_latent_channels": 8,
        "audio_mel_bins": 16,
        "model_version": "2.3.0",
        "is_v2": true,
        "apply_gated_attention": true,
        "cross_attention_adaln": true,
        "connector_positional_embedding_max_pos": [4096],
        "connector_rope_type": "SPLIT",
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 64,
        "caption_projection_first_linear": false,
        "caption_projection_second_linear": false,
    })
}

/// `build_embedded_config("2.3")` — the structural `embedded_config.json` the loader reads.
fn build_embedded_config() -> serde_json::Value {
    serde_json::json!({
        "transformer": {
            "_class_name": "AVTransformer3DModel",
            "attention_head_dim": 128,
            "attention_type": "default",
            "caption_channels": 3840,
            "cross_attention_dim": 4096,
            "in_channels": 128,
            "norm_eps": 1e-6,
            "num_attention_heads": 32,
            "num_layers": 48,
            "out_channels": 128,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_out_channels": 128,
            "audio_cross_attention_dim": 2048,
            "audio_positional_embedding_max_pos": [20],
            "use_embeddings_connector": true,
            "connector_attention_head_dim": 128,
            "connector_num_attention_heads": 32,
            "connector_num_layers": 8,
            "connector_positional_embedding_max_pos": [4096],
            "connector_num_learnable_registers": 128,
            "use_middle_indices_grid": true,
            "apply_gated_attention": true,
            "connector_apply_gated_attention": true,
            "caption_projection_first_linear": false,
            "caption_projection_second_linear": false,
            "audio_connector_attention_head_dim": 64,
            "audio_connector_num_attention_heads": 32,
            "cross_attention_adaln": true,
            "text_encoder_norm_type": "per_token_rms",
            "rope_type": "split",
            "frequencies_precision": "float64",
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "timestep_scale_multiplier": 1000,
            "av_ca_timestep_scale_multiplier": 1000.0
        },
        "vae": {
            "_class_name": "CausalVideoAutoencoder",
            "dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "patch_size": 4,
            "norm_layer": "pixel_norm",
            "spatial_padding_mode": "zeros",
            "timestep_conditioning": false,
            "decoder_blocks": [
                ["res_x", {"num_layers": 4}],
                ["compress_space", {"multiplier": 2}],
                ["res_x", {"num_layers": 6}],
                ["compress_time", {"multiplier": 2}],
                ["res_x", {"num_layers": 4}],
                ["compress_all", {"multiplier": 1}],
                ["res_x", {"num_layers": 2}],
                ["compress_all", {"multiplier": 2}],
                ["res_x", {"num_layers": 2}]
            ],
            "encoder_blocks": [
                ["res_x", {"num_layers": 4}],
                ["compress_space_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 6}],
                ["compress_time_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 4}],
                ["compress_all_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 2}],
                ["compress_all_res", {"multiplier": 1}],
                ["res_x", {"num_layers": 2}]
            ]
        },
        "audio_vae": {
            "model": {
                "params": {
                    "ddconfig": {
                        "double_z": true,
                        "mel_bins": 64,
                        "z_channels": 8,
                        "resolution": 256,
                        "in_channels": 2,
                        "out_ch": 2,
                        "ch": 128,
                        "ch_mult": [1, 2, 4],
                        "num_res_blocks": 2,
                        "dropout": 0.0,
                        "mid_block_add_attention": false,
                        "norm_type": "pixel",
                        "causality_axis": "height"
                    },
                    "sampling_rate": 16000
                }
            }
        },
        "scheduler": {
            "_class_name": "RectifiedFlowScheduler",
            "_diffusers_version": "0.25.1",
            "num_train_timesteps": 1000,
            "sampler": "LinearQuadratic"
        },
        "vocoder": {
            "vocoder": {
                "upsample_initial_channel": 1536,
                "resblock": "AMP1",
                "upsample_rates": [5, 2, 2, 2, 2, 2],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "stereo": true,
                "use_tanh_at_final": false,
                "activation": "snakebeta",
                "use_bias_at_final": false
            },
            "bwe": {
                "upsample_initial_channel": 512,
                "resblock": "AMP1",
                "upsample_rates": [6, 5, 2, 2, 2],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_kernel_sizes": [12, 11, 4, 4, 4],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "stereo": true,
                "use_tanh_at_final": false,
                "activation": "snakebeta",
                "use_bias_at_final": false,
                "apply_final_activation": false,
                "input_sampling_rate": 16000,
                "output_sampling_rate": 48000,
                "hop_length": 80,
                "n_fft": 512,
                "win_size": 512,
                "num_mels": 64
            }
        }
    })
}

/// Convert a single-file LTX-2.3 checkpoint (`source_file`) into a complete split MLX model dir at
/// `out_dir`, optionally Q4/Q8-quantizing the transformer and merging the latent upsampler(s) found
/// in `upscaler_dir` (the base `Lightricks/LTX-2.3` snapshot; `None` skips them). Returns `out_dir`.
///
/// Faithful Rust/MLX port of `mlx_video.convert` (sc-3233). The result loads directly through
/// [`crate::model::load`] (engine id `ltx_2_3`) via the worker's `modelPath` seam.
pub fn convert_and_assemble(
    source_file: impl AsRef<Path>,
    upscaler_dir: Option<impl AsRef<Path>>,
    out_dir: impl AsRef<Path>,
    opts: &LtxConvertOpts,
) -> Result<PathBuf> {
    let source = source_file.as_ref();
    let out = out_dir.as_ref();
    if !source.is_file() {
        return Err(Error::Msg(format!(
            "source LTX checkpoint not found: {}",
            source.display()
        )));
    }
    std::fs::create_dir_all(out)?;

    let raw = Weights::from_file(source)?;

    // Build + save each component, recording the emitted names in the reference's order.
    let mut components: Vec<String> = Vec::new();

    let mut transformer = sanitize_transformer(&raw);
    if opts.quantize {
        transformer = quantize_transformer(transformer, opts.bits, opts.group_size)?;
    }
    save_component(out, "transformer", &transformer)?;
    components.push("transformer".into());
    drop(transformer);

    let connector = build_connector(&raw);
    if !connector.is_empty() {
        save_component(out, "connector", &connector)?;
        components.push("connector".into());
    }

    let vae_decoder = sanitize_vae_decoder(&raw)?;
    if !vae_decoder.is_empty() {
        save_component(out, "vae_decoder", &vae_decoder)?;
        components.push("vae_decoder".into());
    }
    let vae_encoder = sanitize_vae_encoder(&raw)?;
    if !vae_encoder.is_empty() {
        save_component(out, "vae_encoder", &vae_encoder)?;
        components.push("vae_encoder".into());
    }

    if opts.include_audio {
        let audio_vae = sanitize_audio_vae(&raw)?;
        if !audio_vae.is_empty() {
            save_component(out, "audio_vae", &audio_vae)?;
            components.push("audio_vae".into());
        }
        let vocoder = sanitize_vocoder(&raw)?;
        if !vocoder.is_empty() {
            save_component(out, "vocoder", &vocoder)?;
            components.push("vocoder".into());
        }
    }

    // Merge the latent upsampler component(s) from the base repo (raw copies, no transform).
    if let Some(updir) = upscaler_dir.as_ref().map(AsRef::as_ref) {
        for (prefix, filename) in UPSCALER_SPECS_23 {
            let path = updir.join(filename);
            if !path.is_file() {
                continue;
            }
            let up = Weights::from_file(&path)?;
            let map: HashMap<String, Array> = up
                .keys()
                .map(|k| {
                    (
                        k.to_string(),
                        up.require(k).expect("key from keys()").clone(),
                    )
                })
                .collect();
            save_component(out, prefix, &map)?;
            components.push((*prefix).to_string());
        }
    }

    // Emit the config + manifest sidecars.
    write_json(
        out.join("config.json"),
        &build_output_config(opts.include_audio),
    )?;
    write_json(out.join("embedded_config.json"), &build_embedded_config())?;

    let mut manifest = serde_json::json!({
        "format": "split",
        "model_version": "2.3.0",
        "components": components,
        "source": source.display().to_string(),
        "variant": "distilled",
    });
    if opts.quantize {
        manifest["quantized"] = serde_json::Value::Bool(true);
        manifest["quantization_bits"] = serde_json::Value::from(opts.bits);
        manifest["quantization_group_size"] = serde_json::Value::from(opts.group_size);
        write_json(
            out.join("quantize_config.json"),
            &serde_json::json!({"quantization": {"bits": opts.bits, "group_size": opts.group_size}}),
        )?;
    }
    write_json(out.join("split_model.json"), &manifest)?;

    Ok(out.to_path_buf())
}

/// Pretty-print a JSON value to `path` (matching the reference `json.dump(..., indent=2)`).
fn write_json(path: PathBuf, value: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(&path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::all_close;

    /// Exact (bit-equal) array comparison.
    fn exact_eq(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape() && all_close(a, b, 0.0, 0.0, false).unwrap().item::<bool>()
    }

    /// The quant predicate matches exactly the reference's selected Linears: attention q/k/v/out
    /// (self, cross, and the cross-modal audio↔video attentions) + video/audio FFN in/out.
    #[test]
    fn quant_predicate_selects_reference_linears() {
        let q = |k: &str| {
            k.strip_suffix(".weight")
                .is_some_and(|b| QUANT_SUFFIXES.iter().any(|s| b.ends_with(s)))
        };
        // Quantized.
        for k in [
            "transformer_blocks.0.attn1.to_q.weight",
            "transformer_blocks.5.attn2.to_out.weight",
            "transformer_blocks.9.audio_attn1.to_v.weight",
            "transformer_blocks.9.audio_to_video_attn.to_k.weight",
            "transformer_blocks.9.video_to_audio_attn.to_out.weight",
            "transformer_blocks.3.ff.proj_in.weight",
            "transformer_blocks.3.audio_ff.proj_out.weight",
        ] {
            assert!(q(k), "should quantize: {k}");
        }
        // Dense (norms / gate logits / adaLN / patchify / proj_out / biases).
        for k in [
            "transformer_blocks.0.attn1.q_norm.weight",
            "transformer_blocks.0.attn1.to_gate_logits.weight",
            "transformer_blocks.0.attn1.to_q.bias",
            "adaln_single.linear.weight",
            "patchify_proj.weight",
            "proj_out.weight",
        ] {
            assert!(!q(k), "should stay dense: {k}");
        }
    }

    /// `sanitize_transformer` drops the embeddings connectors, strips the prefix, and applies the
    /// FFN / `to_out` / adaLN-linear renames.
    #[test]
    fn sanitize_transformer_renames_and_drops_connector() {
        let ones = |r: i32, c: i32| Array::ones::<f32>(&[r, c]).unwrap();
        let mut w = Weights::from_file(write_tmp(&[
            ("model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight", ones(4, 4)),
            ("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight", ones(8, 4)),
            ("model.diffusion_model.transformer_blocks.0.ff.net.2.weight", ones(4, 8)),
            ("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight", ones(4, 4)),
            ("model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight", ones(4, 4)),
            ("text_embedding_projection.video_aggregate_embed.weight", ones(4, 4)),
        ]))
        .unwrap();
        // a non-mdm key must also be ignored by the transformer sanitizer
        w.insert("vae.decoder.conv_in.weight".to_string(), ones(2, 2));

        let t = sanitize_transformer(&w);
        let mut keys: Vec<&str> = t.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "adaln_single.emb.timestep_embedder.linear1.weight",
                "transformer_blocks.0.attn1.to_out.weight",
                "transformer_blocks.0.ff.proj_in.weight",
                "transformer_blocks.0.ff.proj_out.weight",
            ]
        );
        // connector + text_proj + vae keys excluded from the transformer component
        assert!(!t
            .keys()
            .any(|k| k.contains("connector") || k.contains("aggregate") || k.contains("vae")));
    }

    /// `build_connector` keeps the two connectors (prefix-stripped, raw naming) + text projection.
    #[test]
    fn connector_keeps_raw_naming() {
        let ones = |r: i32, c: i32| Array::ones::<f32>(&[r, c]).unwrap();
        let w = Weights::from_file(write_tmp(&[
            ("model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight", ones(4, 4)),
            ("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight", ones(4, 4)),
            ("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight", ones(4, 4)),
            ("text_embedding_projection.audio_aggregate_embed.weight", ones(4, 4)),
        ]))
        .unwrap();
        let c = build_connector(&w);
        let mut keys: Vec<&str> = c.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                // raw .to_out.0 / .ff.net.* naming preserved (NOT sanitized like the transformer)
                "audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight",
                "text_embedding_projection.audio_aggregate_embed.weight",
                "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight",
            ]
        );
    }

    /// Conv channels-last transpose, gated on `conv`+`weight`. Conv2d `[O,I,H,W]→[O,H,W,I]`: a
    /// `[1,2,2,1]` source (row-major 0..3 over `(O,I,H,W)`) reorders to `[1,2,1,2]` values `[0,2,1,3]`.
    #[test]
    fn conv_transpose_channels_last() {
        let v = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 2, 1]);
        let out = conv_channels_last("decoder.conv_in.weight", &v, true, true).unwrap();
        let expected = Array::from_slice(&[0.0f32, 2.0, 1.0, 3.0], &[1, 2, 1, 2]);
        assert!(exact_eq(&out, &expected), "conv2d channels-last reorder");

        // audio path forbids ndim-5; a Conv3d weight there is left untouched (allow_3d = false).
        let v3 = Array::ones::<f32>(&[2, 3, 1, 1, 1]).unwrap();
        assert_eq!(
            conv_channels_last("decoder.conv_in.weight", &v3, false, true)
                .unwrap()
                .shape(),
            &[2, 3, 1, 1, 1]
        );
        // a non-`conv` ndim-4 weight is never transposed.
        let nonconv = Array::ones::<f32>(&[2, 3, 4, 5]).unwrap();
        assert_eq!(
            conv_channels_last("decoder.norm.weight", &nonconv, true, true)
                .unwrap()
                .shape(),
            &[2, 3, 4, 5]
        );
    }

    /// Write tensors to a unique temp safetensors file and return its path.
    fn write_tmp(entries: &[(&str, Array)]) -> PathBuf {
        let tag: usize = entries.iter().map(|(k, _)| k.len()).sum::<usize>() + entries.len();
        let path = std::env::temp_dir().join(format!("mlx_gen_ltx_convert_test_{tag}.safetensors"));
        Array::save_safetensors(
            entries.iter().map(|(k, v)| (*k, v)),
            None::<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        path
    }
}
