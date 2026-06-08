//! Snapshot-layout helpers for FLUX.1. The full checkpoint tree mirrors diffusers/mflux:
//! `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `transformer/`, `vae/`.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};
use mlx_gen_z_image::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

use crate::config::{FluxTokenizerKind, FluxVariant};
use crate::image_encoder::FluxIpImageEncoder;
use crate::ip_adapter::FluxIpAdapter;
use crate::text_encoder::{ClipTextEncoder, T5TextEncoder};
use crate::transformer::{FluxTransformer, FluxTransformerConfig};

pub fn load_clip_tokenizer(root: &Path) -> Result<TextTokenizer> {
    load_tokenizer(root, FluxTokenizerKind::Clip, FluxVariant::Schnell)
}

pub fn load_t5_tokenizer(root: &Path, variant: FluxVariant) -> Result<TextTokenizer> {
    load_tokenizer(root, FluxTokenizerKind::T5, variant)
}

pub fn load_clip_encoder(root: &Path) -> Result<ClipTextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    ClipTextEncoder::from_weights(&w, "")
}

pub fn load_t5_encoder(root: &Path) -> Result<T5TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder_2"))?;
    T5TextEncoder::from_weights(&w, "")
}

pub fn load_transformer(root: &Path, variant: FluxVariant) -> Result<FluxTransformer> {
    let w = Weights::from_dir(root.join("transformer"))?;
    FluxTransformer::from_weights(&w, "", &FluxTransformerConfig::for_variant(variant))
}

/// Load the XLabs FLUX IP-Adapter (epic 3621) from a directory containing the adapter weights and a
/// CLIP-ViT-L/14 image tower. The layout mirrors SDXL's `LoadSpec::ip_adapter` contract (the dir
/// carries both the adapter and its image encoder):
///
/// ```text
/// <dir>/ip_adapter.safetensors            # XLabs-AI/flux-ip-adapter
/// <dir>/image_encoder/model.safetensors   # openai/clip-vit-large-patch14 (vision tower)
/// ```
///
/// SceneWorks stages the CLIP tower next to the adapter (sc-3625); the engine only resolves the two
/// files here. Returns the image encoder (sc-3622) and the adapter modules (sc-3623).
pub fn load_flux_ip_adapter(dir: &Path) -> Result<(FluxIpImageEncoder, FluxIpAdapter)> {
    let adapter =
        FluxIpAdapter::from_weights(&Weights::from_file(dir.join("ip_adapter.safetensors"))?)?;
    let encoder = FluxIpImageEncoder::from_weights(&Weights::from_file(
        dir.join("image_encoder/model.safetensors"),
    )?)?;
    Ok((encoder, adapter))
}

pub fn load_vae(root: &Path) -> Result<Vae> {
    let mut w = Weights::from_dir(root.join("vae"))?;
    remap_vae_decoder(&mut w)?;
    remap_vae_encoder(&mut w)?;
    Vae::from_weights(&w, "", &VaeDecoderConfig::default_z_image())?.with_encoder(
        &w,
        "encoder",
        &VaeEncoderConfig::default_z_image(),
    )
}

pub fn load_vae_from_source(source: &WeightsSource) -> Result<Vae> {
    match source {
        WeightsSource::Dir(root) => load_vae(root),
        WeightsSource::File(path) => {
            let mut w = Weights::from_file(path)?;
            remap_vae_decoder(&mut w)?;
            remap_vae_encoder(&mut w)?;
            Vae::from_weights(&w, "", &VaeDecoderConfig::default_z_image())?.with_encoder(
                &w,
                "encoder",
                &VaeEncoderConfig::default_z_image(),
            )
        }
    }
}

pub fn remap_vae_decoder(w: &mut Weights) -> Result<()> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("decoder."))
        .map(String::from)
        .collect();
    for k in keys {
        let rest = k.strip_prefix("decoder.").unwrap();
        let (target, transpose): (String, bool) = match rest {
            "conv_in.weight" => ("conv_in.conv.weight".into(), true),
            "conv_in.bias" => ("conv_in.conv.bias".into(), false),
            "conv_out.weight" => ("conv_out.conv.weight".into(), true),
            "conv_out.bias" => ("conv_out.conv.bias".into(), false),
            "conv_norm_out.weight" => ("conv_norm_out.norm.weight".into(), false),
            "conv_norm_out.bias" => ("conv_norm_out.norm.bias".into(), false),
            _ => {
                let is_conv_w = rest.ends_with(".weight")
                    && (rest.contains(".conv1.")
                        || rest.contains(".conv2.")
                        || rest.contains(".conv_shortcut.")
                        || rest.contains(".upsamplers.0.conv."));
                (rest.to_string(), is_conv_w)
            }
        };
        let t = w.require(&k)?.clone();
        let t = if transpose {
            t.transpose_axes(&[0, 2, 3, 1])?
        } else {
            t
        };
        w.insert(target, t);
    }
    Ok(())
}

pub fn remap_vae_encoder(w: &mut Weights) -> Result<()> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("encoder."))
        .map(String::from)
        .collect();
    for k in keys {
        let rest = k.strip_prefix("encoder.").unwrap();
        let (suffix, transpose): (String, bool) = match rest {
            "conv_in.weight" => ("conv_in.conv.weight".into(), true),
            "conv_in.bias" => ("conv_in.conv.bias".into(), false),
            "conv_out.weight" => ("conv_out.conv.weight".into(), true),
            "conv_out.bias" => ("conv_out.conv.bias".into(), false),
            "conv_norm_out.weight" => ("conv_norm_out.norm.weight".into(), false),
            "conv_norm_out.bias" => ("conv_norm_out.norm.bias".into(), false),
            _ => {
                let is_conv_w = rest.ends_with(".weight")
                    && (rest.contains(".conv1.")
                        || rest.contains(".conv2.")
                        || rest.contains(".conv_shortcut.")
                        || rest.contains(".downsamplers.0.conv."));
                (rest.to_string(), is_conv_w)
            }
        };
        let target = format!("encoder.{suffix}");
        let t = w.require(&k)?.clone();
        let t = if transpose {
            t.transpose_axes(&[0, 2, 3, 1])?
        } else {
            t
        };
        w.insert(target, t);
    }
    Ok(())
}

fn load_tokenizer(
    root: &Path,
    kind: FluxTokenizerKind,
    variant: FluxVariant,
) -> Result<TextTokenizer> {
    let config = TokenizerConfig {
        max_length: kind.max_length(variant),
        pad_token_id: kind.pad_token_id(),
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    match kind {
        FluxTokenizerKind::Clip => {
            // The FLUX repo ships CLIP only as `vocab.json`+`merges.txt` (no `tokenizer.json`), and
            // the core byte-level `from_clip_bpe` mis-tokenizes CLIP (GPT-2 byte-level vs CLIP's
            // lowercased word-BPE with `</w>`), silently corrupting the pooled conditioning on every
            // render (sc-2787). Load the vendored, HF-faithful CLIP `tokenizer.json` compiled into the
            // crate and NEVER fall back to the broken path — a missing/invalid asset errors loudly.
            const CLIP_TOKENIZER_JSON: &str = include_str!("../assets/clip_tokenizer.json");
            TextTokenizer::from_json_str(CLIP_TOKENIZER_JSON, config)
        }
        FluxTokenizerKind::T5 => {
            // T5 ships a real `tokenizer.json` in `tokenizer_2/` (verified fork-identical); use it,
            // erroring loudly if absent rather than guessing.
            TextTokenizer::from_file(root.join(kind.subdir()).join("tokenizer.json"), config)
        }
    }
}
