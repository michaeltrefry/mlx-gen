//! Snapshot-layout helpers for FLUX.1. The full checkpoint tree mirrors diffusers/mflux:
//! `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `transformer/`, `vae/`.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};
use mlx_gen_z_image::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

use crate::config::{FluxTokenizerKind, FluxVariant};
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
    let dir = root.join(kind.subdir());
    let config = TokenizerConfig {
        max_length: kind.max_length(variant),
        pad_token_id: kind.pad_token_id(),
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    let fast = dir.join("tokenizer.json");
    if fast.exists() {
        return TextTokenizer::from_file(fast, config);
    }
    match kind {
        FluxTokenizerKind::Clip => {
            TextTokenizer::from_clip_bpe(dir.join("vocab.json"), dir.join("merges.txt"), config)
        }
        FluxTokenizerKind::T5 => TextTokenizer::from_file(fast, config),
    }
}
