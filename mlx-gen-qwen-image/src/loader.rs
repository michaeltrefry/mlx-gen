//! Real-checkpoint loading for Qwen-Image: assemble the tokenizer, Qwen2.5-VL text encoder,
//! 60-layer MMDiT transformer, and causal-Conv3d VAE from a `Qwen/Qwen-Image` snapshot directory,
//! applying the diffusers-checkpoint → internal-name remaps (the fork's `qwen_weight_mapping.py`).
//!
//! Snapshot layout (standard diffusers multi-component tree):
//! ```text
//!   <root>/tokenizer/{tokenizer.json | vocab.json + merges.txt}
//!   <root>/text_encoder/*.safetensors
//!   <root>/transformer/*.safetensors
//!   <root>/vae/*.safetensors
//! ```
//! The transformer/VAE checkpoints are keyed by the diffusers tree; we remap to the *internal*
//! names the slice-1/2/3 modules expect (the same `to_pattern` the parity goldens were dumped
//! under). The text-encoder layout (`model.*`) maps directly onto the encoder under the `"model"`
//! prefix, so it needs no remap.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};
use mlx_rs::Array;

use crate::control_transformer::{QwenControlNet, QwenControlNetConfig};
use crate::text_encoder::vision::{VisionConfig, VisionTransformer};
use crate::text_encoder::{QwenTextEncoder, QwenTextEncoderConfig, QwenVisionLanguageEncoder};
use crate::transformer::{QwenTransformer, QwenTransformerConfig};
use crate::vae::QwenVae;

/// Qwen2 pad token id (`<|endoftext|>`).
const PAD_TOKEN_ID: i32 = 151643;
/// The fork's `LanguageTokenizer` max_length for the `qwen` tokenizer.
const MAX_LENGTH: usize = 1058;

/// Load the Qwen2 tokenizer with the Qwen-Image T2I template + padding policy (`padding="longest"`
/// → no max-length padding for a single prompt). The snapshot must contain `tokenizer/tokenizer.json`
/// (the HF *fast* serialization); the upstream repo ships only `vocab.json` + `merges.txt`, so run
/// `tools/build_qwen_tokenizer.py` once to materialize it (the same fast tokenizer the fork builds
/// at runtime).
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let path = root.join("tokenizer/tokenizer.json");
    if !path.exists() {
        return Err(Error::Msg(format!(
            "missing {}: the Qwen-Image snapshot ships only vocab.json + merges.txt; run \
             tools/build_qwen_tokenizer.py to materialize the fast tokenizer.json",
            path.display()
        )));
    }
    TextTokenizer::from_file(
        path,
        TokenizerConfig {
            max_length: MAX_LENGTH,
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::QwenImage,
            pad_to_max_length: false,
        },
    )
}

/// Load the Qwen2.5-VL text encoder (text path). The on-disk `model.*` keys map directly onto the
/// encoder tree under the `"model"` prefix (validated in slice 2) — no remap needed.
pub fn load_text_encoder(root: &Path) -> Result<QwenTextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    QwenTextEncoder::from_weights(&w, "model", &QwenTextEncoderConfig::qwen_image())
}

/// Load the Qwen2.5-VL **vision transformer** (Qwen-Image-Edit) from a snapshot's `text_encoder/`
/// shards. The vision weights live under `visual.*` alongside the LM; we apply the fork's vision
/// rules ([`remap_vision_keys`]) then read under the `"visual"` prefix. Edit-only — the T2I snapshot
/// has no `visual.*` weights.
pub fn load_vision_encoder(root: &Path) -> Result<VisionTransformer> {
    let mut w = Weights::from_dir(root.join("text_encoder"))?;
    remap_vision_keys(&mut w)?;
    VisionTransformer::from_weights(&w, "visual", &VisionConfig::qwen_image_edit())
}

/// Load the Qwen-Image-**Edit** vision-language conditioning encoder: the Qwen2.5-VL LM (`model.*`,
/// same layout as T2I) + the vision transformer (`visual.*`), composed into a
/// [`QwenVisionLanguageEncoder`]. Edit-only.
pub fn load_vision_language_encoder(root: &Path) -> Result<QwenVisionLanguageEncoder> {
    let lm = load_text_encoder(root)?;
    let visual = load_vision_encoder(root)?;
    Ok(QwenVisionLanguageEncoder::new(lm, visual))
}

/// The fork's vision weight transforms (`qwen_weight_mapping.py`), applied in place: transpose the
/// patch-embed conv (PyTorch `[O,I,kD,kH,kW]` → MLX `[O,kD,kH,kW,I]`) and rename the merger
/// `Sequential` `mlp.{0,2}` → `mlp_{0,1}`. Everything else under `visual.*` matches 1:1.
pub fn remap_vision_keys(w: &mut Weights) -> Result<()> {
    const PATCH_EMBED: &str = "visual.patch_embed.proj.weight";
    if let Some(t) = w.get(PATCH_EMBED).cloned() {
        if t.shape().len() == 5 {
            w.insert(PATCH_EMBED, t.transpose_axes(&[0, 2, 3, 4, 1])?);
        }
    }
    for (from, to) in [
        ("visual.merger.mlp.0.weight", "visual.merger.mlp_0.weight"),
        ("visual.merger.mlp.0.bias", "visual.merger.mlp_0.bias"),
        ("visual.merger.mlp.2.weight", "visual.merger.mlp_1.weight"),
        ("visual.merger.mlp.2.bias", "visual.merger.mlp_1.bias"),
    ] {
        w.alias(from, to);
    }
    Ok(())
}

/// Load the 60-layer MMDiT transformer, applying the diffusers→internal key renames.
pub fn load_transformer(root: &Path) -> Result<QwenTransformer> {
    let mut w = Weights::from_dir(root.join("transformer"))?;
    remap_transformer_keys(&mut w);
    QwenTransformer::from_weights(&w, "", &QwenTransformerConfig::qwen_image())
}

/// Load the transformer for Qwen-Image-**Edit-2511** — identical to [`load_transformer`] but with
/// `zero_cond_t` on (the conditioning-image latent tokens are modulated as clean / timestep 0).
pub fn load_transformer_edit(root: &Path) -> Result<QwenTransformer> {
    let mut w = Weights::from_dir(root.join("transformer"))?;
    remap_transformer_keys(&mut w);
    QwenTransformer::from_weights(&w, "", &QwenTransformerConfig::qwen_image_edit())
}

/// Load the InstantX `Qwen-Image-ControlNet-Union` control transformer (epic 3401). The checkpoint
/// is a single `diffusion_pytorch_model.safetensors` (`File`) or a dir of shards (`Dir`). Its block
/// keys (`transformer_blocks.{i}.attn.to_out.0`, `…img_mod.1`, `…img_mlp.net.0.proj`, …) are the
/// same diffusers names as the base, so we apply the same [`remap_transformer_keys`]; the control
/// top-level modules (`img_in`/`txt_in`/`txt_norm`/`time_text_embed`/`controlnet_x_embedder`/
/// `controlnet_blocks.{i}`) match 1:1 and pass through unchanged.
pub fn load_controlnet(control: &WeightsSource) -> Result<QwenControlNet> {
    let mut w = match control {
        WeightsSource::File(p) => Weights::from_file(p)?,
        WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    remap_transformer_keys(&mut w);
    QwenControlNet::from_weights(&w, "", &QwenControlNetConfig::qwen_image_union())
}

/// Load the causal-Conv3d VAE, applying the diffusers→internal key remap (structural renames +
/// conv-weight transposes + RMSNorm `gamma`→1-D).
pub fn load_vae(root: &Path) -> Result<QwenVae> {
    let mut w = Weights::from_dir(root.join("vae"))?;
    remap_vae_keys(&mut w)?;
    QwenVae::from_weights(&w)
}

/// diffusers transformer checkpoint → internal names (port of `QwenWeightMapping`'s transformer
/// rules). All weights are plain Linears (no transpose); only a handful of modules are renamed —
/// `to_out.0`→`attn_to_out.0`, `{img,txt}_mod.1`→`{img,txt}_mod_linear`, the
/// `{img,txt}_mlp.net.{0.proj,2}` feed-forwards → `{img,txt}_ff.mlp_{in,out}`. Everything else
/// matches 1:1 and is left in place. Applied across all 60 blocks by substring match.
pub fn remap_transformer_keys(w: &mut Weights) {
    const RENAMES: &[(&str, &str)] = &[
        (".attn.to_out.0.", ".attn.attn_to_out.0."),
        (".img_mod.1.", ".img_mod_linear."),
        (".txt_mod.1.", ".txt_mod_linear."),
        (".img_mlp.net.0.proj.", ".img_ff.mlp_in."),
        (".img_mlp.net.2.", ".img_ff.mlp_out."),
        (".txt_mlp.net.0.proj.", ".txt_ff.mlp_in."),
        (".txt_mlp.net.2.", ".txt_ff.mlp_out."),
    ];
    let keys: Vec<String> = w.keys().map(String::from).collect();
    for k in keys {
        for (from, to) in RENAMES {
            if k.contains(from) {
                w.alias(&k, &k.replace(from, to));
                break;
            }
        }
    }
}

/// diffusers VAE checkpoint → internal names (port of `QwenWeightMapping`'s VAE rules). Renames the
/// structure (decoder `up_blocks.{b}`→`up_block{b}`; the encoder's *flat* `down_blocks.{0..10}`→the
/// grouped `down_blocks.{g}.resnets.{r}` / `downsamplers.0` tree), inserts `.conv3d` for the
/// `CausalConv3d` modules, renames `conv_shortcut`→`skip_conv` and `resample.1`→`resample_conv`,
/// and applies the conv-weight transposes (`[O,I,D,H,W]`→`[O,D,H,W,I]`, `[O,I,H,W]`→`[O,H,W,I]`)
/// + RMSNorm `gamma`→1-D. The unused temporal `time_conv` is skipped (the fork never calls it).
pub fn remap_vae_keys(w: &mut Weights) -> Result<()> {
    let keys: Vec<String> = w.keys().map(String::from).collect();
    for k in &keys {
        let Some(target) = vae_internal_key(k) else {
            continue; // skipped (time_conv)
        };
        let t = w.require(k)?;
        let t = transform_vae_tensor(k, t)?;
        w.insert(target, t);
    }
    Ok(())
}

/// Map one on-disk VAE key to its internal name, or `None` to skip it.
fn vae_internal_key(k: &str) -> Option<String> {
    if k.contains(".time_conv.") {
        return None; // unused temporal conv (T2I up/down-sampling is purely spatial)
    }
    // Encoder: flat `down_blocks.{flat}` → grouped `down_blocks.{g}.resnets.{r}` / `downsamplers.0`.
    if let Some(rest) = k.strip_prefix("encoder.down_blocks.") {
        let (flat_str, tail) = rest.split_once('.')?;
        let flat: usize = flat_str.parse().ok()?;
        let (group, slot) = (flat / 3, flat % 3);
        if slot == 2 {
            let leaf = tail.strip_prefix("resample.1.")?;
            return Some(format!(
                "encoder.down_blocks.{group}.downsamplers.0.resample_conv.{leaf}"
            ));
        }
        return Some(format!(
            "encoder.down_blocks.{group}.resnets.{slot}.{}",
            remap_resnet_tail(tail)
        ));
    }
    // Decoder: `up_blocks.{b}` → `up_block{b}`.
    if let Some(rest) = k.strip_prefix("decoder.up_blocks.") {
        let (b, tail) = rest.split_once('.')?;
        if let Some(up) = tail.strip_prefix("upsamplers.0.") {
            let leaf = up.strip_prefix("resample.1.")?;
            return Some(format!(
                "decoder.up_block{b}.upsamplers.0.resample_conv.{leaf}"
            ));
        }
        let after = tail.strip_prefix("resnets.")?;
        let (r, rtail) = after.split_once('.')?;
        return Some(format!(
            "decoder.up_block{b}.resnets.{r}.{}",
            remap_resnet_tail(rtail)
        ));
    }
    Some(remap_generic_vae(k))
}

/// Leaf rename for a resnet sub-tree tail (`conv1.weight`, `norm1.gamma`, `conv_shortcut.bias`, …).
fn remap_resnet_tail(tail: &str) -> String {
    if let Some(p) = tail.strip_suffix(".gamma") {
        return format!("{p}.weight");
    }
    let Some((parent, leaf)) = tail.rsplit_once('.') else {
        return tail.to_string();
    };
    match parent {
        "conv1" | "conv2" => format!("{parent}.conv3d.{leaf}"),
        "conv_shortcut" => format!("skip_conv.conv3d.{leaf}"),
        _ => tail.to_string(),
    }
}

/// Leaf rename for the regular (non-down/up-block) VAE keys: `gamma`→`weight`, and `.conv3d` insert
/// for the `CausalConv3d` modules (attention `to_qkv`/`proj` stay flat — they're 2-D convs).
fn remap_generic_vae(k: &str) -> String {
    if let Some(p) = k.strip_suffix(".gamma") {
        return format!("{p}.weight");
    }
    let Some((parent, leaf)) = k.rsplit_once('.') else {
        return k.to_string();
    };
    let parent_name = parent.rsplit('.').next().unwrap_or(parent);
    if matches!(
        parent_name,
        "conv_in" | "conv_out" | "conv1" | "conv2" | "quant_conv" | "post_quant_conv"
    ) {
        return format!("{parent}.conv3d.{leaf}");
    }
    k.to_string()
}

/// Apply the fork's weight transform for a VAE tensor, keyed off the leaf + rank (mirrors
/// `WeightTransforms`): `gamma`→1-D, rank-5 conv weight `[O,I,D,H,W]`→`[O,D,H,W,I]`, rank-4 conv
/// weight `[O,I,H,W]`→`[O,H,W,I]`, biases unchanged.
fn transform_vae_tensor(src_key: &str, t: &Array) -> Result<Array> {
    if src_key.ends_with(".gamma") {
        return Ok(t.reshape(&[t.shape()[0]])?);
    }
    if src_key.ends_with(".weight") {
        return Ok(match t.shape().len() {
            5 => t.transpose_axes(&[0, 2, 3, 4, 1])?,
            4 => t.transpose_axes(&[0, 2, 3, 1])?,
            _ => t.clone(),
        });
    }
    Ok(t.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_renames() {
        // a representative key from each rename family.
        let cases = [
            (
                "transformer_blocks.7.attn.to_out.0.weight",
                "transformer_blocks.7.attn.attn_to_out.0.weight",
            ),
            (
                "transformer_blocks.12.img_mod.1.bias",
                "transformer_blocks.12.img_mod_linear.bias",
            ),
            (
                "transformer_blocks.3.txt_mlp.net.0.proj.weight",
                "transformer_blocks.3.txt_ff.mlp_in.weight",
            ),
            (
                "transformer_blocks.3.img_mlp.net.2.weight",
                "transformer_blocks.3.img_ff.mlp_out.weight",
            ),
        ];
        const RENAMES: &[(&str, &str)] = &[
            (".attn.to_out.0.", ".attn.attn_to_out.0."),
            (".img_mod.1.", ".img_mod_linear."),
            (".txt_mod.1.", ".txt_mod_linear."),
            (".img_mlp.net.0.proj.", ".img_ff.mlp_in."),
            (".img_mlp.net.2.", ".img_ff.mlp_out."),
            (".txt_mlp.net.0.proj.", ".txt_ff.mlp_in."),
            (".txt_mlp.net.2.", ".txt_ff.mlp_out."),
        ];
        for (from, want) in cases {
            let got = RENAMES
                .iter()
                .find(|(f, _)| from.contains(f))
                .map(|(f, t)| from.replace(f, t))
                .unwrap();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn vae_encoder_flat_to_grouped() {
        assert_eq!(
            vae_internal_key("encoder.down_blocks.0.conv1.weight").unwrap(),
            "encoder.down_blocks.0.resnets.0.conv1.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.1.norm2.gamma").unwrap(),
            "encoder.down_blocks.0.resnets.1.norm2.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.2.resample.1.bias").unwrap(),
            "encoder.down_blocks.0.downsamplers.0.resample_conv.bias"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.3.conv_shortcut.weight").unwrap(),
            "encoder.down_blocks.1.resnets.0.skip_conv.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.9.conv1.weight").unwrap(),
            "encoder.down_blocks.3.resnets.0.conv1.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.10.conv2.bias").unwrap(),
            "encoder.down_blocks.3.resnets.1.conv2.conv3d.bias"
        );
        assert!(vae_internal_key("encoder.down_blocks.8.time_conv.weight").is_none());
    }

    #[test]
    fn vae_decoder_and_generic() {
        assert_eq!(
            vae_internal_key("decoder.up_blocks.0.resnets.2.conv1.weight").unwrap(),
            "decoder.up_block0.resnets.2.conv1.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.up_blocks.1.resnets.0.conv_shortcut.weight").unwrap(),
            "decoder.up_block1.resnets.0.skip_conv.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.up_blocks.0.upsamplers.0.resample.1.weight").unwrap(),
            "decoder.up_block0.upsamplers.0.resample_conv.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.conv_in.weight").unwrap(),
            "decoder.conv_in.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.norm_out.gamma").unwrap(),
            "decoder.norm_out.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.mid_block.attentions.0.to_qkv.weight").unwrap(),
            "decoder.mid_block.attentions.0.to_qkv.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.mid_block.attentions.0.norm.gamma").unwrap(),
            "decoder.mid_block.attentions.0.norm.weight"
        );
        assert_eq!(
            vae_internal_key("post_quant_conv.weight").unwrap(),
            "post_quant_conv.conv3d.weight"
        );
    }
}
