//! Qwen-Image-Edit VL tokenization (sc-2465 slice 6b-2). Port of the fork's
//! `QwenVisionLanguageTokenizer` (`use_picture_prefix=False`) + `QwenVisionLanguageProcessor`:
//! turn a reference image + edit prompt into the four inputs the [`QwenVisionLanguageEncoder`]
//! consumes — `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`.
//!
//! Pipeline: resize the reference to the **condition size** (~384² area, sides rounded to /32,
//! BICUBIC) → the [`QwenImageProcessor`] (smart_resize to /28 + patchify, already parity-tested) →
//! expand the single `<|image_pad|>` in the fixed edit template to `prod(grid)//merge²` copies →
//! tokenize the formatted string with special tokens.
//!
//! [`QwenVisionLanguageEncoder`]: crate::text_encoder::QwenVisionLanguageEncoder

use mlx_rs::Array;

use mlx_gen::array::host_i32;
use mlx_gen::tokenizer::{TextTokenizer, TokenizerOutput};
use mlx_gen::Result;

use crate::image_processor::{
    resize_bicubic_u8, resize_lanczos_u8, ImageInput, QwenImageProcessor,
};
use crate::pipeline::pack_latents;
use crate::vae::QwenVae;

/// Inputs for [`crate::text_encoder::QwenVisionLanguageEncoder::encode`].
pub struct EditInputs {
    pub input_ids: Array,
    pub attention_mask: Array,
    pub pixel_values: Array,
    pub grid_thw: Array,
}

/// The image-only output of [`preprocess_edit_image`] — `pixel_values` + `grid_thw` for the vision
/// tower, plus the `<|image_pad|>` count the grid expands to. Depends only on the reference image,
/// so the Edit generator computes it **once** and reuses it for the positive + negative prompts.
pub struct EditImage {
    pub pixel_values: Array,
    pub grid_thw: Array,
    pub n_image_tokens: usize,
}

/// Target condition area (fork `CONDITION_IMAGE_SIZE`): the reference is scaled to ~384²,
/// preserving aspect, with each side rounded to a multiple of 32.
const CONDITION_AREA: f64 = 384.0 * 384.0;

/// Condition-resize dims `(width, height)`: `w = round(√(area·ratio)/32)·32`, `h = round((w/ratio)
/// /32)·32`, `ratio = width/height`. Round-half-to-even (Python `round`).
pub fn condition_resize_dims(width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let cw = (CONDITION_AREA * ratio).sqrt();
    let ch = cw / ratio;
    let cw = (cw / 32.0).round_ties_even() * 32.0;
    let ch = (ch / 32.0).round_ties_even() * 32.0;
    (cw as usize, ch as usize)
}

/// The fork's edit chat template (`use_picture_prefix=False`) with the single `<|image_pad|>`
/// expanded to `n_image_tokens` copies and the user prompt inserted.
pub fn build_edit_text(prompt: &str, n_image_tokens: usize) -> String {
    let pads = "<|image_pad|>".repeat(n_image_tokens);
    format!(
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, \
         texture, objects, background), then explain how the user's text instruction should alter \
         or modify the image. Generate a new image that meets the user's requirements while \
         maintaining consistency with the original input where appropriate.<|im_end|>\n\
         <|im_start|>user\n<|vision_start|>{pads}<|vision_end|>{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Image-only half of [`tokenize_edit`]: condition-resize (BICUBIC, /32) + patchify the reference,
/// returning the `pixel_values`, `grid_thw`, and the `<|image_pad|>` count. `image` is RGB uint8
/// HWC. Independent of the prompt, so the Edit generator runs this **once** per generation (F-004).
pub fn preprocess_edit_image(
    processor: &QwenImageProcessor,
    image: ImageInput,
) -> Result<EditImage> {
    // 1. Condition resize (BICUBIC, /32) — clip8-rounded f32 back to u8 for the processor.
    let (cw, ch) = condition_resize_dims(image.width, image.height);
    let resized: Vec<u8> = if (image.height, image.width) == (ch, cw) {
        image.data.to_vec()
    } else {
        resize_bicubic_u8(image.data, image.height, image.width, ch, cw)
            .iter()
            .map(|&v| v as u8)
            .collect()
    };

    // 2. Patchify (smart_resize to /28 + flatten) — parity-tested in image_processor.
    let processed = processor.preprocess(ImageInput {
        data: &resized,
        height: ch,
        width: cw,
    })?;

    // image_pad count = prod(grid) // merge².
    let grid = host_i32(&processed.grid_thw)?; // [grid_t, grid_h, grid_w]
    let merge2 = (processor.merge_size * processor.merge_size) as i32;
    let n_image_tokens = (grid[0] * grid[1] * grid[2] / merge2) as usize;

    Ok(EditImage {
        pixel_values: processed.pixel_values,
        grid_thw: processed.grid_thw,
        n_image_tokens,
    })
}

/// Prompt-only half of [`tokenize_edit`]: expand the edit template's `<|image_pad|>` to
/// `n_image_tokens` copies (from [`preprocess_edit_image`]) and tokenize. Run once per prompt
/// (positive + negative) while the image preprocessing is shared.
pub fn tokenize_edit_text(
    tokenizer: &TextTokenizer,
    prompt: &str,
    n_image_tokens: usize,
) -> Result<TokenizerOutput> {
    tokenizer
        .tokenize_preformatted(&build_edit_text(prompt, n_image_tokens))
        .map_err(Into::into)
}

/// Tokenize a reference image + edit prompt for the VL encoder. `image` is RGB uint8 HWC. Composes
/// [`preprocess_edit_image`] + [`tokenize_edit_text`]; the Edit generator calls those two directly
/// so the image work is shared across the positive/negative prompts.
pub fn tokenize_edit(
    tokenizer: &TextTokenizer,
    processor: &QwenImageProcessor,
    prompt: &str,
    image: ImageInput,
) -> Result<EditInputs> {
    let pre = preprocess_edit_image(processor, image)?;
    let tok = tokenize_edit_text(tokenizer, prompt, pre.n_image_tokens)?;
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
    Ok(EditInputs {
        input_ids,
        attention_mask,
        pixel_values: pre.pixel_values,
        grid_thw: pre.grid_thw,
    })
}

/// VAE-encode + pack a reference image for the dual-latent path. Resize to `(calc_w, calc_h)` via
/// **LANCZOS** (the fork's `scale_to_dimensions`), normalize `[0,255] → [-1,1]` as NCHW, VAE-encode,
/// drop the temporal axis, and `pack_latents`. Returns `(image_latents [1, (calc_h/16)·(calc_w/16),
/// 64], cond_grid (latent_h, latent_w))`. Port of `QwenEditUtil.create_image_conditioning_latents`
/// for a single reference (`calc` = the VL condition dims).
pub fn encode_reference_latents(
    vae: &QwenVae,
    image: ImageInput,
    calc_w: u32,
    calc_h: u32,
) -> Result<(Array, (usize, usize))> {
    let (cw, ch) = (calc_w as usize, calc_h as usize);
    let resized = if (image.height, image.width) == (ch, cw) {
        image.data.iter().map(|&p| p as f32).collect::<Vec<f32>>()
    } else {
        resize_lanczos_u8(image.data, image.height, image.width, ch, cw)
    };
    // [0,255] → [-1,1], laid out NCHW [1, 3, calc_h, calc_w].
    let plane = ch * cw;
    let mut nchw = vec![0f32; 3 * plane];
    for y in 0..ch {
        for x in 0..cw {
            for c in 0..3 {
                let v = resized[(y * cw + x) * 3 + c] / 255.0 * 2.0 - 1.0;
                nchw[c * plane + y * cw + x] = v;
            }
        }
    }
    let img = Array::from_slice(&nchw, &[1, 3, calc_h as i32, calc_w as i32]);
    let latent = vae.encode(&img)?.squeeze_axes(&[2])?; // [1,16,1,h/8,w/8] → [1,16,h/8,w/8]
    let packed = pack_latents(&latent, calc_w, calc_h)?;
    Ok((packed, ((calc_h / 16) as usize, (calc_w / 16) as usize)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_text_expands_image_pad_and_keeps_structure() {
        let t = build_edit_text("make it night", 3);
        assert_eq!(t.matches("<|image_pad|>").count(), 3);
        assert!(t.contains("<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>make it night<|im_end|>"));
        assert!(t.starts_with("<|im_start|>system\nDescribe the key features"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn condition_resize_dims_match_reference() {
        assert_eq!(condition_resize_dims(512, 512), (384, 384)); // square
        assert_eq!(condition_resize_dims(768, 512), (480, 320)); // 3:2 landscape
        assert_eq!(condition_resize_dims(512, 768), (320, 480)); // 2:3 portrait
    }
}
