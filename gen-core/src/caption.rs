//! The `Captioner` contract: image-to-text caption generation for training datasets and other
//! caller-owned workflows.
//!
//! Captioners are intentionally separate from [`Generator`](crate::generator::Generator): they
//! consume an image and produce text instead of synthesizing media. The request carries the
//! SceneWorks JoyCaption-facing knobs as generic metadata so the provider boundary can preserve
//! today's job surface while still letting callers construct prompts at their own edge.

use crate::media::Image;
use crate::runtime::{CancelFlag, Progress};
use crate::{Error, Result};

/// An image captioning provider.
pub trait Captioner {
    /// Stable identity + capability metadata, constructible without loading weights through the
    /// registry.
    fn descriptor(&self) -> &CaptionerDescriptor;

    /// Reject a request this captioner cannot serve before running model inference.
    fn validate(&self, req: &CaptionRequest) -> Result<()>;

    /// Generate a caption for one image. Long-running implementations should check
    /// [`CaptionRequest::cancel`] and report token/progress events through `on_progress`.
    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CaptionOutput>;
}

/// A single image-caption request.
#[derive(Clone, Debug, Default)]
pub struct CaptionRequest {
    /// The RGB image to caption.
    pub image: Image,
    /// The actual prompt sent to the model after caller/provider prompt construction. For
    /// JoyCaption this is either the rendered type/length/options template or the custom prompt
    /// override.
    pub prompt: String,
    /// Caption-style metadata that produced `prompt`, kept on the shared API so SceneWorks' current
    /// request surface can cross the Rust boundary without lossy side channels.
    pub options: CaptionOptions,
    /// Sampling controls for the autoregressive text decoder.
    pub sampling: CaptionSampling,
    /// Training trigger words that callers may prepend after raw generation. Captioners return raw
    /// model text; post-processing stays at the dataset boundary.
    pub trigger_words: Vec<String>,
    pub cancel: CancelFlag,
}

/// Caption prompt options preserved from the caller-facing job contract.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptionOptions {
    pub caption_type: String,
    pub caption_length: String,
    pub extra_options: Vec<String>,
    pub name_input: String,
    /// A caller-supplied prompt override. When non-empty, it should be the source of
    /// [`CaptionRequest::prompt`].
    pub custom_prompt: String,
    /// Loader/runtime hint carried by the current JoyCaption surface. MLX providers may ignore it
    /// when the backend has no analogous low-memory mode.
    pub low_vram: bool,
}

impl Default for CaptionOptions {
    fn default() -> Self {
        Self {
            caption_type: "Descriptive".to_owned(),
            caption_length: "long".to_owned(),
            extra_options: Vec::new(),
            name_input: String::new(),
            custom_prompt: String::new(),
            low_vram: false,
        }
    }
}

/// Autoregressive sampling knobs for text caption generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptionSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: u32,
    /// RNG seed for stochastic sampling (`temperature > 0`). `None` draws a fresh per-call seed via
    /// [`default_seed`](crate::generator::default_seed) so repeated calls vary; pass `Some(seed)` to
    /// reproduce an exact caption. (At `temperature == 0` decoding is greedy and the seed is unused.)
    pub seed: Option<u64>,
}

impl Default for CaptionSampling {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.9,
            max_new_tokens: 256,
            seed: None,
        }
    }
}

/// Caption generation result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptionOutput {
    pub text: String,
    pub generated_tokens: Option<u32>,
    pub finish_reason: Option<CaptionFinishReason>,
}

/// Why generation stopped, when the provider can report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionFinishReason {
    StopToken,
    MaxTokens,
    Cancelled,
}

/// A captioner's stable identity + advertised capabilities.
#[derive(Clone, Debug)]
pub struct CaptionerDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this captioner ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub capabilities: CaptionCapabilities,
}

/// The shared caption capability surface. Provider-specific constraints are layered on top by each
/// captioner's own `validate`.
#[derive(Clone, Debug, Default)]
pub struct CaptionCapabilities {
    pub caption_types: Vec<&'static str>,
    pub caption_lengths: Vec<&'static str>,
    pub supports_custom_prompt: bool,
    pub supports_low_vram: bool,
    pub min_image_size: u32,
    pub max_image_size: u32,
    pub max_prompt_chars: usize,
    pub max_name_chars: usize,
    pub max_extra_options: usize,
    pub max_extra_option_chars: usize,
    pub max_trigger_words: usize,
    pub max_trigger_word_chars: usize,
    pub max_new_tokens: u32,
    pub mac_only: bool,
}

impl CaptionCapabilities {
    /// Reject request fields that exceed the advertised shared capability surface.
    pub fn validate_request(&self, id: &str, req: &CaptionRequest) -> Result<()> {
        if req.image.width < self.min_image_size
            || req.image.height < self.min_image_size
            || req.image.width > self.max_image_size
            || req.image.height > self.max_image_size
        {
            return Err(Error::Msg(format!(
                "{id}: image size {}x{} outside supported range {}..={}",
                req.image.width, req.image.height, self.min_image_size, self.max_image_size
            )));
        }
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(format!("{id}: prompt is required")));
        }
        if req.prompt.chars().count() > self.max_prompt_chars {
            return Err(Error::Msg(format!(
                "{id}: prompt is longer than {} characters",
                self.max_prompt_chars
            )));
        }
        if !req.options.custom_prompt.trim().is_empty() && !self.supports_custom_prompt {
            return Err(Error::Msg(format!(
                "{id}: custom prompts are not supported"
            )));
        }
        if req.options.low_vram && !self.supports_low_vram {
            return Err(Error::Msg(format!("{id}: low_vram is not supported")));
        }
        if !self.caption_types.is_empty()
            && !self
                .caption_types
                .contains(&req.options.caption_type.as_str())
        {
            return Err(Error::Msg(format!(
                "{id}: unsupported caption_type {:?} (supported: {:?})",
                req.options.caption_type, self.caption_types
            )));
        }
        if !self.caption_lengths.is_empty()
            && !req
                .options
                .caption_length
                .chars()
                .all(|c| c.is_ascii_digit())
            && !self
                .caption_lengths
                .contains(&req.options.caption_length.as_str())
        {
            return Err(Error::Msg(format!(
                "{id}: unsupported caption_length {:?} (supported: {:?} or a word count)",
                req.options.caption_length, self.caption_lengths
            )));
        }
        if req.options.name_input.chars().count() > self.max_name_chars {
            return Err(Error::Msg(format!(
                "{id}: name_input is longer than {} characters",
                self.max_name_chars
            )));
        }
        if req.options.extra_options.len() > self.max_extra_options {
            return Err(Error::Msg(format!(
                "{id}: too many extra options (max {})",
                self.max_extra_options
            )));
        }
        if req
            .options
            .extra_options
            .iter()
            .any(|option| option.chars().count() > self.max_extra_option_chars)
        {
            return Err(Error::Msg(format!(
                "{id}: an extra option is longer than {} characters",
                self.max_extra_option_chars
            )));
        }
        if req.trigger_words.len() > self.max_trigger_words {
            return Err(Error::Msg(format!(
                "{id}: too many trigger words (max {})",
                self.max_trigger_words
            )));
        }
        if req
            .trigger_words
            .iter()
            .any(|word| word.chars().count() > self.max_trigger_word_chars)
        {
            return Err(Error::Msg(format!(
                "{id}: a trigger word is longer than {} characters",
                self.max_trigger_word_chars
            )));
        }
        if req.sampling.temperature < 0.0 || req.sampling.temperature > 2.0 {
            return Err(Error::Msg(format!(
                "{id}: temperature must be between 0 and 2"
            )));
        }
        if req.sampling.top_p < 0.0 || req.sampling.top_p > 1.0 {
            return Err(Error::Msg(format!("{id}: top_p must be between 0 and 1")));
        }
        if req.sampling.max_new_tokens == 0 || req.sampling.max_new_tokens > self.max_new_tokens {
            return Err(Error::Msg(format!(
                "{id}: max_new_tokens {} out of range 1..={}",
                req.sampling.max_new_tokens, self.max_new_tokens
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    fn caps() -> CaptionCapabilities {
        CaptionCapabilities {
            caption_types: vec!["Descriptive", "Straightforward"],
            caption_lengths: vec!["any", "short", "long"],
            supports_custom_prompt: true,
            supports_low_vram: true,
            min_image_size: 32,
            max_image_size: 4096,
            max_prompt_chars: 4000,
            max_name_chars: 120,
            max_extra_options: 16,
            max_extra_option_chars: 500,
            max_trigger_words: 32,
            max_trigger_word_chars: 120,
            max_new_tokens: 1024,
            mac_only: true,
        }
    }

    fn base_req() -> CaptionRequest {
        CaptionRequest {
            image: img(384, 384),
            prompt: "Write a detailed description for this image.".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn caption_defaults_match_training_caption_surface() {
        let req = CaptionRequest::default();
        assert_eq!(req.options.caption_type, "Descriptive");
        assert_eq!(req.options.caption_length, "long");
        assert_eq!(req.sampling.temperature, 0.6);
        assert_eq!(req.sampling.top_p, 0.9);
        assert_eq!(req.sampling.max_new_tokens, 256);
    }

    #[test]
    fn validate_request_accepts_supported_surface() {
        let c = caps();
        assert!(c.validate_request("captioner", &base_req()).is_ok());
        assert!(c
            .validate_request(
                "captioner",
                &CaptionRequest {
                    options: CaptionOptions {
                        caption_type: "Straightforward".to_owned(),
                        caption_length: "85".to_owned(),
                        extra_options: vec!["Mention clothing details.".to_owned()],
                        name_input: "Alex".to_owned(),
                        custom_prompt: "Use the override.".to_owned(),
                        low_vram: true,
                    },
                    prompt: "Use the override.".to_owned(),
                    trigger_words: vec!["alex_token".to_owned()],
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn validate_request_enforces_shared_surface() {
        let c = caps();
        let too_many_options = vec!["x".to_owned(); 17];
        let cases = vec![
            CaptionRequest {
                image: img(16, 384),
                ..base_req()
            },
            CaptionRequest {
                prompt: String::new(),
                ..base_req()
            },
            CaptionRequest {
                prompt: "x".repeat(4001),
                ..base_req()
            },
            CaptionRequest {
                options: CaptionOptions {
                    caption_type: "Other".to_owned(),
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                options: CaptionOptions {
                    caption_length: "novella".to_owned(),
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                options: CaptionOptions {
                    name_input: "x".repeat(121),
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                options: CaptionOptions {
                    extra_options: too_many_options,
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                trigger_words: vec!["x".repeat(121)],
                ..base_req()
            },
            CaptionRequest {
                sampling: CaptionSampling {
                    temperature: 2.1,
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                sampling: CaptionSampling {
                    top_p: 1.1,
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                sampling: CaptionSampling {
                    max_new_tokens: 0,
                    ..Default::default()
                },
                ..base_req()
            },
            CaptionRequest {
                sampling: CaptionSampling {
                    max_new_tokens: 1025,
                    ..Default::default()
                },
                ..base_req()
            },
        ];
        for (i, req) in cases.iter().enumerate() {
            assert!(
                c.validate_request("captioner", req).is_err(),
                "case {i} should have been rejected"
            );
        }
    }
}
