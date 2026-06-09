//! `mlx-gen-sam2` — native-MLX SAM2 (Segment Anything 2) segmenter for mlx-gen (epic 3704).
//!
//! Ports the SAM2 person/object segmenter to Rust `mlx-rs`, zero Python in the shipped or weight
//! path — the engine for the person-track segmenter (epic 3482 Python Eradication, sc-3488) and a
//! "smart select" mask source for the Image Editor (epic 2427).
//!
//! SAM2 is a unified image+video model, ported in layers (image is the mandatory foundation and is
//! independently useful as first-class still-image segmentation):
//!   * **Phase A — image foundation** (this milestone):
//!     - [`image_encoder`] — the Hiera hierarchical ViT trunk + FPN neck (sc-3705, *this slice*),
//!       turning `pixel_values[1,3,1024,1024]` into the 3 backbone-FPN feature maps + position
//!       encodings the mask decoder consumes.
//!     - prompt encoder + two-way mask decoder + box→mask segmenter (sc-3706, follows).
//!   * **Phase B — video layer**: memory bank + memory attention + propagation (sc-3713/3714).
//!
//! Reference: the MLX-native `avbiswas/sam2-mlx` (`mlx_sam`) and `eisneim/sam2.1_mlx`. SAM2 is a
//! utility segmenter (not a generation provider), so the crate exposes a plain API rather than
//! self-registering into the model registry.

pub mod config;
pub mod hiera;
pub mod image_encoder;
pub mod sam_heads;
pub mod segmenter;

pub use config::{Sam2ImageEncoderConfig, Sam2ModelSize};
pub use image_encoder::{Sam2ImageEncoder, Sam2ImageEncoderOutput};
pub use sam_heads::{MaskDecoder, PromptEncoder};
pub use segmenter::Sam2Segmenter;
