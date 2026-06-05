//! # mlx-gen-face
//!
//! Native MLX face-analysis stack (epic 3079) — shared infrastructure for the PuLID-FLUX
//! (epic 3069) and InstantID (epic 3061) identity ports, replacing the torch/onnx
//! preprocessing with a Rust/MLX path (the "zero Python inference on Mac" north star).
//!
//! Sub-models (per the sc-3080 spike, all Tier-B native):
//! - **ArcFace iresnet100** ([`iresnet`]) — the fidelity-critical 512-d recognition embedding,
//!   a faithful port of antelopev2 `glintr100` (sc-3081).
//! - **SCRFD detector** ([`scrfd`]) — 5-pt landmark + bbox detection (sc-3082).
//! - **5-pt alignment** ([`align`]) — insightface-faithful `norm_crop` (112²) feeding ArcFace, plus
//!   the facexlib `align_warp_face` (512²) crop for the EVA-CLIP / parsing path (sc-3083).
//! - **BiSeNet parsing** ([`bisenet`]) — 19-class face segmentation → PuLID `face_features_image`
//!   (sc-3084).
//! - The unified `FaceAnalysis` API (sc-3085) lands alongside.

pub mod align;
pub mod bisenet;
pub mod iresnet;
pub mod scrfd;

pub use align::{
    align_face_512, estimate_norm, norm_crop, to_arcface_input, warp_affine, Affine2x3,
};
pub use bisenet::{face_features_image, to_parse_input, BiSeNet};
pub use iresnet::ArcFace;
pub use scrfd::{Detection, Scrfd};
