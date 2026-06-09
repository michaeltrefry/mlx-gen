//! # mlx-gen-instantid
//!
//! InstantID (epic 3109) — identity-preserving SDXL. Builds on the existing crates rather than
//! re-implementing them:
//! - the face **Resampler** (ArcFace 512 → 16×2048 face tokens) is `mlx-gen-sdxl`'s generic IP-Adapter
//!   `Resampler` under `ResamplerConfig::instantid_face()` (sc-3110);
//! - the **IdentityNet** is a stock diffusers SDXL `ControlNet` (sc-3112) — `ControlNet::from_weights`
//!   + `UNetConfig::sdxl_base()`, no conversion;
//! - the **decoupled cross-attention** (16 face tokens) reuses `load_ip_kv_pairs` + the UNet's
//!   `install_ip_adapter` (sc-3113).
//!
//! This crate adds the InstantID-specific glue. [`kps`] is the keypoint control-image renderer
//! (sc-3111) — a bit-exact port of the vendored `draw_kps` (OpenCV 4.13) plus the letterbox aspect
//! handling and the canonical multi-view landmark sets.

pub mod kps;
pub mod model;
pub mod openpose;

pub use kps::{draw_kps, letterbox, view_angle_kps, ANGLE_SET_ORDER, VIEW_ANGLE_KPS};
pub use model::{
    InstantId, InstantIdPaths, InstantIdRequest, DEFAULT_CONTROLNET_SCALE, DEFAULT_IP_SCALE,
};
pub use openpose::{
    draw_bodypose, face_box_from_keypoints, normalize_keypoints, square_fit, BodyPoint,
    NUM_BODY_KEYPOINTS, STICKWIDTH,
};
