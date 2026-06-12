//! `mlx-gen-sam3` — native-MLX SAM3 (Segment Anything 3) concept-segmentation detector for mlx-gen
//! (epic 4910).
//!
//! SAM3 adds open-vocabulary **Promptable Concept Segmentation** (PCS): segment *all* instances of
//! a text concept ("person") with no geometric prompt. This crate ports the **detector**
//! (`Sam3Model` — the net-new half) directly from the public Apache-2.0 `transformers` reference;
//! the **tracker** is SAM2 and is reused from `mlx-gen-sam2` (Phase F). Built in phases (sc-4919…):
//!   * **Phase A** (this milestone): [`vision`] — the PE ViT backbone + FPN neck shared by the
//!     detector and tracker, turning `pixel_values[1,3,1008,1008]` into the four 256-channel FPN
//!     feature maps (sc-4919).
//!   * Phase B: CLIP text encoder + tokenizer. Phase C: DETR encoder/decoder + presence + scoring.
//!     Phase D: mask head + processor → end-to-end "segment all *X*".
//!
//! Reference: `facebook/sam3` (PyTorch / `transformers`). No MLX reference port exists — this is a
//! direct-from-PyTorch port. SAM3 is a utility segmenter (not a generation provider), so the crate
//! exposes a plain API rather than self-registering into the model registry.

pub mod config;
pub mod detr;
pub mod geometry;
pub mod mask;
pub mod model;
pub mod text;
pub mod tracker;
pub mod video;
pub mod vision;

pub use config::{Sam3DetrConfig, Sam3GeometryConfig, Sam3TextConfig, Sam3VisionConfig};
pub use detr::{DetectorOutput, Sam3Detector};
pub use geometry::Sam3GeometryEncoder;
pub use mask::{post_process_instances, Instance, Sam3MaskHead};
pub use model::{Sam3ImageSegmenter, SegmentationOutput};
pub use text::{Sam3TextEncoder, Sam3Tokenizer};
pub use tracker::{MemoryFeatures, Sam3Tracker, TrackerFrameOutput, TrackerMask};
pub use video::{Sam3VideoModel, VideoFrameOutput};
pub use vision::Sam3VisionEncoder;
