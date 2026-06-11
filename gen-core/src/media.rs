//! Concrete media types that cross the public `Generator`/`Transform` boundary.
//!
//! Deliberately free of any `mlx-rs` types: a consumer can use the contract without depending
//! on MLX array types. Models decode their internal MLX tensors into these at the edge.

/// An 8-bit RGB image, row-major, with `pixels.len() == width * height * 3`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Interleaved PCM audio — the audio track of a video generation (e.g. LTX-2.3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioTrack {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}
