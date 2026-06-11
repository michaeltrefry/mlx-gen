//! Crate error type. Replaces the early `Box<dyn Error>` placeholder with a typed enum
//! (sc-2373, disciplined-hybrid architecture). `From<&str>`/`From<String>` are provided so
//! existing `"...".into()` / `format!(...).into()` error sites keep compiling, while
//! `#[from]` lets `?` lift `mlx_rs` and IO errors transparently.

use thiserror::Error;

/// Anything that can go wrong in mlx-gen.
#[derive(Debug, Error)]
pub enum Error {
    /// An MLX op (matmul, quantize, SDPA, …) failed on device.
    #[error("MLX op failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// A required tensor key was absent from a loaded checkpoint/adapter.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem error while traversing a model directory.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// safetensors load/save error from mlx-rs.
    #[error("safetensors I/O failed: {0}")]
    SafeTensors(#[from] mlx_rs::error::IoError),

    /// A contextual message (config/validation/adapter-shape errors).
    #[error("{0}")]
    Msg(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}

/// Bridge the rich mlx-gen error into the backend-neutral [`gen_core::Error`] (epic 3720, D3 /
/// Option B). Legal under the orphan rule because the source type (`mlx_gen::Error`) is local. This
/// is what lets a family crate's `Generator::generate` — whose signature is `gen_core::Result` —
/// keep using `?` on the `mlx_gen::Result` helpers that do the actual tensor work: the device
/// exceptions box into [`gen_core::Error::Backend`], while the typed variants map across 1:1.
impl From<Error> for gen_core::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Mlx(ex) => gen_core::Error::backend(ex),
            Error::SafeTensors(io) => gen_core::Error::backend(io),
            Error::MissingTensor(s) => gen_core::Error::MissingTensor(s),
            Error::Io(io) => gen_core::Error::Io(io),
            Error::Msg(s) => gen_core::Error::Msg(s),
        }
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;
