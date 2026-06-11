//! The **backend-neutral** crate error. gen-core cannot name `mlx_rs` (or candle) types, so device
//! failures arrive boxed in [`Error::Backend`] — each tensor backend lifts its own exception into
//! it (mlx-gen via `From<mlx_gen::Error>`, see `mlx-gen/src/error.rs`). The typed [`Error::Canceled`]
//! and [`Error::Unsupported`] variants are contract-load-bearing: the worker and the conformance
//! testkit distinguish cancellation and capability gaps from generic failure (epic 3720, D3).

use thiserror::Error;

/// Anything that can go wrong across a gen-core contract call.
#[derive(Debug, Error)]
pub enum Error {
    /// A backend tensor/device operation failed (an MLX exception, a CUDA error, …), boxed because
    /// gen-core cannot name backend types. Construct via [`Error::backend`].
    #[error("backend op failed: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),

    /// A required tensor key was absent from a loaded checkpoint/adapter.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem error (model-dir traversal, safetensors open).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The request asked for something this engine/backend cannot do. Candle gating depends on this
    /// being typed — do **not** stringify it into [`Error::Msg`].
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Generation/training was cancelled via a `CancelFlag`. Typed so the worker and testkit can
    /// distinguish cancellation from failure; providers must check at step boundaries and return
    /// this (never a partial output).
    #[error("cancelled")]
    Canceled,

    /// A contextual message (config/validation/adapter-shape errors).
    #[error("{0}")]
    Msg(String),
}

impl Error {
    /// Lift a concrete backend error into [`Error::Backend`]. Backends call this (or rely on their
    /// own `From` impl) at the seam where an `mlx_rs`/candle `Result` crosses into gen-core.
    pub fn backend(e: impl std::error::Error + Send + Sync + 'static) -> Error {
        Error::Backend(Box::new(e))
    }
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

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_strings_are_stable() {
        assert_eq!(Error::Canceled.to_string(), "cancelled");
        assert_eq!(
            Error::Unsupported("q4".into()).to_string(),
            "unsupported: q4"
        );
        assert_eq!(
            Error::MissingTensor("unet.x".into()).to_string(),
            "missing tensor: unet.x"
        );
        assert_eq!(Error::Msg("boom".into()).to_string(), "boom");
    }

    #[test]
    fn backend_wraps_a_source_error() {
        let io = std::io::Error::new(std::io::ErrorKind::Other, "device lost");
        let e = Error::backend(io);
        assert!(matches!(e, Error::Backend(_)));
        assert_eq!(e.to_string(), "backend op failed: device lost");
    }

    #[test]
    fn str_and_string_lift_to_msg() {
        assert!(matches!(Error::from("x"), Error::Msg(_)));
        assert!(matches!(Error::from(String::from("y")), Error::Msg(_)));
    }
}
