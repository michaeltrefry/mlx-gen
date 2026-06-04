//! Weight loading — safetensors → MLX arrays by dotted key, plus file metadata, with a
//! dtype helper. No torch dependency: reads safetensors directly via mlx-rs.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{Array, Dtype};

use crate::{Error, Result};

/// A loaded set of named tensors (dotted keys, e.g. `"layers.0.attention.to_q.weight"`)
/// plus the file's string metadata (e.g. a LoKr adapter's `networkType` / `alpha` / `rank`).
pub struct Weights {
    tensors: HashMap<String, Array>,
    metadata: HashMap<String, String>,
}

impl Weights {
    /// Load a single `.safetensors` file (tensors + metadata).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let (tensors, metadata) = Array::load_safetensors_with_metadata(path.as_ref())?;
        Ok(Self { tensors, metadata })
    }

    /// Load and merge every `.safetensors` file under `dir` (sharded checkpoints). Keys
    /// across shards are disjoint, so a plain merge reconstructs the full tensor set
    /// without parsing the index — no torch, no shard map needed.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors files in {}", dir.display()).into());
        }
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for f in files {
            let (t, m) = Array::load_safetensors_with_metadata(&f)?;
            tensors.extend(t);
            metadata.extend(m);
        }
        Ok(Self { tensors, metadata })
    }

    pub fn get(&self, key: &str) -> Option<&Array> {
        self.tensors.get(key)
    }

    /// Get a tensor by key, returning an error (not panicking) when it is absent.
    pub fn require(&self, key: &str) -> Result<&Array> {
        self.tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Insert (or overwrite) a tensor under `key`. Used by checkpoint remapping (diffusers →
    /// internal names + conv-weight transposes) when loading real weights.
    pub fn insert(&mut self, key: impl Into<String>, tensor: Array) {
        self.tensors.insert(key.into(), tensor);
    }

    /// Copy the tensor at `from` to the new key `to` (no-op if `from` is absent). A convenience
    /// for the identity-but-renamed entries in a checkpoint→internal mapping.
    pub fn alias(&mut self, from: &str, to: &str) {
        if let Some(t) = self.tensors.get(from).cloned() {
            self.tensors.insert(to.to_string(), t);
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Cast every tensor to `dtype` in place — mirrors the way a reference loader downcasts the
    /// whole checkpoint at load (e.g. the vendored SDXL `_load_safetensor_weights(..., float16=True)`
    /// applies `v.astype(mx.float16)` to every tensor). A no-op when `dtype` already matches.
    pub fn cast_all(&mut self, dtype: Dtype) -> Result<()> {
        for v in self.tensors.values_mut() {
            if v.dtype() != dtype {
                *v = v.as_dtype(dtype)?;
            }
        }
        Ok(())
    }
}

/// Cast to a target compute dtype (e.g. bf16, mirroring mflux's torch_convert downcast).
pub fn to_dtype(a: &Array, dtype: Dtype) -> Result<Array> {
    Ok(a.as_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_file_with_metadata() {
        let dir = std::env::temp_dir().join("mlx_gen_weights_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.safetensors");

        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        Array::save_safetensors(vec![("blk.weight", &a)], Some(&meta), &path).unwrap();

        let w = Weights::from_file(&path).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w.get("blk.weight").is_some());
        assert!(w.require("blk.weight").is_ok());
        assert!(w.require("missing").is_err());
        assert_eq!(w.metadata("networkType"), Some("lokr"));
    }

    #[test]
    fn to_dtype_casts_to_bf16() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        assert_eq!(
            to_dtype(&a, Dtype::Bfloat16).unwrap().dtype(),
            Dtype::Bfloat16
        );
    }
}
