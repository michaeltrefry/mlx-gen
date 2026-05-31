//! Adapter-file loaders — read a trained LoRA/LoKr `.safetensors` and install it onto a
//! model tree via [`AdaptableHost`]. Closes out sc-2343's loader piece.
//!
//! **LoKr** is generic and faithfully ported from the fork's `LoKrLoader.apply`: keys are
//! bare module paths (`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`) and the file
//! carries `networkType=lokr` + `alpha`/`rank` in safetensors metadata, so the delta and
//! target path are fully determined by the file — no per-model mapping table.
//!
//! **LoRA** here covers the PEFT bare-path convention (`‹prefix›‹path›.lora_A/B.weight` +
//! optional `‹path›.alpha`). The fork's *other* LoRA path — remapping diffusers/kohya key
//! conventions through per-model `LoRATarget` pattern tables — is model-specific and lands
//! with each model port (per ARCHITECTURE.md: model-specific orchestration lives with the
//! model), not in this generic framework.

use std::collections::BTreeMap;

use mlx_rs::Array;

use super::{reconstruct_lokr_delta, AdaptableHost, Adapter};
use crate::weights::Weights;
use crate::Result;

/// PEFT LoKr per-module factor suffixes; each factor is full (`lokr_w1`/`lokr_w2`) or
/// low-rank (`_a` @ `_b`). Exact-suffix matched, so order is for readability only.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// `true` if the file's `networkType` metadata marks it a LoKr adapter.
pub fn is_lokr(w: &Weights) -> bool {
    w.metadata("networkType")
        .map(|s| s.trim().eq_ignore_ascii_case("lokr"))
        .unwrap_or(false)
}

/// Outcome of installing an adapter file: how many target modules were adapted, and any
/// adapter keys that matched no module in the host (surfaced, never silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub applied: usize,
    pub unmatched_paths: Vec<String>,
}

/// Install a LoKr adapter file onto `host`. `scale` is the user-facing strength (the
/// `alpha/rank` factor is baked into the reconstructed delta, mirroring the fork).
pub fn apply_lokr(host: &mut impl AdaptableHost, w: &Weights, scale: f32) -> Result<ApplyReport> {
    let rank = w
        .metadata("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    // alpha defaults to rank (scale 1.0) when absent, matching PEFT.
    let alpha = w
        .metadata("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    // Group every lokr_* tensor by the module path preceding the suffix.
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut grouped: BTreeMap<String, BTreeMap<&str, &Array>> = BTreeMap::new();
    for key in &keys {
        for suffix in LOKR_SUFFIXES {
            if let Some(path) = key.strip_suffix(suffix) {
                let factor = &suffix[1..]; // drop the leading '.'
                grouped
                    .entry(path.to_string())
                    .or_default()
                    .insert(factor, w.require(key)?);
                break;
            }
        }
    }

    let mut report = ApplyReport::default();
    for (path, factors) in grouped {
        let parts: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                let delta = reconstruct_lokr_delta(
                    alpha,
                    rank,
                    &base_shape,
                    factors.get("lokr_w1").copied(),
                    factors.get("lokr_w1_a").copied(),
                    factors.get("lokr_w1_b").copied(),
                    factors.get("lokr_w2").copied(),
                    factors.get("lokr_w2_a").copied(),
                    factors.get("lokr_w2_b").copied(),
                )?;
                lin.push(Adapter::Lokr { delta, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

/// Install a PEFT-format LoRA file (`‹prefix›‹path›.lora_A.weight` / `.lora_B.weight`, with
/// optional `‹prefix›‹path›.alpha`) onto `host`. PEFT stores `lora_A: [r, in]`,
/// `lora_B: [out, r]`; we transpose to the residual form `x·A·B` (`A: [in, r]`, `B: [r, out]`)
/// and fold `alpha/rank` into `B`, matching the fork. `strip_prefix` removes a leading
/// namespace such as `"base_model.model."` or `"transformer."`.
pub fn apply_lora_peft(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let prefix = strip_prefix.unwrap_or("");
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let rest = match key.strip_prefix(prefix) {
            Some(r) => r,
            None => continue,
        };
        if let Some(path) = rest.strip_suffix(".lora_A.weight") {
            groups.entry(path.to_string()).or_default().a = Some(w.require(&key)?.clone());
        } else if let Some(path) = rest.strip_suffix(".lora_B.weight") {
            groups.entry(path.to_string()).or_default().b = Some(w.require(&key)?.clone());
        } else if let Some(path) = rest.strip_suffix(".alpha") {
            groups.entry(path.to_string()).or_default().alpha =
                w.require(&key)?.as_slice::<f32>().first().copied();
        }
    }

    let mut report = ApplyReport::default();
    for (path, parts) in groups {
        let (Some(a_raw), Some(b_raw)) = (parts.a, parts.b) else {
            continue;
        };
        let parents: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parents) {
            Some(lin) => {
                let a = a_raw.t(); // [r, in] -> [in, r]
                let mut b = b_raw.t(); // [out, r] -> [r, out]
                if let Some(alpha) = parts.alpha {
                    let rank = a.shape()[1] as f32; // r
                    b = b.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
                }
                lin.push(Adapter::Lora { a, b, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

#[derive(Default)]
struct LoraParts {
    a: Option<Array>,
    b: Option<Array>,
    alpha: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AdaptableLinear, Adapter};
    use mlx_rs::ops::all_close;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Minimal host with a single adaptable linear at path `["lin"]`.
    struct OneLinear {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for OneLinear {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["lin"] => Some(&mut self.lin),
                _ => None,
            }
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_loader_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn lora_peft_transposes_and_folds_alpha() {
        // base [out=4, in=3]; PEFT lora_A [r=2, in=3], lora_B [out=4, r=2], alpha=4 (rank=2).
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let path = tmp("lora.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: a = A^T [in,r], b = B^T * (alpha/rank=2.0) [r,out], scale 0.5.
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn unmatched_paths_are_reported_not_dropped() {
        // A LoKr file targeting a path the host doesn't have -> applied 0, path reported.
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("lokr_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("missing.path.lokr_w1", &dummy),
                ("missing.path.lokr_w2", &dummy),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_lokr(&w));

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_lokr(&mut host, &w, 1.0).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["missing.path".to_string()]);
    }
}
