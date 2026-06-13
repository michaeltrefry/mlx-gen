//! sc-5134: native Qwen2.5-VL **vision tower** — the planner's image/video ViT encoder.
//!
//! Port of `Qwen2_5_VisionTransformerPretrainedModel`
//! (`_vendor/bernini/bernini/models/modeling_qwen2_5_vl.py:651-814`). Produces `out_hidden`-d
//! (3584) ViT tokens from packed patch pixels + a `grid_thw` geometry. Weights live under `visual.*`
//! in the sc-5144 `qwen2_5_vl.safetensors` snapshot (390 tensors).
//!
//! Structure mirrored faithfully:
//!   - **Patch embed** — a bias-free `Conv3d` with kernel == stride == `[temporal 2, 14, 14]`. Since
//!     the kernel spans the whole patch, the conv is exactly a per-patch matmul, so we fold the 5-D
//!     `[embed, in, t, h, w]` weight to `[embed, in·t·h·w]` and run it as a [`AdaptableLinear`].
//!   - **`depth` blocks** — pre-norm (`Qwen2RMSNorm`, eps 1e-6) → fused-QKV (bias) attention with a
//!     **2-D rotary** (head_dim/2, θ 10000, NeoX `rotate_half`, f32) → `proj`; pre-norm → **SwiGLU**
//!     MLP (`gate`/`up`/`down`, **bias**, SiLU). Attention is windowed (`window_size 112`) on every
//!     block except `fullatt_block_indexes [7,15,23,31]` (full). The window reorder permutes
//!     merge-units; `cu_seqlens` (full, per frame) vs `cu_window_seqlens` (windowed) give the
//!     block-diagonal additive mask; softmax accumulates in f32.
//!   - **Patch merger** — `ln_q` RMSNorm → concat each `spatial_merge_size²`(=4) group → `5120` →
//!     `Linear → GELU → Linear` → `out_hidden`; then the window permutation is undone (`argsort`).
//!
//! All the integer index gymnastics (`rot_pos_emb`, `get_window_index`, `cu_seqlens`) depend only on
//! `grid_thw`, so they are computed host-side in [`VisionTower::build_plan`] — exactly mirroring the
//! reference — and the resulting permutation / rope table / block masks are handed to the MLX graph.
//! Linears are [`AdaptableLinear`]s so sc-5146 can quantize them Q4/Q8 at load.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis, matmul, multiply, softmax_axis, split, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{gelu_exact, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

/// Qwen2.5-VL vision-tower config (the `vision_config` block of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub depth: i32,
    pub fullatt_block_indexes: Vec<i32>,
    pub spatial_merge_size: i32,
    pub window_size: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub out_hidden_size: i32,
}

impl Default for VisionConfig {
    /// Qwen2.5-VL-7B vision tower (the Bernini planner base).
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            num_heads: 16,
            intermediate_size: 3420,
            depth: 32,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            spatial_merge_size: 2,
            window_size: 112,
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            out_hidden_size: 3584,
        }
    }
}

impl VisionConfig {
    /// Read the `vision_config` sub-object of a `qwen2_5_vl_config.json` (the sc-5144 snapshot copy of
    /// `mllm/config.json`).
    pub fn from_config_json(path: &std::path::Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| Error::Msg(format!("parse {}: {e}", path.display())))?;
        let vc = v.get("vision_config").unwrap_or(&v);
        let d = Self::default();
        let i = |k: &str, dv: i32| {
            vc.get(k)
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(dv as i64) as i32
        };
        let fullatt = vc
            .get("fullatt_block_indexes")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_i64().map(|n| n as i32))
                    .collect::<Vec<_>>()
            })
            .unwrap_or(d.fullatt_block_indexes);
        Ok(Self {
            hidden_size: i("hidden_size", d.hidden_size),
            num_heads: i("num_heads", d.num_heads),
            intermediate_size: i("intermediate_size", d.intermediate_size),
            depth: i("depth", d.depth),
            fullatt_block_indexes: fullatt,
            spatial_merge_size: i("spatial_merge_size", d.spatial_merge_size),
            window_size: i("window_size", d.window_size),
            patch_size: i("patch_size", d.patch_size),
            temporal_patch_size: i("temporal_patch_size", d.temporal_patch_size),
            // the package config spells this `in_chans`.
            in_channels: vc
                .get("in_chans")
                .or_else(|| vc.get("in_channels"))
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(d.in_channels as i64) as i32,
            out_hidden_size: i("out_hidden_size", d.out_hidden_size),
        })
    }

    fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }

    /// `spatial_merge_size²` — patches per merged token.
    fn merge_unit(&self) -> i32 {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// Window edge in **merged-token** units: `window // merge // patch`.
    fn vit_merger_window_size(&self) -> i32 {
        self.window_size / self.spatial_merge_size / self.patch_size
    }
}

fn linear(w: &Weights, prefix: &str, bias: bool) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let b = if bias {
        Some(w.require(&format!("{prefix}.bias"))?.clone())
    } else {
        None
    };
    Ok(AdaptableLinear::dense(weight, b))
}

/// HF half-split rotary `rotate_half`: `cat(-x[d/2:], x[:d/2])` on the last axis.
fn rotate_half(x: &Array) -> Result<Array> {
    let ax = x.ndim() as i32 - 1;
    let parts = split(x, 2, ax)?;
    Ok(concatenate_axis(&[&parts[1].negative()?, &parts[0]], ax)?)
}

/// One vision block: pre-norm windowed/full attention + pre-norm SwiGLU MLP, both residual.
struct Block {
    norm1: Array,
    norm2: Array,
    qkv: AdaptableLinear,
    proj: AdaptableLinear,
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Block {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm1: w.require(&format!("{prefix}.norm1.weight"))?.clone(),
            norm2: w.require(&format!("{prefix}.norm2.weight"))?.clone(),
            qkv: linear(w, &format!("{prefix}.attn.qkv"), true)?,
            proj: linear(w, &format!("{prefix}.attn.proj"), true)?,
            gate: linear(w, &format!("{prefix}.mlp.gate_proj"), true)?,
            up: linear(w, &format!("{prefix}.mlp.up_proj"), true)?,
            down: linear(w, &format!("{prefix}.mlp.down_proj"), true)?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.qkv.quantize(bits, None)?;
        self.proj.quantize(bits, None)?;
        self.gate.quantize(bits, None)?;
        self.up.quantize(bits, None)?;
        self.down.quantize(bits, None)
    }

    /// Eager attention over `x` `[seq, dim]` with the precomputed `cos`/`sin` `[seq, head_dim]` (f32)
    /// and an additive `mask` `[1, seq, seq]`. q/k/v project → split heads → 2-D RoPE (q,k cast to f32)
    /// → `softmax(q·kᵀ/√d + mask)·v` (f32 softmax) → proj.
    fn attention(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        nh: i32,
    ) -> Result<Array> {
        let seq = x.shape()[0];
        let dim = x.shape()[1];
        let hd = dim / nh;
        let dtype = x.dtype();

        let qkv = self.qkv.forward(x)?.reshape(&[seq, 3, nh, hd])?;
        let parts = split(&qkv, 3, 1)?; // 3 × [seq, 1, nh, hd]
        let q = parts[0].reshape(&[seq, nh, hd])?;
        let k = parts[1].reshape(&[seq, nh, hd])?;
        let v = parts[2].reshape(&[seq, nh, hd])?;

        // 2-D RoPE in f32, cos/sin broadcast over the head axis ([seq,1,head_dim]).
        let cos = cos.reshape(&[seq, 1, hd])?;
        let sin = sin.reshape(&[seq, 1, hd])?;
        let rope = |t: &Array| -> Result<Array> {
            let f = t.as_dtype(Dtype::Float32)?;
            let r = add(&multiply(&f, &cos)?, &multiply(&rotate_half(&f)?, &sin)?)?;
            Ok(r.as_dtype(dtype)?)
        };
        let q = rope(&q)?.transpose_axes(&[1, 0, 2])?; // [nh, seq, hd]
        let k = rope(&k)?.transpose_axes(&[1, 0, 2])?;
        let v = v.transpose_axes(&[1, 0, 2])?;

        let scale = (hd as f32).powf(-0.5);
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 2, 1])?)?,
            Array::from_f32(scale),
        )?;
        let scores = add(&scores, &mask.as_dtype(scores.dtype())?)?; // [nh, seq, seq] + [1,seq,seq]
        let weights = softmax_axis(&scores, -1, true)?; // f32 accumulation
        let out = matmul(&weights, &v)? // [nh, seq, hd]
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq, dim])?;
        self.proj.forward(&out)
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate.forward(x)?)?, &self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array, nh: i32) -> Result<Array> {
        let a = self.attention(&rms_norm(x, &self.norm1, RMS_EPS)?, cos, sin, mask, nh)?;
        let x = add(x, &a)?;
        let m = self.mlp(&rms_norm(&x, &self.norm2, RMS_EPS)?)?;
        Ok(add(&x, &m)?)
    }
}

/// Patch merger: `ln_q` RMSNorm → concat merge-unit groups → `Linear → GELU → Linear`.
struct Merger {
    ln_q: Array,
    mlp0: AdaptableLinear,
    mlp2: AdaptableLinear,
}

impl Merger {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln_q: w.require(&format!("{prefix}.ln_q.weight"))?.clone(),
            mlp0: linear(w, &format!("{prefix}.mlp.0"), true)?,
            mlp2: linear(w, &format!("{prefix}.mlp.2"), true)?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mlp0.quantize(bits, None)?;
        self.mlp2.quantize(bits, None)
    }

    /// `x` `[seq, context_dim]` → `[merged, out_hidden]` (`merged = seq / merge_unit`).
    fn forward(&self, x: &Array, merged: i32, merge_dim: i32) -> Result<Array> {
        let x = rms_norm(x, &self.ln_q, RMS_EPS)?.reshape(&[merged, merge_dim])?;
        let x = gelu_exact(&self.mlp0.forward(&x)?)?;
        self.mlp2.forward(&x)
    }
}

/// Host-side `grid_thw`-derived plan: the window permutation, its inverse, the f32 rope table (original
/// merge-unit order), and the per-token block ids for the full + windowed additive masks.
struct Plan {
    seq: i32,
    merged: i32,
    window_index: Array,  // i32 [merged]
    reverse_index: Array, // i32 [merged]
    rope: Array,          // f32 [seq, head_dim/2] (original order)
    full_bid: Vec<i32>,   // [seq]
    win_bid: Vec<i32>,    // [seq]
}

/// Remove consecutive duplicates (`torch.unique_consecutive`).
fn dedup_consecutive(v: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(v.len());
    for &x in v {
        if out.last() != Some(&x) {
            out.push(x);
        }
    }
    out
}

/// Per-token block id from cumulative boundaries `cu` (`[0, b1, …, seq]`): token `p` ∈ `[cu[k],cu[k+1])`
/// → `k`. Mirrors the reference's diagonal-block mask construction.
fn block_ids(cu: &[i32], seq: i32) -> Vec<i32> {
    let mut bid = vec![0i32; seq as usize];
    for k in 0..cu.len().saturating_sub(1) {
        for p in cu[k]..cu[k + 1] {
            bid[p as usize] = k as i32;
        }
    }
    bid
}

/// Build an additive attention mask `[1, seq, seq]` (f32): `0` within a block, `-inf` across blocks.
fn additive_mask(bid: &[i32], seq: i32) -> Array {
    let s = seq as usize;
    let mut data = vec![0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if bid[i] != bid[j] {
                data[i * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    Array::from_slice(&data, &[1, seq, seq])
}

/// The native Qwen2.5-VL vision tower.
pub struct VisionTower {
    patch_embed: AdaptableLinear,
    blocks: Vec<Block>,
    merger: Merger,
    cfg: VisionConfig,
}

impl VisionTower {
    /// Build from a converted planner snapshot (`visual.*` keys). `prefix` is the vision namespace —
    /// `"visual"` for the sc-5144 layout.
    pub fn from_weights(w: &Weights, cfg: VisionConfig, prefix: &str) -> Result<Self> {
        // Fold the bias-free Conv3d weight `[embed, in, t, ph, pw]` → `[embed, in·t·ph·pw]` so the
        // full-kernel conv runs as a per-patch matmul.
        let conv = w
            .require(&format!("{prefix}.patch_embed.proj.weight"))?
            .clone();
        let embed = conv.shape()[0];
        let in_dim = conv.shape().iter().skip(1).product::<i32>();
        let patch_embed = AdaptableLinear::dense(conv.reshape(&[embed, in_dim])?, None);

        let blocks = (0..cfg.depth)
            .map(|i| Block::from_weights(w, &format!("{prefix}.blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let merger = Merger::from_weights(w, &format!("{prefix}.merger"))?;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            cfg,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Quantize every block + the merger linears (Q4/Q8, group 64). RMSNorm weights stay dense.
    /// (sc-5146 load-time quantization.)
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        self.merger.quantize(bits)
    }

    /// Compute the `grid_thw`-derived plan host-side (faithful to `rot_pos_emb` + `get_window_index` +
    /// the `forward` cu-seqlen logic). `grid_thw` rows are `[t, h, w]` in patches.
    fn build_plan(&self, grid: &[[i32; 3]]) -> Result<Plan> {
        let c = &self.cfg;
        let sms = c.spatial_merge_size;
        let mu = c.merge_unit();
        let vmws = c.vit_merger_window_size();
        let rd = (c.head_dim() / 2) as usize; // rope width per token
        let half = rd / 2; // inv_freq length = head_dim/4
        let inv: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / ROPE_THETA.powf((2 * j) as f32 / rd as f32))
            .collect();

        let mut rope_rows: Vec<f32> = Vec::new(); // [seq, rd], original merge-unit order
        let mut window_index: Vec<i32> = Vec::new();
        let mut cu_window: Vec<i32> = vec![0];
        let mut cu_seqlens: Vec<i32> = vec![0];
        let mut window_index_id: i32 = 0;

        for g in grid {
            let (t, h, w) = (g[0], g[1], g[2]);
            let (llm_h, llm_w) = (h / sms, w / sms);

            // rope/pos in merge-grouped order (`rot_pos_emb`), repeated over t frames.
            for _f in 0..t {
                for br in 0..llm_h {
                    for bc in 0..llm_w {
                        for ir in 0..sms {
                            for ic in 0..sms {
                                let hpos = (br * sms + ir) as f32;
                                let wpos = (bc * sms + ic) as f32;
                                for &f in &inv {
                                    rope_rows.push(hpos * f);
                                }
                                for &f in &inv {
                                    rope_rows.push(wpos * f);
                                }
                            }
                        }
                    }
                }
            }

            // cu_seqlens (full): h*w patches per frame.
            for _f in 0..t {
                let last = *cu_seqlens.last().unwrap();
                cu_seqlens.push(last + h * w);
            }

            // get_window_index: window-partitioned valid merge-unit indices + cu_window boundaries.
            let pad_h = vmws - llm_h % vmws; // can equal vmws when divisible (harmless 0-count window)
            let pad_w = vmws - llm_w % vmws;
            let nwh = (llm_h + pad_h) / vmws;
            let nww = (llm_w + pad_w) / vmws;
            let mut cu_prev = *cu_window.last().unwrap();
            for f in 0..t {
                for wh in 0..nwh {
                    for ww in 0..nww {
                        let mut count = 0;
                        for ir in 0..vmws {
                            for ic in 0..vmws {
                                let r = wh * vmws + ir;
                                let cc = ww * vmws + ic;
                                if r < llm_h && cc < llm_w {
                                    let val = f * llm_h * llm_w + r * llm_w + cc;
                                    window_index.push(val + window_index_id);
                                    count += 1;
                                }
                            }
                        }
                        cu_prev += count * mu; // cumsum(seqlens)*merge_unit + offset
                        cu_window.push(cu_prev);
                    }
                }
            }
            window_index_id += t * llm_h * llm_w;
        }

        let merged = window_index.len() as i32;
        let seq = merged * mu;
        let cu_window = dedup_consecutive(&cu_window);

        // inverse permutation (`argsort(window_index)` for a permutation).
        let mut reverse = vec![0i32; merged as usize];
        for (i, &wi) in window_index.iter().enumerate() {
            reverse[wi as usize] = i as i32;
        }

        Ok(Plan {
            seq,
            merged,
            window_index: Array::from_slice(&window_index, &[merged]),
            reverse_index: Array::from_slice(&reverse, &[merged]),
            rope: Array::from_slice(&rope_rows, &[seq, rd as i32]),
            full_bid: block_ids(&cu_seqlens, seq),
            win_bid: block_ids(&cu_window, seq),
        })
    }

    /// Encode packed patches → ViT tokens. `pixel_values` is `[sum_patches, in·t·ph·pw]`; `grid_thw`
    /// rows are `[t, h, w]` (patches). Returns `[sum_merged, out_hidden]` in the original (un-windowed)
    /// merge-unit order, where `sum_merged = Σ t·(h/merge)·(w/merge)`.
    pub fn forward(&self, pixel_values: &Array, grid_thw: &[[i32; 3]]) -> Result<Array> {
        let c = &self.cfg;
        let mu = c.merge_unit();
        let dim = c.hidden_size;
        let nh = c.num_heads;
        let rd = c.head_dim() / 2;

        let plan = self.build_plan(grid_thw)?;
        let (seq, merged) = (plan.seq, plan.merged);

        // Patch embed, then reorder hidden + rope by the window permutation (merge-unit granularity).
        let h = self.patch_embed.forward(pixel_values)?; // [seq, dim]
        let h = h
            .reshape(&[merged, mu, dim])?
            .take_axis(&plan.window_index, 0)?
            .reshape(&[seq, dim])?;
        let rope = plan
            .rope
            .reshape(&[merged, mu, rd])?
            .take_axis(&plan.window_index, 0)?
            .reshape(&[seq, rd])?;
        let emb = concatenate_axis(&[&rope, &rope], 1)?; // [seq, head_dim] f32
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        let full_mask = additive_mask(&plan.full_bid, seq);
        let win_mask = additive_mask(&plan.win_bid, seq);

        let mut h = h;
        for (i, blk) in self.blocks.iter().enumerate() {
            let mask = if c.fullatt_block_indexes.contains(&(i as i32)) {
                &full_mask
            } else {
                &win_mask
            };
            h = blk.forward(&h, &cos, &sin, mask, nh)?;
        }

        // Merge + undo the window permutation.
        let h = self.merger.forward(&h, merged, dim * mu)?;
        Ok(h.take_axis(&plan.reverse_index, 0)?)
    }
}

/// `get_vit_features`: split the concatenated tower output `[Σ merged, out_hidden]` back into one
/// `[merged, out_hidden]` chunk per grid, by `t·h·w / merge²` (the reference's
/// `torch.split(image_embeds, grid.prod(-1) // merge²)`).
pub fn split_vit_features(embeds: &Array, grids: &[[i32; 3]], merge: i32) -> Result<Vec<Array>> {
    let m2 = merge * merge;
    let sizes: Vec<i32> = grids.iter().map(|g| g[0] * g[1] * g[2] / m2).collect();
    if sizes.len() <= 1 {
        return Ok(vec![embeds.clone()]);
    }
    let mut pts = Vec::with_capacity(sizes.len() - 1);
    let mut acc = 0;
    for s in &sizes[..sizes.len() - 1] {
        acc += s;
        pts.push(acc);
    }
    Ok(split_sections(embeds, &pts, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// split_vit_features chunks by per-grid merged-token count.
    #[test]
    fn vit_feature_split() {
        // two grids: (1,4,6)->6 merged, (1,4,4)->4 merged; total 10 rows.
        let embeds = Array::zeros::<f32>(&[10, 8]).unwrap();
        let chunks = split_vit_features(&embeds, &[[1, 4, 6], [1, 4, 4]], 2).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[6, 8]);
        assert_eq!(chunks[1].shape(), &[4, 8]);
    }

    /// `vit_merger_window_size = window // merge // patch` and `head_dim` / `merge_unit` derivations
    /// match Qwen2.5-VL-7B.
    #[test]
    fn config_derivations() {
        let c = VisionConfig::default();
        assert_eq!(c.head_dim(), 80);
        assert_eq!(c.merge_unit(), 4);
        assert_eq!(c.vit_merger_window_size(), 4); // 112 / 2 / 14
    }

    /// rotate_half is the NeoX half-split: `[a,b,c,d] → [-c,-d,a,b]`.
    #[test]
    fn rotate_half_neox() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let r = rotate_half(&x).unwrap();
        let got: Vec<f32> = r.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    /// dedup_consecutive collapses runs (mirrors `torch.unique_consecutive`).
    #[test]
    fn dedup_runs() {
        assert_eq!(
            dedup_consecutive(&[0, 16, 24, 32, 36, 36, 36]),
            vec![0, 16, 24, 32, 36]
        );
        assert_eq!(dedup_consecutive(&[0, 0, 5]), vec![0, 5]);
    }

    /// block_ids partitions positions into the diagonal blocks named by `cu`.
    #[test]
    fn block_ids_partition() {
        // two images: [0,4) and [4,7).
        assert_eq!(block_ids(&[0, 4, 7], 7), vec![0, 0, 0, 0, 1, 1, 1]);
    }
}
