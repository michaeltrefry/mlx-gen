//! Z-Image 3-axis RoPE table. Port of `RopeEmbedder`: per-axis precomputed cos/sin tables
//! (`freqs[j] = theta^(-2j/d)`, angle = pos·freq), gathered by 3-D position ids and
//! concatenated → `(N, Σ axes_dims / 2, 2)` (= `(N, head_dim/2, 2)`, the block's `freqs_cis`).
//! Weightless: fully determined by `theta` / `axes_dims` / `axes_lens`.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::Result;

pub struct RopeEmbedder {
    /// One `(axes_lens[i], axes_dims[i]/2, 2)` cos/sin table per axis.
    tables: Vec<Array>,
    n_axes: usize,
}

impl RopeEmbedder {
    pub fn new(theta: f32, axes_dims: &[i32], axes_lens: &[i32]) -> Self {
        let mut tables = Vec::with_capacity(axes_dims.len());
        for (&d, &e) in axes_dims.iter().zip(axes_lens) {
            let (e, half) = (e as usize, (d / 2) as usize);
            let mut data = vec![0f32; e * half * 2];
            for pos in 0..e {
                for j in 0..half {
                    let freq = 1.0 / theta.powf((2 * j) as f32 / d as f32);
                    let angle = pos as f32 * freq;
                    data[(pos * half + j) * 2] = angle.cos();
                    data[(pos * half + j) * 2 + 1] = angle.sin();
                }
            }
            tables.push(Array::from_slice(&data, &[e as i32, half as i32, 2]));
        }
        Self {
            n_axes: axes_dims.len(),
            tables,
        }
    }

    /// `ids`: `(N, n_axes)` int32 position ids → `(N, Σ axes_dims / 2, 2)`.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let n = ids.shape()[0] as usize;
        let flat = ids.as_slice::<i32>(); // row-major (N, n_axes)
        let mut parts = Vec::with_capacity(self.n_axes);
        for axis in 0..self.n_axes {
            let col: Vec<i32> = (0..n).map(|row| flat[row * self.n_axes + axis]).collect();
            let index = Array::from_slice(&col, &[n as i32]);
            parts.push(self.tables[axis].take_axis(&index, 0)?); // (N, half_i, 2)
        }
        let refs: Vec<&Array> = parts.iter().collect();
        Ok(concatenate_axis(&refs, 1)?)
    }
}
