//! replace_person mask parity vs Pillow (epic 3040 / sc-3053). Gates `apply_replacement_mask` — the
//! port of the worker's `_apply_replacement_mask` (gray-118 neutralization that builds the masked
//! control clip the IC-LoRA keyframe-append regenerates) — byte-for-byte against the actual PIL
//! `convert("L")` → `point(int(v·s))` → `composite` (`tools/dump_ltx_replace_mask_golden.py`).
//! Pure pixel op, no weights → runs without `--ignored`.

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_ltx::apply_replacement_mask;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_replace_mask_golden.safetensors"
);

/// Load an (H,W,3) uint8 tensor from the golden into an `Image`.
fn image_from(g: &Weights, key: &str) -> Image {
    let a = g.require(key).unwrap();
    let sh = a.shape(); // (H, W, 3)
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let pixels = a
        .as_dtype(mlx_rs::Dtype::Uint8)
        .unwrap()
        .as_slice::<u8>()
        .to_vec();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

#[test]
fn replace_mask_matches_pillow() {
    let g = Weights::from_file(GOLDEN)
        .expect("replace-mask golden (run tools/dump_ltx_replace_mask_golden.py)");
    let frame = image_from(&g, "frame");
    let mask = image_from(&g, "mask");
    for (tag, strength) in [("s100", 1.0_f32), ("s060", 0.6), ("s000", 0.0)] {
        let out = apply_replacement_mask(&frame, &mask, strength).unwrap();
        let want = image_from(&g, &format!("{tag}_out"));
        assert_eq!(out.pixels, want.pixels, "{tag}: byte mismatch vs Pillow");
    }
    // Sanity: strength 0 is a passthrough of the original frame.
    let s0 = apply_replacement_mask(&frame, &mask, 0.0).unwrap();
    assert_eq!(s0.pixels, frame.pixels);
}
