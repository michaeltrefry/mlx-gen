//! sc-3379: native OpenPose body-skeleton rasterizer — pixel-for-pixel parity vs cv2 `draw_bodypose`.
//!
//! `#[ignore]`d — needs the golden from `tools/dump_instantid_openpose_golden.py` (cv2 4.13 ground
//! truth). Run:
//!   cargo test -p mlx-gen-instantid --release --test instantid_openpose -- --ignored --nocapture
//!
//! Four cases (real gallery pose on a square 1024² canvas, the same on a non-square canvas, an
//! occluded-head pose, and a tiny 128²) — each compared byte-for-byte against the OpenCV output
//! (zero differing pixels required).

use mlx_gen::weights::Weights;
use mlx_gen_instantid::{draw_bodypose, BodyPoint, STICKWIDTH};
use mlx_rs::Dtype;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/instantid_openpose_golden.safetensors"
);

fn check_case(g: &Weights, name: &str) {
    let wh = g.require(&format!("{name}_wh")).unwrap();
    let wh = wh.as_dtype(Dtype::Int32).unwrap();
    let wh = wh.as_slice::<i32>();
    let (w, h) = (wh[0] as u32, wh[1] as u32);

    // kps golden is [18, 3] = (x, y, present); present=0 ⇒ None.
    let kps_arr = g.require(&format!("{name}_kps")).unwrap();
    let kps_arr = kps_arr.as_dtype(Dtype::Float32).unwrap();
    let kps_flat = kps_arr.as_slice::<f32>();
    let kps: Vec<BodyPoint> = kps_flat
        .chunks_exact(3)
        .map(|c| {
            if c[2] != 0.0 {
                Some((c[0] as f64, c[1] as f64))
            } else {
                None
            }
        })
        .collect();

    let golden = g.require(&format!("{name}_img")).unwrap();
    let golden = golden.as_dtype(Dtype::Uint8).unwrap();
    let golden = golden.as_slice::<u8>();

    let img = draw_bodypose(w, h, &kps, STICKWIDTH);
    assert_eq!(
        img.pixels.len(),
        golden.len(),
        "case {name}: buffer len {} != golden {}",
        img.pixels.len(),
        golden.len()
    );

    let mut diff = 0usize;
    let mut first: Option<(usize, u8, u8)> = None;
    for (i, (&a, &b)) in img.pixels.iter().zip(golden).enumerate() {
        if a != b {
            diff += 1;
            if first.is_none() {
                first = Some((i, a, b));
            }
        }
    }
    if diff != 0 {
        let (i, a, b) = first.unwrap();
        let (px, ch) = (i / 3, i % 3);
        let (yy, xx) = (px / w as usize, px % w as usize);
        panic!(
            "case {name} ({w}x{h}): {diff} differing bytes; first @ (x={xx},y={yy},ch={ch}) mine={a} golden={b}"
        );
    }
    println!(
        "case {name} ({w}x{h}): pixel-for-pixel match ({} bytes)",
        golden.len()
    );
}

#[test]
#[ignore = "needs the openpose golden (tools/dump_instantid_openpose_golden.py)"]
fn draw_bodypose_matches_opencv() {
    let g = Weights::from_file(GOLDEN).expect("openpose golden (run the dump script)");
    for name in [
        "square_1024",
        "nonsquare_768x1024",
        "occluded_head_1024",
        "tiny_128",
    ] {
        check_case(&g, name);
    }
}
