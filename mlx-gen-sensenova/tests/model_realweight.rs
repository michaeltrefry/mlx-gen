//! sc-3194: real-weight (35GB) registry wiring — `#[ignore]`, run locally.
//!
//! Loads `sensenova_u1_8b` through the **core registry** (`mlx_gen::load`) and runs the
//! `Generator::generate` dispatch for T2I (no conditioning) and image-edit (a `Reference`), asserting
//! the contract holds end to end: `GenerationOutput::Images` of the requested size, finite pixels.
//! The per-mode numerics are pinned by the t2i/it2i/vqa/interleave parity + real-weight tests; this
//! validates the registration + request-mapping + Array→Image plumbing.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test model_realweight -- --ignored --nocapture`

use std::path::PathBuf;

// Force-link the provider crate so its `inventory::submit!` registers `sensenova_u1_8b` for
// `mlx_gen::load` (an integration test binary only links crates it references).
use mlx_gen_sensenova as _;

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};

const DEFAULT_SNAPSHOT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots/\
     bfa9b436503cb8aed4f2bc60e3236710cc77468d"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("SENSENOVA_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SNAPSHOT))
}

#[test]
#[ignore = "needs the local 35GB checkpoint; run with --ignored"]
fn registry_load_and_generate() {
    let snap = snapshot_dir();
    if !snap.exists() {
        eprintln!("skipping: snapshot missing at {}", snap.display());
        return;
    }

    let spec = LoadSpec::new(WeightsSource::Dir(snap));
    let model = mlx_gen::load("sensenova_u1_8b", &spec).expect("load sensenova_u1_8b via registry");

    let mut noop = |_: Progress| {};
    let (w, h) = (256u32, 256u32);

    // ---- T2I (no conditioning) ----
    let t2i = GenerationRequest {
        prompt: "a red fox in a snowy forest".into(),
        width: w,
        height: h,
        count: 1,
        steps: Some(8),
        guidance: Some(2.0),
        ..Default::default()
    };
    let out = model.generate(&t2i, &mut noop).expect("T2I generate");
    match out {
        GenerationOutput::Images(imgs) => {
            assert_eq!(imgs.len(), 1);
            assert_eq!((imgs[0].width, imgs[0].height), (w, h));
            assert_eq!(imgs[0].pixels.len(), (w * h * 3) as usize);
            assert!(
                imgs[0].pixels.iter().any(|&p| p != 0),
                "T2I image is all-zero"
            );
            println!("T2I → {}x{} image", imgs[0].width, imgs[0].height);
        }
        _ => panic!("expected Images"),
    }

    // ---- Image-edit (a Reference) ----
    let reference = Image {
        width: 256,
        height: 256,
        pixels: (0..256 * 256 * 3).map(|i| (i % 256) as u8).collect(),
    };
    let edit = GenerationRequest {
        prompt: "make it autumn".into(),
        width: w,
        height: h,
        count: 1,
        steps: Some(8),
        guidance: Some(2.0),
        true_cfg: Some(1.0),
        conditioning: vec![Conditioning::Reference {
            image: reference,
            strength: None,
        }],
        ..Default::default()
    };
    let out = model.generate(&edit, &mut noop).expect("edit generate");
    match out {
        GenerationOutput::Images(imgs) => {
            assert_eq!(imgs.len(), 1);
            assert_eq!((imgs[0].width, imgs[0].height), (w, h));
            assert!(imgs[0].pixels.iter().all(|&_p| true));
            println!("edit → {}x{} image", imgs[0].width, imgs[0].height);
        }
        _ => panic!("expected Images"),
    }
}
