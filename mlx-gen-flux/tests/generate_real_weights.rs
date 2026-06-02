//! Ignored real-weights generation smoke for FLUX.1-schnell.
//!
//! This is intentionally tiny (256x256, one step) but still evaluates the full transformer and VAE.

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_flux as _;

#[test]
#[ignore = "evaluates the full FLUX.1 transformer/VAE; set MLX_GEN_FLUX_SNAPSHOT"]
fn flux1_schnell_generates_one_image_smoke() {
    let root = PathBuf::from(
        std::env::var("MLX_GEN_FLUX_SNAPSHOT")
            .expect("set MLX_GEN_FLUX_SNAPSHOT to a FLUX.1-schnell snapshot directory"),
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let model = mlx_gen::load("flux1_schnell", &spec).unwrap();
    let req = GenerationRequest {
        prompt: "a red fox".into(),
        width: 256,
        height: 256,
        steps: Some(1),
        seed: Some(7),
        ..Default::default()
    };
    let out = model.generate(&req, &mut |_| {}).unwrap();
    match out {
        GenerationOutput::Images(images) => {
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].width, 256);
            assert_eq!(images[0].height, 256);
            assert_eq!(images[0].pixels.len(), 256 * 256 * 3);
        }
        other => panic!("expected image output, got {other:?}"),
    }
}
