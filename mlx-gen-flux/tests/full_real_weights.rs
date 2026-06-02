//! Ignored real-weights checks for assembling the full FLUX.1 provider.
//!
//! Run intentionally with a local FLUX.1 snapshot:
//!
//! ```text
//! MLX_GEN_FLUX_SNAPSHOT=/path/to/FLUX.1-schnell/snapshot \
//!   cargo test -p mlx-gen-flux --test full_real_weights -- --ignored
//! ```

use std::path::PathBuf;

use mlx_gen::{LoadSpec, WeightsSource};
use mlx_gen_flux as _;

#[test]
#[ignore = "loads the full multi-component FLUX.1 snapshot; set MLX_GEN_FLUX_SNAPSHOT"]
fn flux1_provider_loads_full_snapshot() {
    let root = PathBuf::from(
        std::env::var("MLX_GEN_FLUX_SNAPSHOT")
            .expect("set MLX_GEN_FLUX_SNAPSHOT to a FLUX.1 snapshot directory"),
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let model = mlx_gen::load("flux1_schnell", &spec).unwrap();
    assert_eq!(model.descriptor().id, "flux1_schnell");
}
