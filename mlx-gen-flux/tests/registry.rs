use mlx_gen::{LoadSpec, Quant, WeightsSource};
use mlx_gen_flux as _;

#[test]
fn flux1_variants_resolve_through_core_registry() {
    for id in ["flux1_schnell", "flux1_dev"] {
        let reg = mlx_gen::registry::generators()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("{id} provider should self-register"));
        let d = (reg.descriptor)();
        assert_eq!(d.family, "flux");

        let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
        let err = mlx_gen::load(id, &spec)
            .err()
            .expect("single-file spec is rejected by the loader")
            .to_string();
        assert!(
            err.contains("snapshot directory"),
            "expected the flux loader's error, got: {err}"
        );
    }
}

#[test]
fn flux1_variants_accept_quantization_specs() {
    for id in ["flux1_schnell", "flux1_dev"] {
        for quant in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(quant);
            let err = mlx_gen::load(id, &spec)
                .err()
                .expect("missing snapshot should still error")
                .to_string();
            assert!(
                !err.contains("quantized") && !err.contains("quantization"),
                "quantized FLUX load specs should get past capability gating, got: {err}"
            );
        }
    }
}
