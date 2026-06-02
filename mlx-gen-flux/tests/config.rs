use mlx_gen_flux::{
    build_linear_sigmas, image_seq_len, FluxTokenizerKind, FluxVariant, FLUX1_DEV_ID,
    FLUX1_SCHNELL_ID,
};

#[test]
fn variant_config_matches_mflux_flux1() {
    assert_eq!(FluxVariant::Schnell.id(), FLUX1_SCHNELL_ID);
    assert_eq!(
        FluxVariant::Schnell.hf_model(),
        "black-forest-labs/FLUX.1-schnell"
    );
    assert_eq!(FluxVariant::Schnell.default_steps(), 4);
    assert_eq!(FluxVariant::Schnell.max_sequence_length(), 256);
    assert!(!FluxVariant::Schnell.supports_guidance());
    assert!(!FluxVariant::Schnell.requires_sigma_shift());

    assert_eq!(FluxVariant::Dev.id(), FLUX1_DEV_ID);
    assert_eq!(FluxVariant::Dev.hf_model(), "black-forest-labs/FLUX.1-dev");
    assert_eq!(FluxVariant::Dev.default_steps(), 25);
    assert_eq!(FluxVariant::Dev.max_sequence_length(), 512);
    assert!(FluxVariant::Dev.supports_guidance());
    assert!(FluxVariant::Dev.requires_sigma_shift());
}

#[test]
fn tokenizer_contract_matches_mflux_flux_definition() {
    assert_eq!(FluxTokenizerKind::Clip.subdir(), "tokenizer");
    assert_eq!(FluxTokenizerKind::Clip.max_length(FluxVariant::Dev), 77);
    assert_eq!(FluxTokenizerKind::T5.subdir(), "tokenizer_2");
    assert_eq!(FluxTokenizerKind::T5.max_length(FluxVariant::Schnell), 256);
    assert_eq!(FluxTokenizerKind::T5.max_length(FluxVariant::Dev), 512);
}

#[test]
fn linear_sigmas_match_unshifted_schnell_shape() {
    let sigmas = build_linear_sigmas(4, 1024, 1024, false);
    assert_eq!(sigmas, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
}

#[test]
fn shifted_dev_sigmas_use_flux_linear_mu() {
    assert_eq!(image_seq_len(1024, 1024), 4096);
    let sigmas = build_linear_sigmas(4, 1024, 1024, true);
    // mflux LinearScheduler with base/max shift 0.5/1.15 and image_seq_len 4096.
    let expected = [1.0_f32, 0.904_531, 0.759_511, 0.512_844, 0.0];
    for (got, want) in sigmas.iter().zip(expected) {
        assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
    }
}
