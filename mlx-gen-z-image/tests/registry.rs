//! Proves the architecture's central claim (docs/MODEL_ARCHITECTURE.md §4): linking the
//! provider crate self-registers Z-Image into `mlx-gen`'s link-time `inventory` registry — the
//! core has no central match to edit — so `mlx_gen::load("z_image_turbo", …)` resolves across
//! the crate boundary. This is the Rust stand-in for a DI container's resolve-by-id.
//!
//! NOTE: a provider must actually be *linked* into the consumer for its `inventory::submit!` to
//! take effect — a dependency that is declared but never referenced can have its link-section
//! statics dropped by the linker. The `use … as _` below forces the link (the SceneWorks worker
//! references every provider it serves, so this is automatic there). This is the "DI container
//! must know about the assembly" detail.

use mlx_gen::{LoadSpec, WeightsSource};
use mlx_gen_z_image as _;

#[test]
fn z_image_turbo_resolves_through_core_registry() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let g = mlx_gen::load("z_image_turbo", &spec).expect("provider self-registered via inventory");
    assert_eq!(g.descriptor().id, "z_image_turbo");
    assert_eq!(g.descriptor().family, "z-image");
}

#[test]
fn z_image_turbo_visible_in_registry_iteration() {
    assert!(mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "z_image_turbo"));
}

#[test]
fn unknown_id_still_errors() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    assert!(mlx_gen::load("not_a_model", &spec).is_err());
}
