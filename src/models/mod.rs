//! Model families that still live in the core crate. Each mirrors the layout of the Python
//! mflux fork it ports from, so weight keys and parity fixtures line up 1:1.
//!
//! NOTE: Z-Image moved out to the `mlx-gen-z-image` provider crate (sc-2403 restructure);
//! Qwen is next in the migration order (z-image → qwen → others) and stays here until then.

pub mod qwen;
