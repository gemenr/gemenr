//! Context persistence primitives for runtime state reconstruction.

pub mod soul;
pub mod tape;

pub use soul::{SoulError, SoulManager};
pub use tape::{AnchorEntry, InMemoryTapeStore, JsonlTapeStore, TapeError, TapeStore};
