//! Runtime kernel — agent loop, prompt composition, and turn control.

pub mod prompt;
pub mod turn;

pub use prompt::PromptComposer;
pub use turn::{ActionDecision, TurnController};
