//! Gemenr core library types and traits.

/// Core chat message types.
pub mod message;

/// Re-export of the core chat message type.
pub use message::ChatMessage;
/// Re-export of chat participant roles.
pub use message::ChatRole;
