//! Gemenr core library types and traits.

/// Configuration loading for model providers.
pub mod config;
/// Structured errors for configuration and model interactions.
pub mod error;
/// Core chat message types.
pub mod message;
/// Model provider abstractions and request/response types.
pub mod model;
/// Runtime protocol types.
pub mod protocol;

/// Re-export of the application configuration type.
pub use config::Config;
/// Re-export of configuration loading errors.
pub use config::ConfigError;
/// Re-export of selectable model definitions.
pub use config::ModelConfig;
/// Re-export of provider definitions.
pub use config::ProviderConfig;
/// Re-export of supported provider types.
pub use config::ProviderType;
/// Re-export of model interaction errors.
pub use error::ModelError;
/// Re-export of the core chat message type.
pub use message::ChatMessage;
/// Re-export of chat participant roles.
pub use message::ChatRole;
/// Re-export of model completion finish reasons.
pub use model::FinishReason;
/// Re-export of the model provider trait.
pub use model::ModelProvider;
/// Re-export of model completion requests.
pub use model::ModelRequest;
/// Re-export of model completion responses.
pub use model::ModelResponse;
/// Re-export of unified event envelopes.
pub use protocol::EventEnvelope;
/// Re-export of event identifiers.
pub use protocol::EventId;
/// Re-export of event kind categories.
pub use protocol::EventKind;
/// Re-export of structured message parts.
pub use protocol::MessagePart;
/// Re-export of runtime input operations.
pub use protocol::Op;
/// Re-export of session routing hints.
pub use protocol::SessionHint;
/// Re-export of session identifiers.
pub use protocol::SessionId;
/// Re-export of turn identifiers.
pub use protocol::TurnId;
