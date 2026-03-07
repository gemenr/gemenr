//! Gemenr core library types and traits.

/// Configuration loading for model providers.
pub mod config;
/// Structured errors for configuration and model interactions.
pub mod error;
/// Core chat message types.
pub mod message;
/// Model provider abstractions and request/response types.
pub mod model;

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
