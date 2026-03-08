//! Gemenr core library types and traits.

/// Agent loop components such as tool dispatch strategies.
pub mod agent;
/// Configuration loading for model providers.
pub mod config;
/// Context persistence and reconstruction primitives.
pub mod context;
/// Structured errors for configuration and model interactions.
pub mod error;
/// Core chat message types.
pub mod message;
/// Model provider abstractions and request/response types.
pub mod model;
/// Runtime protocol types.
pub mod protocol;
/// Tool invocation abstraction shared with the runtime.
pub mod tool_invoker;
/// Tool specification types shared across providers and tools.
pub mod tool_spec;

/// Re-export of the application configuration type.
pub use agent::ConversationMessage;
/// Re-export of the native tool dispatcher.
pub use agent::NativeToolDispatcher;
/// Re-export of parsed tool call values.
pub use agent::ParsedToolCall;
/// Re-export of the shared tool dispatcher trait.
pub use agent::ToolDispatcher;
/// Re-export of tool execution result values.
pub use agent::ToolExecutionResult;
/// Re-export of the XML tool dispatcher.
pub use agent::XmlToolDispatcher;
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
/// Re-export of anchor checkpoint entries.
pub use context::AnchorEntry;
/// Re-export of the in-memory tape store implementation.
pub use context::InMemoryTapeStore;
/// Re-export of the JSONL tape store implementation.
pub use context::JsonlTapeStore;
/// Re-export of SOUL.md management errors.
pub use context::SoulError;
/// Re-export of the SOUL.md manager.
pub use context::SoulManager;
/// Re-export of tape storage errors.
pub use context::TapeError;
/// Re-export of the tape store abstraction.
pub use context::TapeStore;
/// Re-export of model interaction errors.
pub use error::ModelError;
/// Re-export of the core chat message type.
pub use message::ChatMessage;
/// Re-export of chat participant roles.
pub use message::ChatRole;
/// Re-export of structured chat requests.
pub use model::ChatRequest;
/// Re-export of structured chat responses.
pub use model::ChatResponse;
/// Re-export of model completion finish reasons.
pub use model::FinishReason;
/// Re-export of model provider capability declarations.
pub use model::ModelCapabilities;
/// Re-export of the model provider trait.
pub use model::ModelProvider;
/// Re-export of model completion requests.
pub use model::ModelRequest;
/// Re-export of model completion responses.
pub use model::ModelResponse;
/// Re-export of model router abstraction.
pub use model::ModelRouter;
/// Re-export of model token usage statistics.
pub use model::TokenUsage;
/// Re-export of structured model tool calls.
pub use model::ToolCall;
/// Re-export of provider-specific tool payloads.
pub use model::ToolsPayload;
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
/// Re-export of tool policy decisions.
pub use tool_invoker::PolicyDecision;
/// Re-export of tool invocation errors.
pub use tool_invoker::ToolInvokeError;
/// Re-export of successful tool invocation results.
pub use tool_invoker::ToolInvokeResult;
/// Re-export of the tool invocation trait.
pub use tool_invoker::ToolInvoker;
/// Re-export of tool risk levels.
pub use tool_spec::RiskLevel;
/// Re-export of unified tool specifications.
pub use tool_spec::ToolSpec;
