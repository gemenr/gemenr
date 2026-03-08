//! Gemenr core library types and traits.

/// Shared access-layer message and routing models.
pub mod access;
/// Agent loop components such as tool dispatch strategies.
pub mod agent;
/// Runtime builder for assembling agent runtimes.
pub mod builder;
/// Configuration loading for model providers.
pub mod config;
/// Context persistence and reconstruction primitives.
pub mod context;
/// Structured errors for configuration and model interactions.
pub mod error;
/// Runtime kernel components for prompt composition and turn decisions.
pub mod kernel;
/// Core chat message types.
pub mod message;
/// Model provider abstractions and request/response types.
pub mod model;
/// Runtime protocol types.
pub mod protocol;
/// Long-lived conversation runtime management.
pub mod runtime_manager;
/// Tool invocation abstraction shared with the runtime.
pub mod tool_invoker;
/// Tool specification types shared across providers and tools.
pub mod tool_spec;

/// Re-export of access adapter traits.
pub use access::AccessAdapter;
/// Re-export of access-layer errors.
pub use access::AccessError;
/// Re-export of normalized inbound access messages.
pub use access::AccessInbound;
/// Re-export of normalized outbound access messages.
pub use access::AccessOutbound;
/// Re-export of access router.
pub use access::AccessRouter;
/// Re-export of conversation driver trait.
pub use access::ConversationDriver;
/// Re-export of access-layer conversation identifiers.
pub use access::ConversationId;
/// Re-export of access-layer reply routes.
pub use access::ReplyRoute;
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
/// Re-export of runtime builder.
pub use builder::RuntimeBuilder;
/// Re-export of access-layer configuration.
pub use config::AccessConfig;
/// Re-export of access-layer composition-root configuration view.
pub use config::AccessConfigView;
/// Re-export of the application configuration type.
pub use config::Config;
/// Re-export of configuration loading errors.
pub use config::ConfigError;
/// Re-export of runtime composition-root configuration view.
pub use config::CoreRuntimeConfigView;
/// Re-export of cron job configuration.
pub use config::CronJobConfig;
/// Re-export of Lark access configuration.
pub use config::LarkConfig;
/// Re-export of MCP configuration.
pub use config::McpConfig;
/// Re-export of external MCP server definitions.
pub use config::McpServerConfig;
/// Re-export of selectable model definitions.
pub use config::ModelConfig;
/// Re-export of provider fallback configuration.
pub use config::ModelFallbackConfig;
/// Re-export of scoped policy configuration.
pub use config::PolicyConfig;
/// Re-export of configured policy effects.
pub use config::PolicyEffect;
/// Re-export of configured policy rules.
pub use config::PolicyRuleConfig;
/// Re-export of provider definitions.
pub use config::ProviderConfig;
/// Re-export of supported provider types.
pub use config::ProviderType;
/// Re-export of configured policy scopes.
pub use config::ScopedPolicyConfig;
/// Re-export of tool-plane composition-root configuration view.
pub use config::ToolingConfigView;
/// Re-export of anchor checkpoint entries.
pub use context::AnchorEntry;
/// Re-export of anchor identifiers.
pub use context::AnchorId;
/// Re-export of context build outcomes.
pub use context::ContextBuildResult;
/// Re-export of the session context manager.
pub use context::ContextManager;
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
/// Re-export of token budget configuration.
pub use context::TokenBudget;
/// Re-export of model interaction errors.
pub use error::ModelError;
/// Re-export of turn action decisions.
pub use kernel::ActionDecision;
/// Re-export of runtime errors.
pub use kernel::AgentError;
/// Re-export of agent runtime.
pub use kernel::AgentRuntime;
/// Re-export of typed approval decisions.
pub use kernel::ApprovalDecision;
/// Re-export of approval handling for risky tool execution.
pub use kernel::ApprovalHandler;
/// Re-export of typed approval requests.
pub use kernel::ApprovalRequest;
/// Re-export of the default deny-all approval handler.
pub use kernel::DenyAllApprovals;
/// Re-export of runtime event delivery interface.
pub use kernel::EventSink;
/// Re-export of the default no-op event sink.
pub use kernel::NoopEventSink;
/// Re-export of prompt composition helpers.
pub use kernel::PromptComposer;
/// Re-export of turn decision helpers.
pub use kernel::TurnController;
/// Re-export of structured runtime turn input.
pub use kernel::TurnInput;
/// Re-export of the core chat message type.
pub use message::ChatMessage;
/// Re-export of chat participant roles.
pub use message::ChatRole;
/// Re-export of structured chat requests.
pub use model::ChatRequest;
/// Re-export of structured chat responses.
pub use model::ChatResponse;
/// Re-export of provider fallback plans.
pub use model::FallbackPlan;
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
/// Re-export of shared request context used by model and tool execution.
pub use model::RequestContext;
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
/// Re-export of the runtime manager for long-lived conversations.
pub use runtime_manager::RuntimeManager;
/// Re-export of runtime manager errors.
pub use runtime_manager::RuntimeManagerError;
/// Re-export of tool authorization decisions.
pub use tool_invoker::AuthorizationDecision;
/// Re-export of final execution policies.
pub use tool_invoker::ExecutionPolicy;
/// Re-export of policy evaluation context.
pub use tool_invoker::PolicyContext;
/// Re-export of prepared tool calls that carry authorization results.
pub use tool_invoker::PreparedToolCall;
/// Re-export of sandbox selection kinds.
pub use tool_invoker::SandboxKind;
/// Re-export of tool authorization contract.
pub use tool_invoker::ToolAuthorizer;
/// Re-export of tool call request payloads.
pub use tool_invoker::ToolCallRequest;
/// Re-export of tool discovery contract.
pub use tool_invoker::ToolCatalog;
/// Re-export of tool execution contract.
pub use tool_invoker::ToolExecutor;
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
