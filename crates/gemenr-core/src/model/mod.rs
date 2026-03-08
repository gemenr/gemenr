mod anthropic;

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::ConfigError;
use crate::error::ModelError;
use crate::message::ChatMessage;
use crate::tool_spec::ToolSpec;

/// Re-export of the Anthropic model provider implementation.
pub use anthropic::AnthropicProvider;

/// Reason why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Model naturally completed its response.
    Stop,
    /// Response was truncated due to max token limit.
    MaxTokens,
}

/// Request to send to a model provider.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// The conversation messages to send.
    pub messages: Vec<ChatMessage>,
    /// Model identifier (for example, `claude-haiku-4-5-20251001`).
    pub model: String,
    /// Maximum tokens to generate. `None` means provider default.
    pub max_tokens: Option<u32>,
}

/// Complete response from a model provider.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// The generated text content.
    pub content: String,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
}

/// Capabilities declared by a model provider.
///
/// Used by the runtime to automatically select tool dispatch strategy and adapt
/// request formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelCapabilities {
    /// Whether the provider supports native tool calling via API.
    pub native_tool_calling: bool,
    /// Whether the provider supports vision or image input.
    pub vision: bool,
}

/// Provider-specific tool definition payloads.
///
/// Different providers need different JSON structures for tool registration.
/// This enum keeps those differences behind a unified abstraction.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolsPayload {
    /// OpenAI Chat Completions function-calling format.
    OpenAI { tools: Vec<serde_json::Value> },
    /// Anthropic Messages API tool format.
    Anthropic { tools: Vec<serde_json::Value> },
    /// Prompt-guided fallback where tool descriptions are injected as text.
    PromptGuided { instructions: String },
}

/// Structured chat request used by the agent loop.
///
/// This extends [`ModelRequest`] with optional tool definitions for providers
/// that support native tool calling.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Model identifier.
    pub model: String,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Optional tool definitions for native tool-calling providers.
    pub tools: Option<Vec<ToolSpec>>,
}

/// Structured chat response supporting both text and tool calls.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    /// Text content of the response, if any.
    pub text: Option<String>,
    /// Tool calls extracted from the response.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage statistics, if the provider reports them.
    pub usage: Option<TokenUsage>,
}

/// A structured tool call returned by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments encoded as a JSON string.
    pub arguments: String,
}

/// Token usage statistics from a model response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input or prompt tokens.
    pub input_tokens: u32,
    /// Number of output or completion tokens.
    pub output_tokens: u32,
}

/// Unified interface for LLM providers.
///
/// Implementations must return a complete (non-streaming) response.
/// Internally, a provider may use streaming APIs but must accumulate the full
/// response before returning from [`ModelProvider::complete`].
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Send a request to the model and return the complete response.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError>;

    /// Declare provider capabilities.
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::default()
    }

    /// Convenience helper for checking native tool-calling support.
    fn supports_native_tools(&self) -> bool {
        self.capabilities().native_tool_calling
    }

    /// Convert unified tool definitions into a provider-native format.
    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        ToolsPayload::PromptGuided {
            instructions: build_tool_instructions_text(tools),
        }
    }

    /// Structured chat interface with optional tool support.
    ///
    /// The default implementation delegates to [`ModelProvider::complete`] and
    /// ignores any provided tools.
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let model_request = ModelRequest {
            messages: request.messages,
            model: request.model,
            max_tokens: request.max_tokens,
        };
        let response = self.complete(model_request).await?;

        Ok(ChatResponse {
            text: Some(response.content),
            tool_calls: Vec::new(),
            usage: None,
        })
    }
}

/// Shared handle to a model provider.
pub type SharedModelProvider = Arc<dyn ModelProvider>;

/// Convert optional unified tools into a provider-specific payload.
#[must_use]
pub fn convert_request_tools(
    provider: &dyn ModelProvider,
    tools: Option<&[ToolSpec]>,
) -> Option<ToolsPayload> {
    tools
        .filter(|tools| !tools.is_empty())
        .map(|tools| provider.convert_tools(tools))
}

/// Build a text description of tools for prompt-guided tool calling.
///
/// Providers that do not support native tool calling use this helper to inject
/// tool descriptions into the prompt as plain text.
#[must_use]
pub fn build_tool_instructions_text(tools: &[ToolSpec]) -> String {
    let mut instructions = String::from("## Available Tools\n\n");

    if tools.is_empty() {
        instructions.push_str("No tools are currently available.\n");
        return instructions;
    }

    for tool in tools {
        let schema =
            serde_json::to_string_pretty(&tool.input_schema).unwrap_or_else(|_| "{}".to_string());
        let _ = writeln!(&mut instructions, "Name: {}", tool.name);
        let _ = writeln!(&mut instructions, "Description: {}", tool.description);
        let _ = writeln!(&mut instructions, "Risk: {:?}", tool.risk_level);
        let _ = writeln!(&mut instructions, "Input schema:\n{schema}");
        instructions.push('\n');
    }

    instructions
}

/// Selects provider fallback order for retryable model failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackPlan {
    /// Primary provider key.
    pub primary: String,
    /// Backup provider keys in failover order.
    pub backups: Vec<String>,
}

/// Routes model requests to the appropriate provider.
///
/// Phase 1 uses a single default provider, but the router keeps provider
/// selection out of the runtime so future multi-provider support stays
/// backward compatible.
#[derive(Clone)]
pub struct ModelRouter {
    providers: HashMap<String, SharedModelProvider>,
    default: String,
    fallback_plan: Option<FallbackPlan>,
}

impl ModelRouter {
    /// Create a new router with a default provider.
    #[must_use]
    pub fn new(name: String, provider: SharedModelProvider) -> Self {
        let mut providers = HashMap::new();
        providers.insert(name.clone(), provider);

        Self {
            providers,
            default: name,
            fallback_plan: None,
        }
    }

    /// Add an additional provider.
    pub fn add_provider(&mut self, name: String, provider: SharedModelProvider) {
        self.providers.insert(name, provider);
    }

    /// Configure provider fallback order.
    pub fn set_fallback_plan(&mut self, plan: FallbackPlan) -> Result<(), ConfigError> {
        self.validate_fallback_plan(&plan)?;
        self.fallback_plan = Some(plan);
        Ok(())
    }

    /// Validate that a fallback plan only references known providers and keeps
    /// native tool support consistent across the group.
    pub fn validate_fallback_plan(&self, plan: &FallbackPlan) -> Result<(), ConfigError> {
        let Some(primary) = self.providers.get(&plan.primary) else {
            return Err(ConfigError::Invalid(format!(
                "fallback.primary references unknown provider `{}`",
                plan.primary
            )));
        };

        let primary_supports_native_tools = primary.supports_native_tools();
        for backup in &plan.backups {
            let Some(provider) = self.providers.get(backup) else {
                return Err(ConfigError::Invalid(format!(
                    "fallback.backups references unknown provider `{backup}`"
                )));
            };
            if provider.supports_native_tools() != primary_supports_native_tools {
                return Err(ConfigError::Invalid(format!(
                    "fallback provider `{backup}` must share supports_native_tools() with primary `{}`",
                    plan.primary
                )));
            }
        }

        Ok(())
    }

    /// Get a shared handle to the default provider.
    #[must_use]
    pub fn default_provider(&self) -> SharedModelProvider {
        self.provider(&self.default)
            .expect("default provider must exist in router")
    }

    /// Get a shared handle to a provider by name.
    #[must_use]
    pub fn provider(&self, name: &str) -> Option<SharedModelProvider> {
        self.providers.get(name).cloned()
    }

    /// Run a structured chat request with configured fallback behavior.
    pub async fn chat_with_fallback(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, ModelError> {
        let provider_names = self.provider_order();
        let mut last_error = None;

        for (index, provider_name) in provider_names.iter().enumerate() {
            let provider = self
                .provider(provider_name)
                .expect("provider in order must be registered");
            match provider.chat(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) if is_retryable(&error) && index + 1 < provider_names.len() => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }

        Err(last_error.expect("provider order should contain at least one provider"))
    }

    async fn complete_with_fallback(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let provider_names = self.provider_order();
        let mut last_error = None;

        for (index, provider_name) in provider_names.iter().enumerate() {
            let provider = self
                .provider(provider_name)
                .expect("provider in order must be registered");
            match provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) if is_retryable(&error) && index + 1 < provider_names.len() => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }

        Err(last_error.expect("provider order should contain at least one provider"))
    }

    fn provider_order(&self) -> Vec<String> {
        if let Some(plan) = &self.fallback_plan {
            let mut names = Vec::with_capacity(plan.backups.len() + 1);
            names.push(plan.primary.clone());
            names.extend(plan.backups.iter().cloned());
            names
        } else {
            vec![self.default.clone()]
        }
    }
}

#[async_trait]
impl ModelProvider for ModelRouter {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
        self.complete_with_fallback(request).await
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.default_provider().capabilities()
    }

    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        self.default_provider().convert_tools(tools)
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.chat_with_fallback(request).await
    }
}

fn is_retryable(error: &ModelError) -> bool {
    matches!(
        error,
        ModelError::Timeout | ModelError::RateLimit { .. } | ModelError::Network(_)
    )
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use super::{
        ChatRequest, ChatResponse, FallbackPlan, FinishReason, ModelCapabilities, ModelProvider,
        ModelRequest, ModelResponse, ModelRouter, SharedModelProvider, TokenUsage, ToolCall,
        build_tool_instructions_text,
    };
    use crate::message::ChatMessage;
    use crate::tool_spec::{RiskLevel, ToolSpec};
    use crate::{ConfigError, ModelError};

    struct DummyProvider {
        response_text: &'static str,
        native_tool_calling: bool,
    }

    struct ScriptedChatProvider {
        chat_results: Mutex<VecDeque<Result<ChatResponse, crate::ModelError>>>,
        native_tool_calling: bool,
    }

    impl ScriptedChatProvider {
        fn new(
            chat_results: Vec<Result<ChatResponse, crate::ModelError>>,
            native_tool_calling: bool,
        ) -> Self {
            Self {
                chat_results: Mutex::new(chat_results.into()),
                native_tool_calling,
            }
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for ScriptedChatProvider {
        async fn complete(
            &self,
            _request: ModelRequest,
        ) -> Result<ModelResponse, crate::ModelError> {
            Ok(ModelResponse {
                content: "complete".to_string(),
                finish_reason: FinishReason::Stop,
            })
        }

        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, crate::ModelError> {
            self.chat_results
                .lock()
                .expect("chat results lock should not be poisoned")
                .pop_front()
                .expect("scripted chat result should exist")
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                native_tool_calling: self.native_tool_calling,
                vision: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for DummyProvider {
        async fn complete(
            &self,
            request: ModelRequest,
        ) -> Result<ModelResponse, crate::ModelError> {
            let content = if self.response_text.is_empty() {
                format!("received {} messages", request.messages.len())
            } else {
                self.response_text.to_string()
            };

            Ok(ModelResponse {
                content,
                finish_reason: FinishReason::Stop,
            })
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                native_tool_calling: self.native_tool_calling,
                vision: false,
            }
        }
    }

    #[test]
    fn chat_request_can_include_tools() {
        let request = ChatRequest {
            messages: vec![ChatMessage::user("inspect workspace")],
            model: "claude-haiku-4-5-20251001".to_string(),
            max_tokens: Some(512),
            tools: Some(vec![ToolSpec {
                name: "shell".to_string(),
                description: "Run a shell command".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
                risk_level: RiskLevel::High,
            }]),
        };

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.tools.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn chat_response_can_have_empty_tool_calls() {
        let response = ChatResponse {
            text: Some("All done".to_string()),
            tool_calls: Vec::new(),
            usage: Some(TokenUsage {
                input_tokens: 128,
                output_tokens: 64,
            }),
        };

        assert_eq!(response.text.as_deref(), Some("All done"));
        assert!(response.tool_calls.is_empty());
        assert_eq!(
            response.usage.expect("usage should exist").output_tokens,
            64
        );
    }

    #[test]
    fn tool_call_and_usage_round_trip_through_json() {
        let tool_call = ToolCall {
            id: "tool-1".to_string(),
            name: "shell".to_string(),
            arguments: "{\"command\":\"pwd\"}".to_string(),
        };
        let usage = TokenUsage {
            input_tokens: 200,
            output_tokens: 50,
        };

        let tool_call_json = serde_json::to_string(&tool_call).expect("tool call should serialize");
        let usage_json = serde_json::to_string(&usage).expect("usage should serialize");

        assert_eq!(
            serde_json::from_str::<ToolCall>(&tool_call_json)
                .expect("tool call should deserialize"),
            tool_call
        );
        assert_eq!(
            serde_json::from_str::<TokenUsage>(&usage_json).expect("usage should deserialize"),
            usage
        );
    }

    #[tokio::test]
    async fn default_chat_delegates_to_complete() {
        let provider = DummyProvider {
            response_text: "delegated",
            native_tool_calling: false,
        };

        let response = provider
            .chat(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: Some(64),
                tools: None,
            })
            .await
            .expect("chat should succeed");

        assert_eq!(response.text.as_deref(), Some("delegated"));
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn build_tool_instructions_text_includes_tool_details() {
        let tools = vec![ToolSpec {
            name: "shell".to_string(),
            description: "Execute shell commands".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
            risk_level: RiskLevel::High,
        }];

        let instructions = build_tool_instructions_text(&tools);

        assert!(instructions.contains("## Available Tools"));
        assert!(instructions.contains("Name: shell"));
        assert!(instructions.contains("Description: Execute shell commands"));
        assert!(instructions.contains("Risk: High"));
        assert!(instructions.contains("command"));
    }

    #[test]
    fn model_provider_trait_remains_object_safe() {
        fn accept_provider(_provider: &dyn ModelProvider) {}

        let provider = DummyProvider {
            response_text: "ok",
            native_tool_calling: false,
        };

        accept_provider(&provider);
    }

    #[tokio::test]
    async fn model_router_returns_default_provider() {
        let primary: SharedModelProvider = Arc::new(DummyProvider {
            response_text: "primary",
            native_tool_calling: false,
        });
        let router = ModelRouter::new("primary".to_string(), primary.clone());

        assert!(Arc::ptr_eq(&router.default_provider(), &primary));
        assert_eq!(
            router
                .default_provider()
                .complete(ModelRequest {
                    messages: vec![ChatMessage::user("hello")],
                    model: "claude-haiku-4-5-20251001".to_string(),
                    max_tokens: Some(64),
                })
                .await
                .expect("request should succeed")
                .content,
            "primary"
        );
    }

    #[test]
    fn model_router_can_lookup_named_providers() {
        let primary: SharedModelProvider = Arc::new(DummyProvider {
            response_text: "primary",
            native_tool_calling: false,
        });
        let backup: SharedModelProvider = Arc::new(DummyProvider {
            response_text: "backup",
            native_tool_calling: false,
        });
        let mut router = ModelRouter::new("primary".to_string(), primary);
        router.add_provider("backup".to_string(), backup.clone());

        assert!(router.provider("missing").is_none());
        assert!(Arc::ptr_eq(
            &router
                .provider("backup")
                .expect("backup provider should exist"),
            &backup
        ));
    }

    #[tokio::test]
    async fn timeout_errors_fall_back_to_backup_provider() {
        let mut router = ModelRouter::new(
            "primary".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Err(ModelError::Timeout)],
                false,
            )),
        );
        router.add_provider(
            "backup".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Ok(ChatResponse {
                    text: Some("backup".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                })],
                false,
            )),
        );
        router
            .set_fallback_plan(FallbackPlan {
                primary: "primary".to_string(),
                backups: vec!["backup".to_string()],
            })
            .expect("fallback plan should be valid");

        let response = router
            .chat_with_fallback(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                model: "test-model".to_string(),
                max_tokens: None,
                tools: None,
            })
            .await
            .expect("backup should succeed");

        assert_eq!(response.text.as_deref(), Some("backup"));
    }

    #[tokio::test]
    async fn rate_limit_errors_fall_back_to_backup_provider() {
        let mut router = ModelRouter::new(
            "primary".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Err(ModelError::RateLimit { retry_after: None })],
                false,
            )),
        );
        router.add_provider(
            "backup".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Ok(ChatResponse {
                    text: Some("backup".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                })],
                false,
            )),
        );
        router
            .set_fallback_plan(FallbackPlan {
                primary: "primary".to_string(),
                backups: vec!["backup".to_string()],
            })
            .expect("fallback plan should be valid");

        let response = router
            .chat_with_fallback(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                model: "test-model".to_string(),
                max_tokens: None,
                tools: None,
            })
            .await
            .expect("backup should succeed");

        assert_eq!(response.text.as_deref(), Some("backup"));
    }

    #[tokio::test]
    async fn network_errors_fall_back_to_backup_provider() {
        let mut router = ModelRouter::new(
            "primary".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Err(ModelError::Network("down".to_string()))],
                false,
            )),
        );
        router.add_provider(
            "backup".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Ok(ChatResponse {
                    text: Some("backup".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                })],
                false,
            )),
        );
        router
            .set_fallback_plan(FallbackPlan {
                primary: "primary".to_string(),
                backups: vec!["backup".to_string()],
            })
            .expect("fallback plan should be valid");

        let response = router
            .chat_with_fallback(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                model: "test-model".to_string(),
                max_tokens: None,
                tools: None,
            })
            .await
            .expect("backup should succeed");

        assert_eq!(response.text.as_deref(), Some("backup"));
    }

    #[tokio::test]
    async fn auth_errors_do_not_fall_back() {
        let mut router = ModelRouter::new(
            "primary".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Err(ModelError::Auth("bad key".to_string()))],
                false,
            )),
        );
        router.add_provider(
            "backup".to_string(),
            Arc::new(ScriptedChatProvider::new(Vec::new(), false)),
        );
        router
            .set_fallback_plan(FallbackPlan {
                primary: "primary".to_string(),
                backups: vec!["backup".to_string()],
            })
            .expect("fallback plan should be valid");

        let error = router
            .chat_with_fallback(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                model: "test-model".to_string(),
                max_tokens: None,
                tools: None,
            })
            .await
            .expect_err("auth error should not fall back");

        assert!(matches!(error, ModelError::Auth(_)));
    }

    #[tokio::test]
    async fn api_errors_do_not_fall_back() {
        let mut router = ModelRouter::new(
            "primary".to_string(),
            Arc::new(ScriptedChatProvider::new(
                vec![Err(ModelError::Api {
                    status: 400,
                    message: "bad request".to_string(),
                })],
                false,
            )),
        );
        router.add_provider(
            "backup".to_string(),
            Arc::new(ScriptedChatProvider::new(Vec::new(), false)),
        );
        router
            .set_fallback_plan(FallbackPlan {
                primary: "primary".to_string(),
                backups: vec!["backup".to_string()],
            })
            .expect("fallback plan should be valid");

        let error = router
            .chat_with_fallback(ChatRequest {
                messages: vec![ChatMessage::user("hello")],
                model: "test-model".to_string(),
                max_tokens: None,
                tools: None,
            })
            .await
            .expect_err("api error should not fall back");

        assert!(matches!(error, ModelError::Api { .. }));
    }

    #[test]
    fn fallback_plan_rejects_mismatched_native_tool_capabilities() {
        let primary: SharedModelProvider = Arc::new(DummyProvider {
            response_text: "primary",
            native_tool_calling: true,
        });
        let backup: SharedModelProvider = Arc::new(DummyProvider {
            response_text: "backup",
            native_tool_calling: false,
        });
        let mut router = ModelRouter::new("primary".to_string(), primary);
        router.add_provider("backup".to_string(), backup);

        let error = router
            .set_fallback_plan(FallbackPlan {
                primary: "primary".to_string(),
                backups: vec!["backup".to_string()],
            })
            .expect_err("capability mismatch should fail");

        assert!(
            matches!(error, ConfigError::Invalid(message) if message.contains("supports_native_tools"))
        );
    }
}
