mod anthropic;

use std::collections::HashMap;
use std::fmt::Write;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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

/// Routes model requests to the appropriate provider.
///
/// Phase 1 uses a single default provider, but the router keeps provider
/// selection out of the runtime so future multi-provider support stays
/// backward compatible.
pub struct ModelRouter {
    providers: HashMap<String, Box<dyn ModelProvider>>,
    default: String,
}

impl ModelRouter {
    /// Create a new router with a default provider.
    #[must_use]
    pub fn new(name: String, provider: Box<dyn ModelProvider>) -> Self {
        let mut providers = HashMap::new();
        providers.insert(name.clone(), provider);

        Self {
            providers,
            default: name,
        }
    }

    /// Add an additional provider.
    pub fn add_provider(&mut self, name: String, provider: Box<dyn ModelProvider>) {
        self.providers.insert(name, provider);
    }

    /// Get the default provider.
    #[must_use]
    pub fn default_provider(&self) -> &dyn ModelProvider {
        self.provider(&self.default)
            .expect("default provider must exist in router")
    }

    /// Get a provider by name.
    #[must_use]
    pub fn provider(&self, name: &str) -> Option<&dyn ModelProvider> {
        self.providers.get(name).map(Box::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
        ModelResponse, ModelRouter, TokenUsage, ToolCall, ToolsPayload,
        build_tool_instructions_text,
    };
    use crate::message::ChatMessage;
    use crate::tool_spec::{RiskLevel, ToolSpec};

    struct DummyProvider {
        response_text: &'static str,
        native_tool_calling: bool,
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
    fn model_request_construction_preserves_messages_and_parameters() {
        let request = ModelRequest {
            messages: vec![
                ChatMessage::system("Be concise."),
                ChatMessage::user("Hello!"),
            ],
            model: "claude-haiku-4-5-20251001".to_string(),
            max_tokens: Some(256),
        };

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0], ChatMessage::system("Be concise."));
        assert_eq!(request.messages[1], ChatMessage::user("Hello!"));
        assert_eq!(request.model, "claude-haiku-4-5-20251001");
        assert_eq!(request.max_tokens, Some(256));
    }

    #[test]
    fn model_response_construction_preserves_fields() {
        let response = ModelResponse {
            content: "Hello there".to_string(),
            finish_reason: FinishReason::MaxTokens,
        };

        assert_eq!(response.content, "Hello there");
        assert_eq!(response.finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn model_capabilities_default_to_disabled_features() {
        let capabilities = ModelCapabilities::default();

        assert!(!capabilities.native_tool_calling);
        assert!(!capabilities.vision);
    }

    #[test]
    fn supports_native_tools_reflects_capabilities() {
        let provider = DummyProvider {
            response_text: "",
            native_tool_calling: true,
        };

        assert!(provider.supports_native_tools());
    }

    #[test]
    fn default_convert_tools_returns_prompt_guided_payload() {
        let provider = DummyProvider {
            response_text: "",
            native_tool_calling: false,
        };
        let tools = vec![ToolSpec {
            name: "shell".to_string(),
            description: "Execute shell commands".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
            risk_level: RiskLevel::High,
        }];

        let payload = provider.convert_tools(&tools);

        assert!(matches!(payload, ToolsPayload::PromptGuided { .. }));
    }

    #[tokio::test]
    async fn default_chat_delegates_to_complete() {
        let provider = DummyProvider {
            response_text: "delegated",
            native_tool_calling: false,
        };
        let response = provider
            .chat(ChatRequest {
                messages: vec![ChatMessage::user("Ping")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: Some(32),
                tools: Some(vec![ToolSpec {
                    name: "noop".to_string(),
                    description: "Do nothing".to_string(),
                    input_schema: serde_json::json!({"type": "object"}),
                    risk_level: RiskLevel::Low,
                }]),
            })
            .await
            .expect("dummy provider should complete successfully");

        assert_eq!(response.text.as_deref(), Some("delegated"));
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.usage, None);
    }

    #[test]
    fn build_tool_instructions_text_includes_tool_details() {
        let instructions = build_tool_instructions_text(&[ToolSpec {
            name: "shell".to_string(),
            description: "Execute shell commands".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                }
            }),
            risk_level: RiskLevel::High,
        }]);

        assert!(instructions.contains("shell"));
        assert!(instructions.contains("Execute shell commands"));
        assert!(instructions.contains("Input schema"));
    }

    #[tokio::test]
    async fn model_router_returns_default_provider() {
        let router = ModelRouter::new(
            "primary".to_string(),
            Box::new(DummyProvider {
                response_text: "primary",
                native_tool_calling: false,
            }),
        );

        let response = router
            .default_provider()
            .complete(ModelRequest {
                messages: vec![ChatMessage::user("Hello")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: None,
            })
            .await
            .expect("default provider should respond");

        assert_eq!(response.content, "primary");
    }

    #[tokio::test]
    async fn model_router_can_lookup_named_providers() {
        let mut router = ModelRouter::new(
            "primary".to_string(),
            Box::new(DummyProvider {
                response_text: "primary",
                native_tool_calling: false,
            }),
        );
        router.add_provider(
            "backup".to_string(),
            Box::new(DummyProvider {
                response_text: "backup",
                native_tool_calling: false,
            }),
        );

        let response = router
            .provider("backup")
            .expect("named provider should exist")
            .complete(ModelRequest {
                messages: vec![ChatMessage::user("Hello")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: None,
            })
            .await
            .expect("backup provider should respond");

        assert_eq!(response.content, "backup");
        assert!(router.provider("missing").is_none());
    }

    #[test]
    fn chat_request_can_include_tools() {
        let request = ChatRequest {
            messages: vec![ChatMessage::user("Use a tool")],
            model: "claude-haiku-4-5-20251001".to_string(),
            max_tokens: Some(128),
            tools: Some(vec![ToolSpec {
                name: "shell".to_string(),
                description: "Execute shell commands".to_string(),
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
            text: Some("done".to_string()),
            tool_calls: Vec::new(),
            usage: Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
            }),
        };

        assert!(response.tool_calls.is_empty());
        assert_eq!(response.text.as_deref(), Some("done"));
    }

    #[test]
    fn tool_call_and_usage_round_trip_through_json() {
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "shell".to_string(),
            arguments: r#"{"command":"pwd"}"#.to_string(),
        };
        let usage = TokenUsage {
            input_tokens: 12,
            output_tokens: 34,
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
    async fn model_provider_trait_remains_object_safe() {
        let provider: Arc<dyn ModelProvider> = Arc::new(DummyProvider {
            response_text: "",
            native_tool_calling: false,
        });
        let response = provider
            .complete(ModelRequest {
                messages: vec![ChatMessage::user("Ping")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: None,
            })
            .await
            .expect("dummy provider should complete successfully");

        assert_eq!(response.content, "received 1 messages");
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }
}
