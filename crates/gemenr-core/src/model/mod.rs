mod anthropic;

use async_trait::async_trait;

use crate::error::ModelError;
use crate::message::ChatMessage;

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
    /// Model identifier (e.g., "claude-sonnet-4-20250514").
    pub model: String,
    /// Sampling temperature (0.0 - 1.0).
    pub temperature: f64,
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

/// Unified interface for LLM providers.
///
/// Implementations must return a complete (non-streaming) response.
/// Internally, a provider may use streaming APIs but must accumulate
/// the full response before returning from `complete()`.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Send a request to the model and return the complete response.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{FinishReason, ModelProvider, ModelRequest, ModelResponse};
    use crate::message::ChatMessage;

    struct DummyProvider;

    #[async_trait::async_trait]
    impl ModelProvider for DummyProvider {
        async fn complete(
            &self,
            request: ModelRequest,
        ) -> Result<ModelResponse, crate::ModelError> {
            Ok(ModelResponse {
                content: format!("received {} messages", request.messages.len()),
                finish_reason: FinishReason::Stop,
            })
        }
    }

    #[test]
    fn model_request_construction_preserves_messages_and_parameters() {
        let request = ModelRequest {
            messages: vec![
                ChatMessage::system("Be concise."),
                ChatMessage::user("Hello!"),
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            temperature: 0.3,
            max_tokens: Some(256),
        };

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0], ChatMessage::system("Be concise."));
        assert_eq!(request.messages[1], ChatMessage::user("Hello!"));
        assert_eq!(request.model, "claude-sonnet-4-20250514");
        assert_eq!(request.temperature, 0.3);
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

    #[tokio::test]
    async fn model_provider_trait_is_object_safe() {
        let provider: Arc<dyn ModelProvider> = Arc::new(DummyProvider);
        let response = provider
            .complete(ModelRequest {
                messages: vec![ChatMessage::user("Ping")],
                model: "claude-sonnet-4-20250514".to_string(),
                temperature: 0.7,
                max_tokens: None,
            })
            .await
            .expect("dummy provider should complete successfully");

        assert_eq!(response.content, "received 1 messages");
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }
}
