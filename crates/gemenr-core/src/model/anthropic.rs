use std::time::Duration;

use async_trait::async_trait;
use reqwest::{
    Client, StatusCode,
    header::{HeaderMap, RETRY_AFTER},
};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tracing::{debug, error, warn};

use crate::config::{Config, ConfigError, ProviderType};
use crate::error::ModelError;
use crate::message::{ChatMessage, ChatRole};
use crate::model::{FinishReason, ModelProvider, ModelRequest, ModelResponse};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const MAX_RETRIES: u32 = 3;
const BASE_RETRY_DELAY_SECS: u64 = 1;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const MAX_ERROR_MESSAGE_CHARS: usize = 200;

/// Anthropic Claude API provider.
///
/// Sends requests to the Anthropic Messages API and returns complete responses.
/// Implements exponential backoff retry for transient errors.
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    api_endpoint: String,
    default_model: String,
}

impl AnthropicProvider {
    /// Creates a new Anthropic provider from the selected configuration entry.
    pub fn new(config: &Config) -> Result<Self, ConfigError> {
        let selected_model = config.selected_model()?;
        let selected_provider = config.selected_provider()?;

        if selected_provider.provider_type != ProviderType::Anthropic {
            return Err(ConfigError::Invalid(format!(
                "selected model `{}` uses unsupported provider type for AnthropicProvider",
                config.model
            )));
        }

        Ok(Self {
            client: Client::new(),
            api_key: selected_provider.api_key.clone(),
            api_endpoint: selected_provider
                .api_endpoint
                .clone()
                .unwrap_or_else(|| ANTHROPIC_API_URL.to_string()),
            default_model: selected_model.model.clone(),
        })
    }

    async fn complete_with_retry(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        for attempt in 0..=MAX_RETRIES {
            match self.do_complete(&request).await {
                Ok(response) => return Ok(response),
                Err(err) => {
                    let is_retryable = should_retry(&err);
                    let exhausted = attempt == MAX_RETRIES;

                    if !is_retryable || exhausted {
                        error!(
                            attempt = attempt + 1,
                            max_retries = MAX_RETRIES,
                            error = %err,
                            "anthropic request failed"
                        );
                        return Err(err);
                    }

                    let delay = retry_delay(attempt, &err);
                    warn!(
                        attempt = attempt + 1,
                        max_retries = MAX_RETRIES,
                        delay_ms = delay.as_millis() as u64,
                        error = %err,
                        "retrying anthropic request"
                    );
                    sleep(delay).await;
                }
            }
        }

        unreachable!("retry loop should always return a result");
    }

    async fn do_complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let request_body = build_request(request, &self.default_model);
        debug!(
            model = %request_body.model,
            message_count = request_body.messages.len(),
            has_system = request_body.system.is_some(),
            "sending anthropic request"
        );

        let response = self
            .client
            .post(&self.api_endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request_body)
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .send()
            .await
            .map_err(map_request_error)?;

        let status = response.status();
        let retry_after = parse_retry_after(response.headers());
        let body = response.text().await.map_err(map_request_error)?;

        if status.is_success() {
            return parse_response_body(&body).map_err(|err| ModelError::Api {
                status: status.as_u16(),
                message: truncate_error_message(&format!(
                    "failed to parse Anthropic response: {err}"
                )),
            });
        }

        Err(map_error_response(status, retry_after, &body))
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
        self.complete_with_retry(request).await
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    _error_type: String,
    message: String,
}

fn build_request(request: &ModelRequest, default_model: &str) -> AnthropicRequest {
    let (system, messages) = split_system_messages(&request.messages);
    let model = if request.model.trim().is_empty() {
        default_model.to_string()
    } else {
        request.model.clone()
    };

    AnthropicRequest {
        model,
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages,
        system,
        temperature: Some(request.temperature),
    }
}

fn split_system_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system_parts = Vec::new();
    let mut anthropic_messages = Vec::new();

    for message in messages {
        match message.role {
            ChatRole::System => system_parts.push(message.content.clone()),
            ChatRole::User | ChatRole::Assistant => {
                anthropic_messages.push(AnthropicMessage {
                    role: anthropic_role(message.role).to_string(),
                    content: message.content.clone(),
                });
            }
        }
    }

    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n"));
    (system, anthropic_messages)
}

fn anthropic_role(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => unreachable!("system messages are handled separately"),
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    }
}

fn parse_response_body(body: &str) -> Result<ModelResponse, serde_json::Error> {
    let response: AnthropicResponse = serde_json::from_str(body)?;
    let content = response
        .content
        .into_iter()
        .filter(|block| block.content_type == "text")
        .filter_map(|block| block.text)
        .collect::<Vec<_>>()
        .join("");

    Ok(ModelResponse {
        content,
        finish_reason: parse_finish_reason(response.stop_reason.as_deref()),
    })
}

fn parse_finish_reason(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("max_tokens") => FinishReason::MaxTokens,
        _ => FinishReason::Stop,
    }
}

fn map_error_response(status: StatusCode, retry_after: Option<Duration>, body: &str) -> ModelError {
    let message = parse_error_message(body);

    match status {
        StatusCode::UNAUTHORIZED => ModelError::Auth(message),
        StatusCode::TOO_MANY_REQUESTS => ModelError::RateLimit { retry_after },
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ModelError::Timeout,
        _ => ModelError::Api {
            status: status.as_u16(),
            message,
        },
    }
}

fn parse_error_message(body: &str) -> String {
    serde_json::from_str::<AnthropicErrorResponse>(body)
        .ok()
        .map(|response| truncate_error_message(&response.error.message))
        .unwrap_or_else(|| truncate_error_message(body.trim()))
}

fn truncate_error_message(message: &str) -> String {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return "Anthropic API request failed".to_string();
    }

    if trimmed.chars().count() <= MAX_ERROR_MESSAGE_CHARS {
        return trimmed.to_string();
    }

    let mut end = MAX_ERROR_MESSAGE_CHARS;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }

    format!("{}...", &trimmed[..end])
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
        .map(Duration::from_secs_f64)
}

fn map_request_error(error: reqwest::Error) -> ModelError {
    if error.is_timeout() {
        return ModelError::Timeout;
    }

    ModelError::Network(truncate_error_message(&error.to_string()))
}

fn should_retry(error: &ModelError) -> bool {
    matches!(error, ModelError::Timeout | ModelError::RateLimit { .. })
}

fn retry_delay(attempt: u32, error: &ModelError) -> Duration {
    match error {
        ModelError::RateLimit {
            retry_after: Some(delay),
        } => *delay,
        _ => {
            Duration::from_secs(BASE_RETRY_DELAY_SECS.saturating_mul(2u64.saturating_pow(attempt)))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use reqwest::{
        StatusCode,
        header::{HeaderMap, HeaderValue, RETRY_AFTER},
    };
    use serde_json::json;

    use super::{
        ANTHROPIC_API_URL, AnthropicProvider, AnthropicRequest, BASE_RETRY_DELAY_SECS,
        DEFAULT_MAX_TOKENS, build_request, map_error_response, parse_response_body,
        parse_retry_after, retry_delay, should_retry, split_system_messages,
    };
    use crate::config::{Config, ModelConfig, ProviderConfig, ProviderType};
    use crate::error::ModelError;
    use crate::message::ChatMessage;
    use crate::model::{FinishReason, ModelRequest};

    #[test]
    fn split_system_messages_joins_system_prompts_and_preserves_conversation_order() {
        let messages = vec![
            ChatMessage::system("Be helpful."),
            ChatMessage::user("Hello"),
            ChatMessage::system("Be concise."),
            ChatMessage::assistant("Hi there"),
        ];

        let (system, anthropic_messages) = split_system_messages(&messages);

        assert_eq!(system, Some("Be helpful.\nBe concise.".to_string()));
        assert_eq!(anthropic_messages.len(), 2);
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[0].content, "Hello");
        assert_eq!(anthropic_messages[1].role, "assistant");
        assert_eq!(anthropic_messages[1].content, "Hi there");
    }

    #[test]
    fn parse_response_body_maps_stop_reasons_and_collects_text_blocks() {
        let stop_response = r#"{
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " there"}
            ],
            "stop_reason": "end_turn"
        }"#;
        let max_tokens_response = r#"{
            "content": [{"type": "text", "text": "Truncated"}],
            "stop_reason": "max_tokens"
        }"#;

        let stop = parse_response_body(stop_response).expect("response should parse");
        let max_tokens = parse_response_body(max_tokens_response).expect("response should parse");

        assert_eq!(stop.content, "Hello there");
        assert_eq!(stop.finish_reason, FinishReason::Stop);
        assert_eq!(max_tokens.content, "Truncated");
        assert_eq!(max_tokens.finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn map_error_response_covers_auth_rate_limit_and_api_variants() {
        let auth = map_error_response(
            StatusCode::UNAUTHORIZED,
            None,
            r#"{"error":{"type":"authentication_error","message":"bad key"}}"#,
        );
        let rate_limit = map_error_response(
            StatusCode::TOO_MANY_REQUESTS,
            Some(Duration::from_secs(5)),
            r#"{"error":{"type":"rate_limit_error","message":"slow down"}}"#,
        );
        let api = map_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            None,
            r#"{"error":{"type":"api_error","message":"upstream failure"}}"#,
        );

        match auth {
            ModelError::Auth(message) => assert_eq!(message, "bad key"),
            other => panic!("expected auth error, got {other:?}"),
        }

        match rate_limit {
            ModelError::RateLimit { retry_after } => {
                assert_eq!(retry_after, Some(Duration::from_secs(5)));
            }
            other => panic!("expected rate limit error, got {other:?}"),
        }

        match api {
            ModelError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "upstream failure");
            }
            other => panic!("expected API error, got {other:?}"),
        }
    }

    #[test]
    fn request_serialization_omits_missing_system_and_keeps_temperature() {
        let request = build_request(
            &ModelRequest {
                messages: vec![ChatMessage::user("Hello")],
                model: "claude-haiku-4-5-20251001".to_string(),
                temperature: 0.3,
                max_tokens: None,
            },
            "fallback-model",
        );

        let value = serde_json::to_value(&request).expect("request should serialize");

        assert_eq!(value["model"], json!("claude-haiku-4-5-20251001"));
        assert_eq!(value["max_tokens"], json!(DEFAULT_MAX_TOKENS));
        assert_eq!(value["temperature"], json!(0.3));
        assert!(value.get("system").is_none());
        assert_eq!(value["messages"][0]["role"], json!("user"));
        assert_eq!(value["messages"][0]["content"], json!("Hello"));
    }

    #[test]
    fn retry_after_header_parses_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("2.5"));

        let retry_after = parse_retry_after(&headers);

        assert_eq!(retry_after, Some(Duration::from_secs_f64(2.5)));
    }

    #[test]
    fn anthropic_request_skips_system_when_none() {
        let request = AnthropicRequest {
            model: "claude-haiku-4-5-20251001".to_string(),
            max_tokens: 128,
            messages: vec![],
            system: None,
            temperature: Some(0.7),
        };

        let serialized = serde_json::to_string(&request).expect("request should serialize");

        assert!(!serialized.contains("\"system\""));
        assert!(serialized.contains("\"temperature\":0.7"));
    }

    #[test]
    fn provider_uses_default_api_endpoint_when_config_has_none() {
        let provider = AnthropicProvider::new(&test_config(None)).expect("provider should build");

        assert_eq!(provider.api_endpoint, ANTHROPIC_API_URL);
    }

    #[test]
    fn provider_uses_configured_api_endpoint_override() {
        let provider =
            AnthropicProvider::new(&test_config(Some("https://example.com/v1/messages")))
                .expect("provider should build");

        assert_eq!(provider.api_endpoint, "https://example.com/v1/messages");
    }

    #[test]
    fn retry_logic_only_retries_timeouts_and_rate_limits() {
        assert!(should_retry(&ModelError::Timeout));
        assert!(should_retry(&ModelError::RateLimit { retry_after: None }));
        assert!(!should_retry(&ModelError::Auth("bad key".to_string())));
        assert!(!should_retry(&ModelError::Network("offline".to_string())));
        assert!(!should_retry(&ModelError::Api {
            status: 500,
            message: "server error".to_string(),
        }));
    }

    #[test]
    fn retry_delay_uses_retry_after_or_exponential_backoff() {
        assert_eq!(
            retry_delay(0, &ModelError::Timeout),
            Duration::from_secs(BASE_RETRY_DELAY_SECS)
        );
        assert_eq!(
            retry_delay(2, &ModelError::Timeout),
            Duration::from_secs(BASE_RETRY_DELAY_SECS * 4)
        );
        assert_eq!(
            retry_delay(
                1,
                &ModelError::RateLimit {
                    retry_after: Some(Duration::from_secs(7)),
                }
            ),
            Duration::from_secs(7)
        );
    }

    fn test_config(api_endpoint: Option<&str>) -> Config {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_string(),
            ProviderConfig {
                provider_type: ProviderType::Anthropic,
                api_key: "test-key".to_string(),
                api_endpoint: api_endpoint.map(str::to_string),
            },
        );

        let mut models = HashMap::new();
        models.insert(
            "default".to_string(),
            ModelConfig {
                provider: "anthropic".to_string(),
                model: "claude-haiku-4-5-20251001".to_string(),
                temperature: 0.7,
                max_tokens: None,
            },
        );

        Config {
            model: "default".to_string(),
            providers,
            models,
        }
    }
}
