use std::time::Duration;

use async_trait::async_trait;
use reqwest::{
    Client, StatusCode,
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, RETRY_AFTER, USER_AGENT},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::sleep;
use tracing::{debug, error, warn};

use crate::config::{Config, ConfigError, ModelConfig, ProviderConfig, ProviderType};
use crate::error::ModelError;
use crate::message::{ChatMessage, ChatRole};
use crate::model::{
    ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
    ModelResponse, RequestContext, TokenUsage, ToolCall, ToolsPayload, convert_request_tools,
};
use crate::tool_spec::ToolSpec;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const MAX_RETRIES: u32 = 3;
const BASE_RETRY_DELAY_SECS: u64 = 1;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;
const MAX_ERROR_MESSAGE_CHARS: usize = 200;
const CLAUDE_CLI_USER_AGENT: &str = "claude-cli/2.1.71 (external, cli)";
const CLAUDE_CLI_ANTHROPIC_BETA: &str = "claude-code-20250219,adaptive-thinking-2026-01-28,prompt-caching-scope-2026-01-05,effort-2025-11-24";

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

        Self::from_parts(selected_model, selected_provider)
    }

    /// Creates a new Anthropic provider from explicit model and provider definitions.
    pub fn from_parts(model: &ModelConfig, provider: &ProviderConfig) -> Result<Self, ConfigError> {
        if provider.provider_type != ProviderType::Anthropic {
            return Err(ConfigError::Invalid(
                "provider type is unsupported for AnthropicProvider".to_string(),
            ));
        }

        Ok(Self {
            client: Client::new(),
            api_key: provider.api_key.clone(),
            api_endpoint: provider
                .api_endpoint
                .clone()
                .unwrap_or_else(|| ANTHROPIC_API_URL.to_string()),
            default_model: model.model.clone(),
        })
    }

    async fn send_request_with_retry<T>(
        &self,
        request_body: &T,
        context: &RequestContext,
    ) -> Result<String, ModelError>
    where
        T: Serialize + ?Sized,
    {
        for attempt in 0..=MAX_RETRIES {
            if context.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(ModelError::Cancelled);
            }

            match self.do_send_request(request_body, context).await {
                Ok(response_body) => return Ok(response_body),
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
                    sleep_or_cancel(delay, context).await?;
                }
            }
        }

        unreachable!("retry loop should always return a result");
    }

    async fn do_send_request<T>(
        &self,
        request_body: &T,
        context: &RequestContext,
    ) -> Result<String, ModelError>
    where
        T: Serialize + ?Sized,
    {
        let request_timeout = context
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS));
        let response = await_or_cancel(
            self.client
                .post(&self.api_endpoint)
                .headers(default_claude_cli_headers())
                .header("x-api-key", &self.api_key)
                .json(request_body)
                .timeout(request_timeout)
                .send(),
            context,
        )
        .await?;

        let status = response.status();
        let retry_after = parse_retry_after(response.headers());
        let body = await_or_cancel(response.text(), context).await?;

        if status.is_success() {
            return Ok(body);
        }

        Err(map_error_response(status, retry_after, &body))
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            native_tool_calling: true,
            vision: false,
        }
    }

    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        ToolsPayload::Anthropic {
            tools: convert_anthropic_tools(tools),
        }
    }

    async fn complete(
        &self,
        request: ModelRequest,
        context: RequestContext,
    ) -> Result<ModelResponse, ModelError> {
        let request_body = build_request(&request, &self.default_model);
        debug!(
            model = %request_body.model,
            message_count = request_body.messages.len(),
            has_system = request_body.system.is_some(),
            "sending anthropic completion request"
        );

        let response_body = self
            .send_request_with_retry(&request_body, &context)
            .await?;
        parse_response_body(&response_body).map_err(|err| ModelError::Api {
            status: StatusCode::OK.as_u16(),
            message: truncate_error_message(&format!("failed to parse Anthropic response: {err}")),
        })
    }

    async fn chat(
        &self,
        request: ChatRequest,
        context: RequestContext,
    ) -> Result<ChatResponse, ModelError> {
        let request_body = build_chat_request(self, &request, &self.default_model)?;
        debug!(
            model = %request_body.model,
            message_count = request_body.messages.len(),
            has_system = request_body.system.is_some(),
            tool_count = request_body.tools.as_ref().map_or(0, Vec::len),
            "sending anthropic chat request"
        );

        let response_body = self
            .send_request_with_retry(&request_body, &context)
            .await?;
        parse_chat_response_body(&response_body).map_err(|err| ModelError::Api {
            status: StatusCode::OK.as_u16(),
            message: truncate_error_message(&format!("failed to parse Anthropic response: {err}")),
        })
    }
}

async fn await_or_cancel<T, Fut>(future: Fut, context: &RequestContext) -> Result<T, ModelError>
where
    Fut: std::future::Future<Output = Result<T, reqwest::Error>>,
{
    tokio::pin!(future);

    let cancellation_future = wait_for_cancellation(context.cancellation_handle());
    tokio::pin!(cancellation_future);

    tokio::select! {
        result = &mut future => result.map_err(map_request_error),
        _ = &mut cancellation_future => Err(ModelError::Cancelled),
    }
}

async fn sleep_or_cancel(duration: Duration, context: &RequestContext) -> Result<(), ModelError> {
    let sleep_future = sleep(duration);
    tokio::pin!(sleep_future);

    let cancellation_future = wait_for_cancellation(context.cancellation_handle());
    tokio::pin!(cancellation_future);

    tokio::select! {
        _ = &mut sleep_future => Ok(()),
        _ = &mut cancellation_future => Err(ModelError::Cancelled),
    }
}

async fn wait_for_cancellation(cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    while !cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        sleep(Duration::from_millis(10)).await;
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
    tools: Option<Vec<Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<RequestContentBlock>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RequestContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
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

    AnthropicRequest {
        model: resolve_model(&request.model, default_model),
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages,
        system,
        tools: None,
    }
}

fn build_chat_request(
    provider: &dyn ModelProvider,
    request: &ChatRequest,
    default_model: &str,
) -> Result<AnthropicRequest, ModelError> {
    let (system, messages) = split_system_messages(&request.messages);

    Ok(AnthropicRequest {
        model: resolve_model(&request.model, default_model),
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages,
        system,
        tools: anthropic_tools_payload(convert_request_tools(provider, request.tools.as_deref()))?,
    })
}

fn resolve_model(model: &str, default_model: &str) -> String {
    if model.trim().is_empty() {
        default_model.to_string()
    } else {
        model.to_string()
    }
}

fn convert_anthropic_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            })
        })
        .collect()
}

fn anthropic_tools_payload(
    payload: Option<ToolsPayload>,
) -> Result<Option<Vec<Value>>, ModelError> {
    match payload {
        None => Ok(None),
        Some(ToolsPayload::Anthropic { tools }) if tools.is_empty() => Ok(None),
        Some(ToolsPayload::Anthropic { tools }) => Ok(Some(tools)),
        Some(other) => Err(ModelError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
            message: format!(
                "anthropic provider expected Anthropic tool payload, got {}",
                tools_payload_kind(&other)
            ),
        }),
    }
}

fn tools_payload_kind(payload: &ToolsPayload) -> &'static str {
    match payload {
        ToolsPayload::OpenAI { .. } => "OpenAI",
        ToolsPayload::Anthropic { .. } => "Anthropic",
        ToolsPayload::PromptGuided { .. } => "PromptGuided",
    }
}

fn default_claude_cli_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static(CLAUDE_CLI_USER_AGENT));
    headers.insert(
        HeaderName::from_static("x-stainless-arch"),
        HeaderValue::from_static("arm64"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-lang"),
        HeaderValue::from_static("js"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-os"),
        HeaderValue::from_static("MacOS"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-package-version"),
        HeaderValue::from_static("0.74.0"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-retry-count"),
        HeaderValue::from_static("0"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-runtime"),
        HeaderValue::from_static("node"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-runtime-version"),
        HeaderValue::from_static("v24.3.0"),
    );
    headers.insert(
        HeaderName::from_static("x-stainless-timeout"),
        HeaderValue::from_static("600"),
    );
    headers.insert(
        HeaderName::from_static("anthropic-beta"),
        HeaderValue::from_static(CLAUDE_CLI_ANTHROPIC_BETA),
    );
    headers.insert(
        HeaderName::from_static("anthropic-dangerous-direct-browser-access"),
        HeaderValue::from_static("true"),
    );
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static(ANTHROPIC_VERSION),
    );
    headers.insert(
        HeaderName::from_static("x-app"),
        HeaderValue::from_static("cli"),
    );
    headers
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
                    content: anthropic_content(message),
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

fn anthropic_content(message: &ChatMessage) -> AnthropicMessageContent {
    if let Some(blocks) = anthropic_blocks_from_metadata(message) {
        AnthropicMessageContent::Blocks(blocks)
    } else {
        AnthropicMessageContent::Text(message.content.clone())
    }
}

fn anthropic_blocks_from_metadata(message: &ChatMessage) -> Option<Vec<RequestContentBlock>> {
    if let Some(tool_calls) = message.metadata.get("tool_calls") {
        let tool_calls = serde_json::from_str::<Vec<ToolCall>>(tool_calls).ok()?;
        let mut blocks =
            Vec::with_capacity(tool_calls.len() + usize::from(!message.content.is_empty()));
        if !message.content.is_empty() {
            blocks.push(RequestContentBlock::Text {
                text: message.content.clone(),
            });
        }
        blocks.extend(
            tool_calls
                .into_iter()
                .map(|tool_call| RequestContentBlock::ToolUse {
                    id: tool_call.id,
                    name: tool_call.name,
                    input: serde_json::from_str(&tool_call.arguments).unwrap_or(Value::Null),
                }),
        );
        return Some(blocks);
    }

    let tool_use_id = message.metadata.get("tool_result_for")?;

    Some(vec![RequestContentBlock::ToolResult {
        tool_use_id: tool_use_id.clone(),
        content: message
            .metadata
            .get("tool_result_content")
            .cloned()
            .unwrap_or_else(|| message.content.clone()),
        is_error: message
            .metadata
            .get("is_error")
            .is_some_and(|value| value.eq_ignore_ascii_case("true")),
    }])
}

fn parse_response_body(body: &str) -> Result<ModelResponse, serde_json::Error> {
    let response = parse_anthropic_response(body)?;
    let content = collect_text_content(&response.content);

    Ok(ModelResponse {
        content,
        finish_reason: parse_finish_reason(response.stop_reason.as_deref()),
    })
}

fn parse_chat_response_body(body: &str) -> Result<ChatResponse, serde_json::Error> {
    let response = parse_anthropic_response(body)?;
    let text = collect_text_content(&response.content);
    let tool_calls = response
        .content
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
            ContentBlock::Text { .. } | ContentBlock::Unknown => None,
        })
        .map(|(id, name, input)| {
            Ok(ToolCall {
                id,
                name,
                arguments: serde_json::to_string(&input)?,
            })
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;

    Ok(ChatResponse {
        text: (!text.is_empty()).then_some(text),
        tool_calls,
        usage: response.usage.map(|usage| TokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        }),
    })
}

fn parse_anthropic_response(body: &str) -> Result<AnthropicResponse, serde_json::Error> {
    serde_json::from_str(body)
}

fn collect_text_content(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::ToolUse { .. } | ContentBlock::Unknown => None,
        })
        .collect()
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
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Duration;

    use reqwest::{
        Client, StatusCode,
        header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue, RETRY_AFTER, USER_AGENT},
    };
    use serde_json::json;
    use tokio::time::sleep;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    use super::{
        ANTHROPIC_API_URL, ANTHROPIC_VERSION, AnthropicProvider, AnthropicRequest,
        BASE_RETRY_DELAY_SECS, CLAUDE_CLI_ANTHROPIC_BETA, CLAUDE_CLI_USER_AGENT,
        DEFAULT_MAX_TOKENS, anthropic_tools_payload, build_chat_request, build_request,
        default_claude_cli_headers, map_error_response, parse_chat_response_body,
        parse_response_body, parse_retry_after, retry_delay, should_retry, split_system_messages,
    };
    use crate::config::{Config, ModelConfig, ProviderConfig, ProviderType};
    use crate::error::ModelError;
    use crate::message::ChatMessage;
    use crate::model::{
        ChatRequest, FinishReason, ModelCapabilities, ModelProvider, ModelRequest, RequestContext,
        TokenUsage, ToolsPayload,
    };
    use crate::tool_spec::{RiskLevel, ToolSpec};

    fn provider_for_mock(server: &MockServer) -> AnthropicProvider {
        AnthropicProvider {
            client: Client::new(),
            api_key: "test-api-key".to_string(),
            api_endpoint: format!("{}/v1/messages", server.uri()),
            default_model: "claude-haiku-4-5-20251001".to_string(),
        }
    }

    fn success_response_body(text: &str) -> serde_json::Value {
        json!({
            "id": "msg_test_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        })
    }

    fn tool_use_response_body(
        tool_id: &str,
        tool_name: &str,
        input: serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "id": "msg_test_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll use a tool."},
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": input
                }
            ],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 15, "output_tokens": 30}
        })
    }

    fn error_response_body(error_type: &str, message: &str) -> serde_json::Value {
        json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })
    }

    fn default_request_context() -> RequestContext {
        RequestContext::default()
    }

    fn test_chat_request() -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage::user("Hello from the mock test")],
            model: String::new(),
            max_tokens: Some(128),
            tools: None,
        }
    }

    #[test]
    fn capabilities_enable_native_tool_calling_without_vision() {
        let provider = AnthropicProvider::new(&test_config(None)).expect("provider should build");

        assert_eq!(
            provider.capabilities(),
            ModelCapabilities {
                native_tool_calling: true,
                vision: false,
            }
        );
    }

    #[test]
    fn supports_native_tools_returns_true() {
        let provider = AnthropicProvider::new(&test_config(None)).expect("provider should build");

        assert!(provider.supports_native_tools());
    }

    #[test]
    fn convert_tools_returns_anthropic_payload_shape() {
        let provider = AnthropicProvider::new(&test_config(None)).expect("provider should build");
        let tools = vec![test_tool_spec()];

        let payload = provider.convert_tools(&tools);

        match payload {
            ToolsPayload::Anthropic { tools } => {
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0]["name"], json!("shell"));
                assert_eq!(tools[0]["description"], json!("Execute a shell command"));
                assert_eq!(
                    tools[0]["input_schema"],
                    json!({
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    })
                );
            }
            other => panic!("expected Anthropic tools payload, got {other:?}"),
        }
    }

    struct StubAnthropicPayloadProvider;

    #[async_trait::async_trait]
    impl ModelProvider for StubAnthropicPayloadProvider {
        async fn complete(
            &self,
            _request: ModelRequest,
            _context: RequestContext,
        ) -> Result<crate::model::ModelResponse, ModelError> {
            unreachable!("stub provider should not be used for completion")
        }

        fn convert_tools(&self, _tools: &[ToolSpec]) -> ToolsPayload {
            ToolsPayload::Anthropic {
                tools: vec![json!({
                    "name": "shell_from_provider",
                    "description": "Converted by provider",
                    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}
                })],
            }
        }
    }

    struct StubPromptGuidedProvider;

    #[async_trait::async_trait]
    impl ModelProvider for StubPromptGuidedProvider {
        async fn complete(
            &self,
            _request: ModelRequest,
            _context: RequestContext,
        ) -> Result<crate::model::ModelResponse, ModelError> {
            unreachable!("stub provider should not be used for completion")
        }
    }

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
        assert_eq!(
            anthropic_messages[0].content,
            super::AnthropicMessageContent::Text("Hello".to_string())
        );
        assert_eq!(anthropic_messages[1].role, "assistant");
        assert_eq!(
            anthropic_messages[1].content,
            super::AnthropicMessageContent::Text("Hi there".to_string())
        );
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
    fn parse_chat_response_body_extracts_tool_calls() {
        let response = r#"{
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "shell",
                    "input": {"command": "ls -la"}
                }
            ],
            "stop_reason": "tool_use"
        }"#;

        let parsed = parse_chat_response_body(response).expect("response should parse");

        assert_eq!(parsed.text, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "toolu_01A09q90qw90lq917835lq9");
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"command":"ls -la"}"#);
    }

    #[test]
    fn parse_chat_response_body_handles_mixed_text_and_tool_calls() {
        let response = r#"{
            "content": [
                {"type": "text", "text": "Let me check that..."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "shell",
                    "input": {"command": "pwd"}
                }
            ],
            "stop_reason": "tool_use"
        }"#;

        let parsed = parse_chat_response_body(response).expect("response should parse");

        assert_eq!(parsed.text.as_deref(), Some("Let me check that..."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"command":"pwd"}"#);
    }

    #[test]
    fn parse_chat_response_body_keeps_tool_calls_empty_for_plain_text() {
        let response = r#"{
            "content": [
                {"type": "text", "text": "Hello there"}
            ],
            "stop_reason": "end_turn"
        }"#;

        let parsed = parse_chat_response_body(response).expect("response should parse");

        assert_eq!(parsed.text.as_deref(), Some("Hello there"));
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.usage, None);
    }

    #[test]
    fn parse_chat_response_body_maps_usage() {
        let response = r#"{
            "content": [
                {"type": "text", "text": "Done"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }"#;

        let parsed = parse_chat_response_body(response).expect("response should parse");

        assert_eq!(
            parsed.usage,
            Some(crate::model::TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
            })
        );
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
    fn request_serialization_omits_missing_system_and_temperature() {
        let request = build_request(
            &ModelRequest {
                messages: vec![ChatMessage::user("Hello")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: None,
            },
            "fallback-model",
        );

        let value = serde_json::to_value(&request).expect("request should serialize");

        assert_eq!(value["model"], json!("claude-haiku-4-5-20251001"));
        assert_eq!(value["max_tokens"], json!(DEFAULT_MAX_TOKENS));
        assert!(value.get("system").is_none());
        assert!(value.get("temperature").is_none());
        assert_eq!(value["messages"][0]["role"], json!("user"));
        assert_eq!(value["messages"][0]["content"], json!("Hello"));
    }

    #[test]
    fn split_system_messages_preserves_native_tool_call_blocks() {
        let messages = vec![
            ChatMessage::assistant("Thinking...").with_metadata(
                "tool_calls",
                r#"[{"id":"call-1","name":"shell","arguments":"{\"command\":\"pwd\"}"}]"#,
            ),
            ChatMessage::user("/tmp/project")
                .with_metadata("tool_result_for", "call-1")
                .with_metadata("is_error", "false"),
        ];

        let (_, anthropic_messages) = split_system_messages(&messages);
        let value = serde_json::to_value(&anthropic_messages).expect("messages should serialize");

        assert_eq!(
            value[0]["content"][0],
            json!({"type": "text", "text": "Thinking..."})
        );
        assert_eq!(
            value[0]["content"][1],
            json!({
                "type": "tool_use",
                "id": "call-1",
                "name": "shell",
                "input": {"command": "pwd"}
            })
        );
        assert_eq!(
            value[1]["content"][0],
            json!({
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": "/tmp/project"
            })
        );
    }

    #[test]
    fn chat_request_serialization_includes_tools_when_present() {
        let provider = AnthropicProvider::new(&test_config(None)).expect("provider should build");
        let request = build_chat_request(
            &provider,
            &ChatRequest {
                messages: vec![ChatMessage::user("Use a tool")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: Some(256),
                tools: Some(vec![test_tool_spec()]),
            },
            "fallback-model",
        )
        .expect("request should build");

        let value = serde_json::to_value(&request).expect("request should serialize");

        assert_eq!(value["tools"][0]["name"], json!("shell"));
        assert_eq!(
            value["tools"][0]["description"],
            json!("Execute a shell command")
        );
        assert_eq!(
            value["tools"][0]["input_schema"],
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            })
        );
    }

    #[test]
    fn chat_request_uses_provider_tool_conversion_payload() {
        let request = build_chat_request(
            &StubAnthropicPayloadProvider,
            &ChatRequest {
                messages: vec![ChatMessage::user("Use a tool")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: Some(256),
                tools: Some(vec![test_tool_spec()]),
            },
            "fallback-model",
        )
        .expect("request should build");

        let value = serde_json::to_value(&request).expect("request should serialize");

        assert_eq!(value["tools"][0]["name"], json!("shell_from_provider"));
        assert_eq!(
            value["tools"][0]["description"],
            json!("Converted by provider")
        );
        assert_eq!(
            value["tools"][0]["input_schema"]["properties"]["cmd"]["type"],
            json!("string")
        );
    }

    #[test]
    fn chat_request_serialization_omits_tools_when_absent() {
        let provider = AnthropicProvider::new(&test_config(None)).expect("provider should build");
        let request = build_chat_request(
            &provider,
            &ChatRequest {
                messages: vec![ChatMessage::user("Hello")],
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: None,
                tools: None,
            },
            "fallback-model",
        )
        .expect("request should build");

        let value = serde_json::to_value(&request).expect("request should serialize");

        assert!(value.get("tools").is_none());
    }

    #[test]
    fn anthropic_tools_payload_rejects_non_anthropic_payloads() {
        let error = anthropic_tools_payload(Some(
            StubPromptGuidedProvider.convert_tools(&[test_tool_spec()]),
        ))
        .expect_err("prompt-guided payload should be rejected");

        match error {
            ModelError::Api { status, message } => {
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR.as_u16());
                assert!(message.contains("PromptGuided"));
            }
            other => panic!("expected API error, got {other:?}"),
        }
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
            tools: None,
        };

        let serialized = serde_json::to_string(&request).expect("request should serialize");

        assert!(!serialized.contains("\"system\""));
        assert!(!serialized.contains("\"temperature\""));
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
    fn default_headers_match_claude_cli_shape() {
        let headers = default_claude_cli_headers();

        assert_eq!(headers[ACCEPT], "application/json");
        assert_eq!(headers[CONTENT_TYPE], "application/json");
        assert_eq!(headers[USER_AGENT], CLAUDE_CLI_USER_AGENT);
        assert_eq!(headers["x-stainless-arch"], "arm64");
        assert_eq!(headers["x-stainless-lang"], "js");
        assert_eq!(headers["x-stainless-os"], "MacOS");
        assert_eq!(headers["x-stainless-package-version"], "0.74.0");
        assert_eq!(headers["x-stainless-retry-count"], "0");
        assert_eq!(headers["x-stainless-runtime"], "node");
        assert_eq!(headers["x-stainless-runtime-version"], "v24.3.0");
        assert_eq!(headers["x-stainless-timeout"], "600");
        assert_eq!(headers["anthropic-beta"], CLAUDE_CLI_ANTHROPIC_BETA);
        assert_eq!(headers["anthropic-dangerous-direct-browser-access"], "true");
        assert_eq!(headers["anthropic-version"], ANTHROPIC_VERSION);
        assert_eq!(headers["x-app"], "cli");
    }

    #[test]
    fn default_headers_do_not_set_connection_header() {
        let headers = default_claude_cli_headers();

        assert!(headers.get("connection").is_none());
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

    #[tokio::test]
    async fn test_chat_success_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-api-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(success_response_body("Hello, world!")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let response = provider_for_mock(&server)
            .chat(test_chat_request(), default_request_context())
            .await
            .expect("mocked request should succeed");

        assert_eq!(response.text.as_deref(), Some("Hello, world!"));
        assert!(response.tool_calls.is_empty());
        assert_eq!(
            response.usage,
            Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 20,
            })
        );
    }

    #[tokio::test]
    async fn test_rate_limit_returns_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .append_header("retry-after", "0")
                    .set_body_json(error_response_body("rate_limit_error", "Rate limited")),
            )
            .expect(4)
            .mount(&server)
            .await;

        let error = provider_for_mock(&server)
            .chat(test_chat_request(), default_request_context())
            .await
            .expect_err("rate limit response should fail");

        assert!(matches!(
            error,
            ModelError::RateLimit {
                retry_after: Some(delay),
            } if delay.is_zero()
        ));
    }

    #[tokio::test]
    async fn test_auth_failure_returns_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(401).set_body_json(error_response_body(
                    "authentication_error",
                    "Invalid API key",
                )),
            )
            .expect(1)
            .mount(&server)
            .await;

        let error = provider_for_mock(&server)
            .chat(test_chat_request(), default_request_context())
            .await
            .expect_err("auth response should fail");

        assert!(matches!(error, ModelError::Auth(message) if message == "Invalid API key"));
    }

    #[tokio::test]
    async fn test_request_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(250))
                    .set_body_json(success_response_body("delayed")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider_for_mock(&server);
        let request_body =
            build_chat_request(&provider, &test_chat_request(), &provider.default_model)
                .expect("request should build");
        let context = default_request_context().with_timeout(Duration::from_millis(100));
        let error = provider
            .do_send_request(&request_body, &context)
            .await
            .expect_err("delayed response should time out");

        assert!(matches!(error, ModelError::Timeout));
    }

    #[tokio::test]
    async fn test_request_cancellation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(2))
                    .set_body_json(success_response_body("should not complete")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_flag = Arc::clone(&cancelled);
        tokio::spawn(async move {
            sleep(Duration::from_millis(100)).await;
            cancel_flag.store(true, Ordering::Release);
        });

        let error = provider_for_mock(&server)
            .chat(test_chat_request(), RequestContext::new(cancelled))
            .await
            .expect_err("cancelled request should fail");

        assert!(matches!(error, ModelError::Cancelled));
    }

    #[tokio::test]
    async fn test_tool_use_response_parsing() {
        let server = MockServer::start().await;
        let tool_input = json!({ "path": "/tmp/test.txt" });
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(tool_use_response_body(
                    "toolu_01",
                    "fs.read",
                    tool_input.clone(),
                )),
            )
            .expect(1)
            .mount(&server)
            .await;

        let mut request = test_chat_request();
        request.tools = Some(vec![test_tool_spec()]);

        let response = provider_for_mock(&server)
            .chat(request, default_request_context())
            .await
            .expect("tool use response should parse");

        assert_eq!(response.text.as_deref(), Some("I'll use a tool."));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "toolu_01");
        assert_eq!(response.tool_calls[0].name, "fs.read");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&response.tool_calls[0].arguments)
                .expect("tool arguments should be valid json"),
            tool_input
        );
    }

    #[tokio::test]
    async fn test_api_error_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(500)
                    .set_body_json(error_response_body("api_error", "Service unavailable")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let error = provider_for_mock(&server)
            .chat(test_chat_request(), default_request_context())
            .await
            .expect_err("server error should fail");

        assert!(matches!(
            error,
            ModelError::Api { status: 500, ref message } if message == "Service unavailable"
        ));
    }

    #[tokio::test]
    async fn test_request_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-api-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .and(header("content-type", "application/json"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(success_response_body("headers ok")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let response = provider_for_mock(&server)
            .chat(test_chat_request(), default_request_context())
            .await
            .expect("header-validated request should succeed");

        assert_eq!(response.text.as_deref(), Some("headers ok"));
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
                max_tokens: None,
            },
        );

        Config {
            model: "default".to_string(),
            providers,
            models,
            tool_dispatcher: "auto".to_string(),
            access: Default::default(),
            cron: Vec::new(),
            policy: Default::default(),
            fallback: None,
            mcp: Default::default(),
        }
    }

    fn test_tool_spec() -> ToolSpec {
        ToolSpec {
            name: "shell".to_string(),
            description: "Execute a shell command".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
            risk_level: RiskLevel::High,
        }
    }
}
