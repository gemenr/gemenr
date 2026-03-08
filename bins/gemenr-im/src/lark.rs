use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt;
use gemenr_core::{
    AccessAdapter, AccessError, AccessInbound, AccessOutbound, ConversationDriver, ConversationId,
    LarkConfig, ReplyRoute,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::debug;

const DEFAULT_LARK_WS_ENDPOINT: &str = "wss://open.feishu.cn/ws";
const LARK_TOKEN_URL: &str =
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal";
const LARK_SEND_URL: &str = "https://open.feishu.cn/open-apis/im/v1/messages";
const TOKEN_REFRESH_SKEW_SECS: u64 = 60;

/// Cached tenant access token with its refresh schedule.
#[derive(Debug, Clone)]
pub struct CachedTenantToken {
    /// Tenant token value.
    pub token: String,
    /// Hard expiration instant.
    pub expires_at: Instant,
    /// Refresh deadline used before hard expiration.
    pub refresh_at: Instant,
}

/// Long-connection adapter for Lark / Feishu conversations.
pub struct LarkAdapter {
    http: Client,
    config: LarkConfig,
    tenant_token: RwLock<Option<CachedTenantToken>>,
    seen_message_ids: Mutex<HashSet<String>>,
    debounce: Mutex<HashMap<String, PendingConversation>>,
}

/// Errors produced by the Lark transport.
#[derive(Debug, Error)]
pub enum LarkError {
    /// Token or message HTTP request failed.
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// WebSocket connection failed.
    #[error(transparent)]
    WebSocket(#[from] Box<tokio_tungstenite::tungstenite::Error>),
    /// Lark returned a malformed payload.
    #[error("invalid lark payload: {0}")]
    Protocol(String),
    /// Dispatch into the conversation driver failed.
    #[error("conversation driver failed: {0}")]
    Driver(String),
}

#[derive(Debug, Clone)]
struct PendingConversation {
    conversation_id: ConversationId,
    user_id: String,
    route: ReplyRoute,
    metadata: Value,
    text_parts: Vec<String>,
    last_update: Instant,
}

#[derive(Debug, Deserialize)]
struct TokenResponseBody {
    tenant_access_token: String,
    expire: i64,
}

#[derive(Debug, Deserialize)]
struct LarkEnvelope {
    #[serde(default)]
    event: Option<LarkMessageEvent>,
}

#[derive(Debug, Deserialize)]
struct LarkMessageEvent {
    sender: LarkSender,
    message: LarkMessage,
}

#[derive(Debug, Deserialize)]
struct LarkSender {
    sender_id: LarkSenderId,
}

#[derive(Debug, Deserialize)]
struct LarkSenderId {
    #[serde(default)]
    open_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LarkMessage {
    message_id: String,
    chat_id: String,
    #[serde(default)]
    thread_id: Option<String>,
    chat_type: String,
    content: String,
    #[serde(default)]
    mentions: Vec<LarkMention>,
}

#[derive(Debug, Deserialize)]
struct LarkMention {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    is_bot: Option<bool>,
    #[serde(default)]
    id: Option<LarkMentionId>,
}

#[derive(Debug, Deserialize)]
struct LarkMentionId {
    #[serde(default)]
    open_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LarkTextContent {
    text: String,
}

impl CachedTenantToken {
    /// Build a cached token from a token value and Lark TTL seconds.
    #[must_use]
    pub fn from_ttl(token: String, ttl_secs: i64, now: Instant) -> Self {
        let ttl = ttl_secs.max(0) as u64;
        let expires_at = now + Duration::from_secs(ttl);
        let refresh_after = ttl.saturating_sub(TOKEN_REFRESH_SKEW_SECS.min(ttl));
        Self {
            token,
            expires_at,
            refresh_at: now + Duration::from_secs(refresh_after),
        }
    }

    fn should_refresh(&self, now: Instant) -> bool {
        now >= self.refresh_at || now >= self.expires_at
    }
}

impl LarkAdapter {
    /// Create a Lark adapter from config.
    #[must_use]
    pub fn new(config: LarkConfig) -> Self {
        Self {
            http: Client::new(),
            config,
            tenant_token: RwLock::new(None),
            seen_message_ids: Mutex::new(HashSet::new()),
            debounce: Mutex::new(HashMap::new()),
        }
    }

    /// Run one long-connection session until the connection closes.
    pub async fn run(
        &self,
        driver: std::sync::Arc<dyn ConversationDriver>,
    ) -> Result<(), LarkError> {
        let mut stream = self.connect_event_stream().await?;

        while let Some(frame) = stream.next().await {
            match frame.map_err(|error| LarkError::WebSocket(Box::new(error)))? {
                Message::Text(text) => {
                    self.handle_event_text(text.as_ref(), driver.clone())
                        .await?;
                }
                Message::Binary(bytes) => {
                    let text = String::from_utf8(bytes.to_vec())
                        .map_err(|error| LarkError::Protocol(error.to_string()))?;
                    self.handle_event_text(&text, driver.clone()).await?;
                }
                Message::Close(_) => break,
                Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            }
        }

        for inbound in self.flush_all_debounced() {
            self.dispatch_inbound(driver.clone(), inbound).await?;
        }

        Ok(())
    }

    /// Refresh or reuse the cached tenant token.
    pub async fn refresh_tenant_token(&self) -> Result<String, LarkError> {
        let now = Instant::now();
        if let Some(cached) = self.tenant_token.read().await.clone()
            && !cached.should_refresh(now)
        {
            return Ok(cached.token);
        }

        let response = self
            .http
            .post(LARK_TOKEN_URL)
            .json(&json!({
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            }))
            .send()
            .await?
            .error_for_status()?;
        let body: TokenResponseBody = response.json().await?;
        let cached =
            CachedTenantToken::from_ttl(body.tenant_access_token.clone(), body.expire, now);
        *self.tenant_token.write().await = Some(cached.clone());
        Ok(cached.token)
    }

    async fn connect_event_stream(
        &self,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        LarkError,
    > {
        let endpoint = self
            .config
            .ws_endpoint
            .clone()
            .unwrap_or_else(|| DEFAULT_LARK_WS_ENDPOINT.to_string());
        let _token = self.refresh_tenant_token().await?;
        let (stream, _) = connect_async(endpoint)
            .await
            .map_err(|error| LarkError::WebSocket(Box::new(error)))?;
        Ok(stream)
    }

    async fn handle_event_text(
        &self,
        text: &str,
        driver: std::sync::Arc<dyn ConversationDriver>,
    ) -> Result<(), LarkError> {
        let envelope: LarkEnvelope =
            serde_json::from_str(text).map_err(|error| LarkError::Protocol(error.to_string()))?;
        let Some(inbound) = self.normalize_inbound(envelope)? else {
            return Ok(());
        };

        for ready in self.push_debounced(inbound) {
            self.dispatch_inbound(driver.clone(), ready).await?;
        }

        Ok(())
    }

    #[allow(clippy::result_large_err)]
    fn normalize_inbound(
        &self,
        envelope: LarkEnvelope,
    ) -> Result<Option<AccessInbound>, LarkError> {
        let Some(event) = envelope.event else {
            return Ok(None);
        };

        if !self.mark_message_seen(&event.message.message_id) {
            debug!(message_id = %event.message.message_id, "dropping duplicate lark message");
            return Ok(None);
        }

        if event.message.chat_type != "p2p" && !self.mentions_bot(&event.message.mentions) {
            return Ok(None);
        }

        let text = parse_text_content(&event.message.content)?;
        let conversation_key = event
            .message
            .thread_id
            .clone()
            .unwrap_or_else(|| event.message.chat_id.clone());

        Ok(Some(AccessInbound {
            conversation_id: ConversationId(conversation_key),
            user_id: event
                .sender
                .sender_id
                .open_id
                .unwrap_or_else(|| "unknown-user".to_string()),
            text,
            route: ReplyRoute::Lark {
                chat_id: event.message.chat_id,
                thread_id: event.message.thread_id,
            },
            metadata: json!({ "message_id": event.message.message_id, "chat_type": event.message.chat_type }),
        }))
    }

    fn mentions_bot(&self, mentions: &[LarkMention]) -> bool {
        mentions.iter().any(|mention| {
            mention.is_bot == Some(true)
                || mention.id.as_ref().and_then(|id| id.open_id.as_deref())
                    == Some(self.config.app_id.as_str())
                || mention
                    .name
                    .as_deref()
                    .map(|name| {
                        let lowered = name.to_ascii_lowercase();
                        lowered.contains("bot") || lowered.contains("gemenr")
                    })
                    .unwrap_or(false)
        })
    }

    fn mark_message_seen(&self, message_id: &str) -> bool {
        let mut seen = self
            .seen_message_ids
            .lock()
            .expect("seen message set should not be poisoned");
        seen.insert(message_id.to_string())
    }

    fn push_debounced(&self, inbound: AccessInbound) -> Vec<AccessInbound> {
        let mut ready = Vec::new();
        let mut state = self
            .debounce
            .lock()
            .expect("debounce state should not be poisoned");
        let now = Instant::now();
        let debounce_window = Duration::from_millis(self.config.debounce_ms);

        let expired_keys = state
            .iter()
            .filter(|(_, pending)| now.duration_since(pending.last_update) >= debounce_window)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in expired_keys {
            if let Some(pending) = state.remove(&key) {
                ready.push(pending.into_inbound());
            }
        }

        match state.get_mut(&inbound.conversation_id.0) {
            Some(pending) => {
                pending.text_parts.push(inbound.text);
                pending.last_update = now;
            }
            None => {
                state.insert(
                    inbound.conversation_id.0.clone(),
                    PendingConversation::from_inbound(inbound, now),
                );
            }
        }

        ready
    }

    fn flush_all_debounced(&self) -> Vec<AccessInbound> {
        let mut state = self
            .debounce
            .lock()
            .expect("debounce state should not be poisoned");
        state
            .drain()
            .map(|(_, pending)| pending.into_inbound())
            .collect()
    }

    async fn dispatch_inbound(
        &self,
        driver: std::sync::Arc<dyn ConversationDriver>,
        inbound: AccessInbound,
    ) -> Result<(), LarkError> {
        let outbound = driver
            .handle(inbound)
            .await
            .map_err(|error| LarkError::Driver(error.to_string()))?;
        self.send(outbound)
            .await
            .map_err(|error| LarkError::Driver(error.to_string()))
    }
}

impl PendingConversation {
    fn from_inbound(inbound: AccessInbound, now: Instant) -> Self {
        Self {
            conversation_id: inbound.conversation_id,
            user_id: inbound.user_id,
            route: inbound.route,
            metadata: inbound.metadata,
            text_parts: vec![inbound.text],
            last_update: now,
        }
    }

    fn into_inbound(self) -> AccessInbound {
        AccessInbound {
            conversation_id: self.conversation_id,
            user_id: self.user_id,
            text: self.text_parts.join("\n"),
            route: self.route,
            metadata: self.metadata,
        }
    }
}

#[async_trait]
impl AccessAdapter for LarkAdapter {
    fn name(&self) -> &'static str {
        "lark"
    }

    async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
        let ReplyRoute::Lark { chat_id, thread_id } = outbound.route else {
            return Err(AccessError::Delivery(
                "lark adapter can only deliver ReplyRoute::Lark".to_string(),
            ));
        };
        let token = self
            .refresh_tenant_token()
            .await
            .map_err(|error| AccessError::Delivery(error.to_string()))?;
        let mut body = json!({
            "receive_id": chat_id,
            "msg_type": "text",
            "content": serde_json::to_string(&json!({"text": outbound.content}))
                .map_err(|error| AccessError::Delivery(error.to_string()))?,
        });
        if let Some(thread_id) = thread_id {
            body["thread_id"] = json!(thread_id);
        }

        self.http
            .post(LARK_SEND_URL)
            .bearer_auth(token)
            .query(&[("receive_id_type", "chat_id")])
            .json(&body)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| AccessError::Delivery(error.to_string()))?;
        Ok(())
    }
}

#[allow(clippy::result_large_err)]
fn parse_text_content(raw: &str) -> Result<String, LarkError> {
    let content: LarkTextContent =
        serde_json::from_str(raw).map_err(|error| LarkError::Protocol(error.to_string()))?;
    Ok(content.text)
}

#[async_trait]
impl crate::service::LarkRunLoop for LarkAdapter {
    async fn run(&self, driver: std::sync::Arc<dyn ConversationDriver>) -> Result<(), LarkError> {
        LarkAdapter::run(self, driver).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use async_trait::async_trait;
    use gemenr_core::{
        AccessAdapter, AccessError, AccessInbound, AccessOutbound, ConversationDriver,
        ConversationId, ReplyRoute,
    };
    use serde_json::json;

    use super::{CachedTenantToken, LarkAdapter, LarkConfig};

    #[test]
    fn token_ttl_parses_refresh_window() {
        let now = Instant::now();
        let cached = CachedTenantToken::from_ttl("token".to_string(), 300, now);

        assert_eq!(cached.token, "token");
        assert!(cached.expires_at > now);
        assert!(cached.refresh_at >= now);
        assert!(cached.refresh_at < cached.expires_at);
    }

    #[test]
    fn direct_message_normalizes_to_access_inbound() {
        let adapter = adapter();
        let inbound = adapter
            .normalize_inbound(
                serde_json::from_value(json!({
                    "event": {
                        "sender": {"sender_id": {"open_id": "ou_user"}},
                        "message": {
                            "message_id": "om_msg_1",
                            "chat_id": "oc_chat_1",
                            "chat_type": "p2p",
                            "content": "{\"text\":\"hello\"}",
                            "mentions": []
                        }
                    }
                }))
                .expect("event should deserialize"),
            )
            .expect("event should parse")
            .expect("event should normalize");

        assert_eq!(inbound.conversation_id.0, "oc_chat_1");
        assert_eq!(inbound.text, "hello");
        assert!(matches!(inbound.route, ReplyRoute::Lark { .. }));
    }

    #[test]
    fn mention_group_messages_require_bot_mention() {
        let adapter = adapter();
        let dropped = adapter
            .normalize_inbound(
                serde_json::from_value(json!({
                    "event": {
                        "sender": {"sender_id": {"open_id": "ou_user"}},
                        "message": {
                            "message_id": "om_msg_2",
                            "chat_id": "oc_chat_2",
                            "chat_type": "group",
                            "content": "{\"text\":\"hello\"}",
                            "mentions": []
                        }
                    }
                }))
                .expect("event should deserialize"),
            )
            .expect("event should parse");
        assert!(dropped.is_none());

        let forwarded = adapter
            .normalize_inbound(
                serde_json::from_value(json!({
                    "event": {
                        "sender": {"sender_id": {"open_id": "ou_user"}},
                        "message": {
                            "message_id": "om_msg_3",
                            "chat_id": "oc_chat_2",
                            "chat_type": "group",
                            "content": "{\"text\":\"hello\"}",
                            "mentions": [{"is_bot": true, "name": "Gemenr"}]
                        }
                    }
                }))
                .expect("event should deserialize"),
            )
            .expect("event should parse")
            .expect("event should normalize");
        assert_eq!(forwarded.text, "hello");
    }

    #[test]
    fn duplicate_message_ids_are_ignored() {
        let adapter = adapter();
        assert!(adapter.mark_message_seen("om_msg_dup"));
        assert!(!adapter.mark_message_seen("om_msg_dup"));
    }

    #[test]
    fn debounce_merges_messages_in_same_conversation() {
        let adapter = adapter();
        let first = AccessInbound {
            conversation_id: ConversationId("conv-1".to_string()),
            user_id: "user".to_string(),
            text: "hello".to_string(),
            route: ReplyRoute::Stdio,
            metadata: json!({}),
        };
        let second = AccessInbound {
            conversation_id: ConversationId("conv-1".to_string()),
            user_id: "user".to_string(),
            text: "world".to_string(),
            route: ReplyRoute::Stdio,
            metadata: json!({}),
        };

        assert!(adapter.push_debounced(first).is_empty());
        assert!(adapter.push_debounced(second).is_empty());
        let flushed = adapter.flush_all_debounced();

        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].text, "hello\nworld");
    }

    fn adapter() -> LarkAdapter {
        LarkAdapter::new(LarkConfig {
            app_id: "bot-app".to_string(),
            app_secret: "secret".to_string(),
            ws_endpoint: Some("ws://127.0.0.1:9".to_string()),
            debounce_ms: 500,
        })
    }

    struct EchoDriver;

    #[async_trait]
    impl ConversationDriver for EchoDriver {
        async fn handle(&self, inbound: AccessInbound) -> Result<AccessOutbound, AccessError> {
            Ok(AccessOutbound {
                conversation_id: inbound.conversation_id,
                route: inbound.route,
                content: inbound.text,
                metadata: json!({}),
            })
        }
    }

    #[tokio::test]
    async fn send_rejects_non_lark_routes() {
        let adapter = adapter();
        let error = adapter
            .send(AccessOutbound {
                conversation_id: ConversationId("conv-1".to_string()),
                route: ReplyRoute::Stdio,
                content: "hello".to_string(),
                metadata: json!({}),
            })
            .await
            .expect_err("non-lark route should fail");
        assert!(matches!(error, AccessError::Delivery(_)));
    }

    #[tokio::test]
    async fn flushes_debounce_buffer_on_run_end_when_stream_fails() {
        let adapter = adapter();
        let inbound = AccessInbound {
            conversation_id: ConversationId("conv-2".to_string()),
            user_id: "user".to_string(),
            text: "queued".to_string(),
            route: ReplyRoute::Lark {
                chat_id: "chat".to_string(),
                thread_id: None,
            },
            metadata: json!({}),
        };
        assert!(adapter.push_debounced(inbound).is_empty());
        let flushed = adapter.flush_all_debounced();
        assert_eq!(flushed.len(), 1);
    }

    #[tokio::test]
    async fn echo_driver_is_object_safe_for_adapter_run_path() {
        let driver: Arc<dyn ConversationDriver> = Arc::new(EchoDriver);
        let outbound = driver
            .handle(AccessInbound {
                conversation_id: ConversationId("conv-3".to_string()),
                user_id: "user".to_string(),
                text: "hello".to_string(),
                route: ReplyRoute::Stdio,
                metadata: json!({}),
            })
            .await
            .expect("driver should echo");
        assert_eq!(outbound.content, "hello");
    }
}
