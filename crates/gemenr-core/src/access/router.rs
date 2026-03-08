use std::sync::Arc;

use super::{AccessAdapter, AccessError, AccessOutbound, ReplyRoute};

/// Routes normalized outbound messages to registered transport adapters.
pub struct AccessRouter {
    stdio: Option<Arc<dyn AccessAdapter>>,
    lark: Option<Arc<dyn AccessAdapter>>,
}

impl AccessRouter {
    /// Create an empty access router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stdio: None,
            lark: None,
        }
    }

    /// Register the stdio adapter.
    #[must_use]
    pub fn with_stdio(mut self, adapter: Arc<dyn AccessAdapter>) -> Self {
        self.stdio = Some(adapter);
        self
    }

    /// Register the Lark adapter.
    #[must_use]
    pub fn with_lark(mut self, adapter: Arc<dyn AccessAdapter>) -> Self {
        self.lark = Some(adapter);
        self
    }

    /// Parse a textual route such as `stdio:` or `lark:oc_xxx`.
    pub fn parse_route(&self, raw: &str) -> Result<ReplyRoute, AccessError> {
        if raw == "stdio:" {
            return Ok(ReplyRoute::Stdio);
        }

        if let Some(target) = raw.strip_prefix("lark:") {
            if target.is_empty() {
                return Err(AccessError::InvalidRoute(raw.to_string()));
            }

            let (chat_id, thread_id) = match target.split_once('/') {
                Some((chat_id, thread_id)) if !chat_id.is_empty() && !thread_id.is_empty() => {
                    (chat_id.to_string(), Some(thread_id.to_string()))
                }
                Some(_) => return Err(AccessError::InvalidRoute(raw.to_string())),
                None => (target.to_string(), None),
            };

            return Ok(ReplyRoute::Lark { chat_id, thread_id });
        }

        Err(AccessError::InvalidRoute(raw.to_string()))
    }

    /// Deliver one outbound message to the adapter selected by its route.
    pub async fn deliver(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
        match outbound.route {
            ReplyRoute::Stdio => {
                self.stdio
                    .as_ref()
                    .ok_or(AccessError::AdapterUnavailable("stdio"))?
                    .send(outbound)
                    .await
            }
            ReplyRoute::Lark { .. } => {
                self.lark
                    .as_ref()
                    .ok_or(AccessError::AdapterUnavailable("lark"))?
                    .send(outbound)
                    .await
            }
        }
    }
}

impl Default for AccessRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;

    use super::AccessRouter;
    use crate::access::{AccessAdapter, AccessError, AccessOutbound, ConversationId, ReplyRoute};

    #[derive(Default)]
    struct RecordingAdapter {
        delivered: Mutex<Vec<AccessOutbound>>,
    }

    #[async_trait]
    impl AccessAdapter for RecordingAdapter {
        fn name(&self) -> &'static str {
            "recording"
        }

        async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
            self.delivered
                .lock()
                .expect("delivery lock should not be poisoned")
                .push(outbound);
            Ok(())
        }
    }

    #[test]
    fn parses_stdio_route() {
        let router = AccessRouter::new();

        assert_eq!(
            router.parse_route("stdio:").expect("route should parse"),
            ReplyRoute::Stdio
        );
    }

    #[test]
    fn parses_lark_route() {
        let router = AccessRouter::new();

        assert_eq!(
            router
                .parse_route("lark:oc_xxx")
                .expect("route should parse"),
            ReplyRoute::Lark {
                chat_id: "oc_xxx".to_string(),
                thread_id: None,
            }
        );
    }

    #[test]
    fn rejects_unknown_route_scheme() {
        let router = AccessRouter::new();

        let error = router
            .parse_route("discord:123")
            .expect_err("route should be invalid");
        assert!(matches!(error, AccessError::InvalidRoute(route) if route == "discord:123"));
    }

    #[test]
    fn rejects_lark_route_with_empty_thread_id() {
        let router = AccessRouter::new();

        let error = router
            .parse_route("lark:oc_xxx/")
            .expect_err("route should be invalid");
        assert!(matches!(error, AccessError::InvalidRoute(route) if route == "lark:oc_xxx/"));
    }

    #[tokio::test]
    async fn delivers_stdio_messages() {
        let adapter = Arc::new(RecordingAdapter::default());
        let router = AccessRouter::new().with_stdio(adapter.clone());
        let outbound = AccessOutbound {
            conversation_id: ConversationId("conv-1".to_string()),
            route: ReplyRoute::Stdio,
            content: "hello".to_string(),
            metadata: json!({}),
        };

        router
            .deliver(outbound.clone())
            .await
            .expect("delivery should succeed");

        let delivered = adapter
            .delivered
            .lock()
            .expect("delivery lock should not be poisoned")
            .clone();
        assert_eq!(delivered, vec![outbound]);
    }
}
