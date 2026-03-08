use std::collections::HashMap;
use std::sync::Arc;

use super::{AccessAdapter, AccessError, AccessOutbound, ReplyRoute};

/// Routes normalized outbound messages to registered transport adapters.
pub struct AccessRouter {
    adapters: HashMap<String, Arc<dyn AccessAdapter>>,
}

impl AccessRouter {
    /// Create an empty access router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }

    /// Register one adapter by its declared scheme.
    #[must_use]
    pub fn with_adapter(mut self, adapter: Arc<dyn AccessAdapter>) -> Self {
        let scheme = adapter.scheme().to_string();
        self.adapters.insert(scheme, adapter);
        self
    }

    /// Parse a textual route such as `stdio:` or `lark:oc_xxx`.
    pub fn parse_route(&self, raw: &str) -> Result<ReplyRoute, AccessError> {
        let scheme = raw
            .split_once(':')
            .map(|(scheme, _)| scheme)
            .filter(|scheme| !scheme.is_empty())
            .ok_or_else(|| AccessError::InvalidRoute(raw.to_string()))?;

        let adapter = self
            .adapters
            .get(scheme)
            .ok_or_else(|| AccessError::InvalidRoute(raw.to_string()))?;

        adapter
            .parse_route(raw)?
            .ok_or_else(|| AccessError::InvalidRoute(raw.to_string()))
    }

    /// Deliver one outbound message to the adapter selected by its route.
    pub async fn deliver(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
        let scheme = outbound.route.scheme.clone();
        let adapter = self
            .adapters
            .get(&scheme)
            .ok_or_else(|| AccessError::AdapterUnavailable(scheme.clone()))?;
        adapter.send(outbound).await
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

        fn scheme(&self) -> &'static str {
            "recording"
        }

        fn parse_route(&self, raw: &str) -> Result<Option<ReplyRoute>, AccessError> {
            let Some(target) = raw.strip_prefix("recording:") else {
                return Ok(None);
            };

            if target.is_empty() {
                return Err(AccessError::InvalidRoute(raw.to_string()));
            }

            Ok(Some(ReplyRoute::new("recording", target, json!({}))))
        }

        async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
            self.delivered
                .lock()
                .expect("delivery lock should not be poisoned")
                .push(outbound);
            Ok(())
        }
    }

    #[derive(Default)]
    struct StdioStub;

    impl StdioStub {
        fn route() -> ReplyRoute {
            ReplyRoute::new("stdio", "", json!({}))
        }
    }

    #[async_trait]
    impl AccessAdapter for StdioStub {
        fn name(&self) -> &'static str {
            "stdio"
        }

        fn scheme(&self) -> &'static str {
            "stdio"
        }

        fn parse_route(&self, raw: &str) -> Result<Option<ReplyRoute>, AccessError> {
            Ok((raw == "stdio:").then(Self::route))
        }

        async fn send(&self, _outbound: AccessOutbound) -> Result<(), AccessError> {
            Ok(())
        }
    }

    #[test]
    fn parses_stdio_route() {
        let router = AccessRouter::new().with_adapter(Arc::new(StdioStub));

        assert_eq!(
            router.parse_route("stdio:").expect("route should parse"),
            StdioStub::route()
        );
    }

    #[test]
    fn parses_registered_adapter_route() {
        let router = AccessRouter::new().with_adapter(Arc::new(RecordingAdapter::default()));

        assert_eq!(
            router
                .parse_route("recording:target-1")
                .expect("route should parse"),
            ReplyRoute::new("recording", "target-1", json!({}))
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
    fn rejects_registered_route_with_invalid_payload() {
        let router = AccessRouter::new().with_adapter(Arc::new(RecordingAdapter::default()));

        let error = router
            .parse_route("recording:")
            .expect_err("route should be invalid");
        assert!(matches!(error, AccessError::InvalidRoute(route) if route == "recording:"));
    }

    #[tokio::test]
    async fn delivers_registered_messages() {
        let adapter = Arc::new(RecordingAdapter::default());
        let router = AccessRouter::new().with_adapter(adapter.clone());
        let outbound = AccessOutbound {
            conversation_id: ConversationId("conv-1".to_string()),
            route: ReplyRoute::new("recording", "target-1", json!({})),
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

    #[tokio::test]
    async fn rejects_delivery_without_registered_adapter() {
        let router = AccessRouter::new();
        let error = router
            .deliver(AccessOutbound {
                conversation_id: ConversationId("conv-1".to_string()),
                route: ReplyRoute::new("missing", "target-1", json!({})),
                content: "hello".to_string(),
                metadata: json!({}),
            })
            .await
            .expect_err("delivery should fail");

        assert!(matches!(error, AccessError::AdapterUnavailable(route) if route == "missing"));
    }
}
