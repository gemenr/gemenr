use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::ConversationDriver;
use thiserror::Error;
use tracing::warn;

use crate::lark::LarkError;

/// Adapter abstraction used by the IM service loop.
#[async_trait]
pub trait LarkRunLoop: Send + Sync {
    /// Run one long-connection session until it closes.
    async fn run(&self, driver: Arc<dyn ConversationDriver>) -> Result<(), LarkError>;
}

/// Interface used to reclaim idle runtime state.
#[async_trait]
pub trait IdleCollector: Send + Sync {
    /// Hibernate idle conversations and return the number reclaimed.
    async fn hibernate_idle(&self, max_idle: Duration) -> usize;
}

/// Errors returned by the IM service loop.
#[derive(Debug, Error)]
pub enum ServiceError {
    /// The adapter run loop failed too many times.
    #[error("lark service stopped after repeated failures: {0}")]
    Adapter(String),
}

/// Orchestrates reconnection and idle reclamation for the Lark entrypoint.
pub struct LarkService<A, C> {
    adapter: Arc<A>,
    driver: Arc<dyn ConversationDriver>,
    collector: Arc<C>,
    idle_timeout: Duration,
    reconnect_base_delay: Duration,
    reconnect_max_delay: Duration,
}

impl<A, C> LarkService<A, C>
where
    A: LarkRunLoop,
    C: IdleCollector,
{
    /// Create a new Lark service loop.
    #[must_use]
    pub fn new(adapter: Arc<A>, driver: Arc<dyn ConversationDriver>, collector: Arc<C>) -> Self {
        Self {
            adapter,
            driver,
            collector,
            idle_timeout: Duration::from_secs(600),
            reconnect_base_delay: Duration::from_secs(1),
            reconnect_max_delay: Duration::from_secs(30),
        }
    }

    /// Override the idle runtime timeout.
    #[must_use]
    pub fn idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.idle_timeout = idle_timeout;
        self
    }

    /// Override reconnect backoff settings.
    #[must_use]
    pub fn reconnect_backoff(mut self, base_delay: Duration, max_delay: Duration) -> Self {
        self.reconnect_base_delay = base_delay;
        self.reconnect_max_delay = max_delay;
        self
    }

    /// Run forever until the process is terminated.
    pub async fn run(&self) -> Result<(), ServiceError> {
        self.run_attempts(None).await
    }

    async fn run_attempts(&self, max_attempts: Option<usize>) -> Result<(), ServiceError> {
        let mut attempt = 0usize;
        let mut backoff = self.reconnect_base_delay;

        loop {
            attempt += 1;
            let _ = self.collector.hibernate_idle(self.idle_timeout).await;

            match self.adapter.run(Arc::clone(&self.driver)).await {
                Ok(()) => {
                    backoff = self.reconnect_base_delay;
                }
                Err(error) => {
                    warn!(attempt, delay_ms = backoff.as_millis() as u64, error = %error, "lark connection failed; retrying");
                    if max_attempts.is_some_and(|limit| attempt >= limit) {
                        return Err(ServiceError::Adapter(error.to_string()));
                    }
                    tokio::time::sleep(backoff).await;
                    backoff = next_backoff(backoff, self.reconnect_max_delay);
                }
            }
        }
    }
}

fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.saturating_mul(2).min(max)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use gemenr_core::{
        AccessError, AccessInbound, AccessOutbound, ConversationDriver, ConversationId,
    };
    use serde_json::json;

    use super::{IdleCollector, LarkRunLoop, LarkService, ServiceError};
    use crate::lark::LarkError;

    struct MockAdapter {
        failures_before_success: AtomicUsize,
        attempts: AtomicUsize,
    }

    #[async_trait]
    impl LarkRunLoop for MockAdapter {
        async fn run(&self, _driver: Arc<dyn ConversationDriver>) -> Result<(), LarkError> {
            self.attempts.fetch_add(1, Ordering::Relaxed);
            if self.failures_before_success.fetch_sub(1, Ordering::Relaxed) > 0 {
                return Err(LarkError::Protocol("boom".to_string()));
            }
            Ok(())
        }
    }

    struct MockCollector {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl IdleCollector for MockCollector {
        async fn hibernate_idle(&self, _max_idle: Duration) -> usize {
            self.calls.fetch_add(1, Ordering::Relaxed);
            0
        }
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
    async fn reconnects_with_backoff_instead_of_busy_loop() {
        let adapter = Arc::new(MockAdapter {
            failures_before_success: AtomicUsize::new(2),
            attempts: AtomicUsize::new(0),
        });
        let collector = Arc::new(MockCollector {
            calls: AtomicUsize::new(0),
        });
        let service = LarkService::new(adapter.clone(), Arc::new(EchoDriver), collector)
            .reconnect_backoff(Duration::from_millis(1), Duration::from_millis(4));

        let error = service
            .run_attempts(Some(2))
            .await
            .expect_err("service should stop after configured attempts");

        assert!(matches!(error, ServiceError::Adapter(_)));
        assert_eq!(adapter.attempts.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn triggers_idle_collection_on_each_loop() {
        let adapter = Arc::new(MockAdapter {
            failures_before_success: AtomicUsize::new(1),
            attempts: AtomicUsize::new(0),
        });
        let collector = Arc::new(MockCollector {
            calls: AtomicUsize::new(0),
        });
        let service = LarkService::new(adapter, Arc::new(EchoDriver), collector.clone())
            .idle_timeout(Duration::from_secs(30))
            .reconnect_backoff(Duration::from_millis(1), Duration::from_millis(2));

        let _ = service.run_attempts(Some(1)).await;

        assert_eq!(collector.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn next_backoff_doubles_until_maximum() {
        assert_eq!(
            super::next_backoff(Duration::from_secs(1), Duration::from_secs(10)),
            Duration::from_secs(2)
        );
        assert_eq!(
            super::next_backoff(Duration::from_secs(8), Duration::from_secs(10)),
            Duration::from_secs(10)
        );
    }

    #[test]
    fn echo_driver_keeps_routes_stable() {
        let inbound = AccessInbound {
            conversation_id: ConversationId("conv".to_string()),
            user_id: "user".to_string(),
            text: "hello".to_string(),
            route: crate::lark::LarkAdapter::route("chat", Some("thread".to_string())),
            metadata: json!({}),
        };
        let driver = EchoDriver;
        let runtime = tokio::runtime::Runtime::new().expect("runtime should build");
        let outbound = runtime
            .block_on(driver.handle(inbound))
            .expect("driver should echo");
        assert_eq!(outbound.route.scheme, "lark");
        assert_eq!(outbound.route.target, "chat");
    }
}
