use async_trait::async_trait;

use super::{AccessError, AccessInbound, AccessOutbound, ReplyRoute};

/// Handles one normalized inbound message and returns one outbound response.
#[async_trait]
pub trait ConversationDriver: Send + Sync {
    /// Process one normalized inbound message.
    async fn handle(&self, inbound: AccessInbound) -> Result<AccessOutbound, AccessError>;
}

/// Sends normalized outbound messages to a concrete transport.
#[async_trait]
pub trait AccessAdapter: Send + Sync {
    /// Stable adapter name for logs and metrics.
    fn name(&self) -> &'static str;

    /// Stable route scheme owned by this adapter.
    fn scheme(&self) -> &'static str;

    /// Parse one textual route if it belongs to this adapter.
    fn parse_route(&self, raw: &str) -> Result<Option<ReplyRoute>, AccessError>;

    /// Deliver one outbound message.
    async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError>;
}
