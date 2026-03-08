use async_trait::async_trait;

use super::{AccessError, AccessInbound, AccessOutbound};

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

    /// Deliver one outbound message.
    async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError>;
}
