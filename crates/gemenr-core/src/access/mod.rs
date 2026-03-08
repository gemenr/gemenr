//! Shared access-layer message models and adapter contracts.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod adapter;
pub mod router;

pub use adapter::{AccessAdapter, ConversationDriver};
pub use router::AccessRouter;

/// Errors that can occur while routing or delivering access-layer messages.
#[derive(Debug, Error)]
pub enum AccessError {
    /// The provided textual route string is invalid.
    #[error("invalid access route: {0}")]
    InvalidRoute(String),
    /// No adapter is registered for the requested route.
    #[error("no adapter registered for route: {0}")]
    AdapterUnavailable(&'static str),
    /// Driver execution failed.
    #[error("conversation driver failed: {0}")]
    Driver(String),
    /// Adapter delivery failed.
    #[error("access delivery failed: {0}")]
    Delivery(String),
}

/// Stable identifier for one long-lived conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(pub String);

/// Where a reply should be sent back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplyRoute {
    /// Standard IO session.
    Stdio,
    /// Lark chat or thread target.
    Lark {
        /// Lark chat identifier.
        chat_id: String,
        /// Optional thread identifier inside the chat.
        thread_id: Option<String>,
    },
}

/// A normalized inbound user message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccessInbound {
    /// Access-layer conversation identifier.
    pub conversation_id: ConversationId,
    /// Stable user identifier from the source transport.
    pub user_id: String,
    /// Plain-text message content.
    pub text: String,
    /// Reply route for subsequent assistant responses.
    pub route: ReplyRoute,
    /// Transport-specific metadata.
    pub metadata: serde_json::Value,
}

/// A normalized outbound assistant response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccessOutbound {
    /// Access-layer conversation identifier.
    pub conversation_id: ConversationId,
    /// Destination route for the response.
    pub route: ReplyRoute,
    /// Assistant response content.
    pub content: String,
    /// Transport-specific metadata.
    pub metadata: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{AccessInbound, ConversationId, ReplyRoute};

    #[test]
    fn access_inbound_round_trips_through_json() {
        let inbound = AccessInbound {
            conversation_id: ConversationId("conv-42".to_string()),
            user_id: "user-7".to_string(),
            text: "hello".to_string(),
            route: ReplyRoute::Lark {
                chat_id: "chat-1".to_string(),
                thread_id: Some("thread-9".to_string()),
            },
            metadata: json!({"source": "lark", "mentions": ["bot"]}),
        };

        let encoded = serde_json::to_string(&inbound).expect("message should serialize");
        let decoded: AccessInbound =
            serde_json::from_str(&encoded).expect("message should deserialize");

        assert_eq!(decoded, inbound);
    }

    #[test]
    fn lark_reply_route_serializes_chat_and_thread_ids() {
        let route = ReplyRoute::Lark {
            chat_id: "chat-100".to_string(),
            thread_id: Some("thread-200".to_string()),
        };

        let encoded = serde_json::to_value(&route).expect("route should serialize");

        assert_eq!(encoded["Lark"]["chat_id"], json!("chat-100"));
        assert_eq!(encoded["Lark"]["thread_id"], json!("thread-200"));
    }
}
