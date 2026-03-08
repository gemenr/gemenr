use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Unique identifier for an event in the tape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub String);

impl EventId {
    /// Creates a new unique event identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for a session (one task or conversation).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Creates a new unique session identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for a turn within a session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub String);

impl TurnId {
    /// Creates a new unique turn identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for TurnId {
    fn default() -> Self {
        Self::new()
    }
}

/// Categories of events in the runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// User input received.
    UserInput,
    /// Model response received.
    ModelResponse,
    /// Tool execution started.
    ToolStarted,
    /// Tool execution completed successfully.
    ToolCompleted,
    /// Tool execution failed.
    ToolFailed,
    /// Tool execution was denied before it started.
    ToolDenied,
    /// Tool execution timed out.
    ToolTimedOut,
    /// Anchor (stage boundary) created.
    AnchorCreated,
    /// Context was summarized due to token budget.
    ContextSummarized,
    /// Turn completed successfully.
    TurnCompleted,
    /// Turn terminated with an error.
    TurnFailed,
    /// Custom event kind for extensibility.
    Custom(String),
}

/// Unified event envelope — every event in the system is wrapped in this structure.
///
/// Events are appended to the tape and form the complete audit trail of a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventEnvelope {
    /// Unique identifier for this event.
    pub event_id: EventId,
    /// Session this event belongs to.
    pub session_id: SessionId,
    /// Turn within the session (None for session-level events).
    pub turn_id: Option<TurnId>,
    /// Event kind identifier (e.g., "tool.started", "model.response", "anchor.created").
    pub kind: EventKind,
    /// When this event occurred.
    pub timestamp: DateTime<Utc>,
    /// Event-specific payload data.
    pub payload: Value,
}

impl EventEnvelope {
    /// Creates a new event envelope with an auto-generated event ID and current timestamp.
    #[must_use]
    pub fn new(
        session_id: SessionId,
        turn_id: Option<TurnId>,
        kind: EventKind,
        payload: Value,
    ) -> Self {
        Self {
            event_id: EventId::new(),
            session_id,
            turn_id,
            kind,
            timestamp: Utc::now(),
            payload,
        }
    }
}

/// Hint for session routing — used by the runtime to find or create the right session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionHint {
    /// Workspace identifier.
    pub workspace_id: String,
    /// Optional existing session to route to.
    pub session_id: Option<String>,
}

/// A part of a message — messages can contain multiple parts (text, tool calls, tool results).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagePart {
    /// Plain text content.
    Text {
        /// The text content.
        text: String,
    },
    /// A tool call request from the model.
    ToolCall {
        /// Unique identifier for this tool call (used for round-trip matching).
        id: String,
        /// Name of the tool to invoke.
        name: String,
        /// Arguments to pass to the tool (JSON object).
        arguments: Value,
    },
    /// Result of a tool execution.
    ToolResult {
        /// The tool call ID this result corresponds to.
        call_id: String,
        /// The output content from the tool.
        content: String,
        /// Whether the tool execution resulted in an error.
        is_error: bool,
    },
}

/// Input commands to the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// User sends a new turn with message parts.
    UserTurn {
        /// Content of the user's input.
        items: Vec<MessagePart>,
        /// Hint for session routing.
        session_hint: SessionHint,
    },
    /// Interrupt the currently executing turn.
    Interrupt {
        /// Which session to interrupt.
        session_id: SessionId,
    },
    /// Close a session and release its resources.
    Close {
        /// Which session to close.
        session_id: SessionId,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        EventEnvelope, EventId, EventKind, MessagePart, Op, SessionHint, SessionId, TurnId,
    };
    use serde_json::json;

    #[test]
    fn event_id_generation_is_unique() {
        let first = EventId::new();
        let second = EventId::new();

        assert_ne!(first, second);
    }

    #[test]
    fn event_envelope_round_trips_through_json() {
        let envelope = EventEnvelope {
            event_id: EventId::new(),
            session_id: SessionId::new(),
            turn_id: Some(TurnId::new()),
            kind: EventKind::ToolCompleted,
            timestamp: chrono::Utc::now(),
            payload: json!({"result": "ok", "count": 1}),
        };

        let json = serde_json::to_string(&envelope).expect("event envelope should serialize");
        let decoded: EventEnvelope =
            serde_json::from_str(&json).expect("event envelope should deserialize");

        assert_eq!(decoded, envelope);
    }

    #[test]
    fn message_part_text_serializes_with_internal_type_tag() {
        let value = serde_json::to_value(MessagePart::Text {
            text: "hello".to_string(),
        })
        .expect("text message part should serialize");

        assert_eq!(value, json!({"type": "text", "text": "hello"}));
    }

    #[test]
    fn message_part_tool_call_serializes_expected_fields() {
        let value = serde_json::to_value(MessagePart::ToolCall {
            id: "call-1".to_string(),
            name: "list_files".to_string(),
            arguments: json!({"path": "."}),
        })
        .expect("tool call message part should serialize");

        assert_eq!(
            value,
            json!({
                "type": "tool_call",
                "id": "call-1",
                "name": "list_files",
                "arguments": {"path": "."}
            })
        );
    }

    #[test]
    fn message_part_tool_result_serializes_expected_fields() {
        let value = serde_json::to_value(MessagePart::ToolResult {
            call_id: "call-1".to_string(),
            content: "done".to_string(),
            is_error: false,
        })
        .expect("tool result message part should serialize");

        assert_eq!(
            value,
            json!({
                "type": "tool_result",
                "call_id": "call-1",
                "content": "done",
                "is_error": false
            })
        );
    }

    #[test]
    fn op_variants_round_trip_through_json() {
        let user_turn = Op::UserTurn {
            items: vec![MessagePart::Text {
                text: "hello".to_string(),
            }],
            session_hint: SessionHint {
                workspace_id: "workspace-1".to_string(),
                session_id: Some("session-1".to_string()),
            },
        };
        let interrupt = Op::Interrupt {
            session_id: SessionId::new(),
        };
        let close = Op::Close {
            session_id: SessionId::new(),
        };

        for op in [user_turn, interrupt, close] {
            let json = serde_json::to_string(&op).expect("op should serialize");
            let decoded: Op = serde_json::from_str(&json).expect("op should deserialize");
            assert_eq!(decoded, op);
        }
    }

    #[test]
    fn event_kind_serializes_to_snake_case() {
        let json =
            serde_json::to_string(&EventKind::ToolStarted).expect("event kind should serialize");

        assert_eq!(json, r#""tool_started""#);
    }

    #[test]
    fn event_kind_includes_turn_terminal_states() {
        assert_eq!(
            serde_json::to_value(EventKind::TurnCompleted)
                .expect("turn completed should serialize"),
            json!("turn_completed")
        );
        assert_eq!(
            serde_json::to_value(EventKind::TurnFailed).expect("turn failed should serialize"),
            json!("turn_failed")
        );
    }

    #[test]
    fn event_kind_includes_tool_governance_states() {
        assert_eq!(
            serde_json::to_value(EventKind::ToolDenied).expect("tool denied should serialize"),
            json!("tool_denied")
        );
        assert_eq!(
            serde_json::to_value(EventKind::ToolTimedOut).expect("tool timed out should serialize"),
            json!("tool_timed_out")
        );
    }
}
