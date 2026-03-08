use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// System prompt that sets assistant behavior.
    System,
    /// Message from the user.
    User,
    /// Response from the assistant.
    Assistant,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: ChatRole,
    /// The text content of the message.
    pub content: String,
    /// Optional metadata for structured provider-specific annotations.
    ///
    /// Native tool dispatch uses this to preserve tool call and tool result
    /// information while keeping the shared message format provider-agnostic.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl ChatMessage {
    /// Creates a new chat message with the given role and content.
    #[must_use]
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Creates a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    /// Creates a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    /// Creates an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    /// Adds a metadata entry to the message.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, ChatRole};

    #[test]
    fn constructors_create_expected_messages() {
        let created = ChatMessage::new(ChatRole::User, "hello");
        let system = ChatMessage::system("rules");
        let user = ChatMessage::user("question");
        let assistant = ChatMessage::assistant("answer");

        assert_eq!(created.role, ChatRole::User);
        assert_eq!(created.content, "hello");
        assert!(created.metadata.is_empty());

        assert_eq!(system.role, ChatRole::System);
        assert_eq!(system.content, "rules");
        assert!(system.metadata.is_empty());

        assert_eq!(user.role, ChatRole::User);
        assert_eq!(user.content, "question");
        assert!(user.metadata.is_empty());

        assert_eq!(assistant.role, ChatRole::Assistant);
        assert_eq!(assistant.content, "answer");
        assert!(assistant.metadata.is_empty());
    }

    #[test]
    fn message_round_trips_through_json() {
        let message = ChatMessage::assistant("done");

        let json = serde_json::to_string(&message).expect("message should serialize");
        assert_eq!(json, r#"{"role":"assistant","content":"done"}"#);

        let decoded: ChatMessage = serde_json::from_str(&json).expect("message should deserialize");
        assert_eq!(decoded, message);
    }

    #[test]
    fn roles_serialize_to_lowercase_strings() {
        let system = serde_json::to_string(&ChatRole::System).expect("system should serialize");
        let user = serde_json::to_string(&ChatRole::User).expect("user should serialize");
        let assistant =
            serde_json::to_string(&ChatRole::Assistant).expect("assistant should serialize");

        assert_eq!(system, r#""system""#);
        assert_eq!(user, r#""user""#);
        assert_eq!(assistant, r#""assistant""#);
    }

    #[test]
    fn metadata_round_trips_when_present() {
        let message = ChatMessage::assistant("done").with_metadata("tool_calls", "[]");

        let json = serde_json::to_string(&message).expect("message should serialize");
        assert_eq!(
            json,
            r#"{"role":"assistant","content":"done","metadata":{"tool_calls":"[]"}}"#
        );

        let decoded: ChatMessage = serde_json::from_str(&json).expect("message should deserialize");
        assert_eq!(decoded, message);
    }
}
