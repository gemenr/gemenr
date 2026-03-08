//! Context persistence primitives for runtime state reconstruction.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::message::ChatMessage;
use crate::model::ToolCall;
use crate::protocol::{
    AssistantToolCallsPayload, EventEnvelope, EventKind, SessionId, ToolResultPayload,
};

/// SOUL.md persistence and reload support.
pub mod soul;
/// Tape storage backends and anchor loading.
pub mod tape;

pub use soul::{SoulError, SoulManager};
pub use tape::{AnchorEntry, InMemoryTapeStore, JsonlTapeStore, TapeError, TapeStore};

/// Token budget configuration for context building.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Maximum token count for context.
    pub max_tokens: usize,
    /// Threshold ratio (0.0-1.0) to trigger summarization.
    pub threshold: f64,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_tokens: 100_000,
            threshold: 0.7,
        }
    }
}

/// Result of context building.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextBuildResult {
    /// Context is within budget and ready to use.
    Ready(Vec<ChatMessage>),
    /// Context exceeds the configured threshold and should be summarized.
    NeedsSummary {
        /// Messages that should be summarized before continuing.
        messages: Vec<ChatMessage>,
    },
}

/// Unique identifier for an anchor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AnchorId(pub String);

/// Manages session context, tape persistence, and anchor boundaries.
pub struct ContextManager {
    /// Session identifier.
    session_id: SessionId,
    /// In-memory copy of events since the last anchor.
    events: Vec<EventEnvelope>,
    /// Current anchor entry, if one exists.
    current_anchor: Option<AnchorEntry>,
    /// Tape storage backend.
    tape_store: Arc<dyn TapeStore>,
    /// Shared SOUL.md manager.
    soul: Arc<RwLock<SoulManager>>,
}

impl ContextManager {
    /// Create a new context manager for a session.
    #[must_use]
    pub fn new(
        session_id: SessionId,
        tape_store: Arc<dyn TapeStore>,
        soul: Arc<RwLock<SoulManager>>,
    ) -> Self {
        Self {
            session_id,
            events: Vec::new(),
            current_anchor: None,
            tape_store,
            soul,
        }
    }

    /// Append an event to the tape and in-memory log.
    pub async fn append(&mut self, event: EventEnvelope) -> Result<(), TapeError> {
        self.tape_store
            .append(&self.session_id, event.clone())
            .await?;
        self.events.push(event);
        Ok(())
    }

    /// Create an anchor with a summary and clear buffered events.
    pub async fn create_anchor(&mut self, summary: String) -> Result<AnchorId, TapeError> {
        let anchor_id = AnchorId(Uuid::new_v4().to_string());
        let event = EventEnvelope::new(
            self.session_id.clone(),
            None,
            EventKind::AnchorCreated,
            json!({
                "anchor_id": anchor_id.0,
                "summary": summary,
            }),
        );

        self.tape_store
            .append(&self.session_id, event.clone())
            .await?;

        self.current_anchor = Some(AnchorEntry {
            anchor_id: anchor_id.0.clone(),
            summary,
            event,
        });
        self.events.clear();

        Ok(anchor_id)
    }

    /// Build chat context from buffered events and the configured budget.
    #[must_use]
    pub fn build_context(&self, budget: &TokenBudget) -> ContextBuildResult {
        let mut messages =
            Vec::with_capacity(self.events.len() + usize::from(self.current_anchor.is_some()));
        if let Some(anchor) = &self.current_anchor {
            messages.push(ChatMessage::system(format!(
                "Summary of earlier context:\n{}",
                anchor.summary
            )));
        }
        messages.extend(events_to_messages(&self.events));

        if estimated_tokens(&messages) as f64 > budget.max_tokens as f64 * budget.threshold {
            ContextBuildResult::NeedsSummary { messages }
        } else {
            ContextBuildResult::Ready(messages)
        }
    }

    /// Apply a summary by creating a new anchor.
    pub async fn apply_summary(&mut self, summary: String) -> Result<AnchorId, TapeError> {
        self.create_anchor(summary).await
    }

    /// Restore the latest anchor and post-anchor events from tape.
    pub async fn restore_from_tape(&mut self) -> Result<(), TapeError> {
        self.current_anchor = self.tape_store.load_last_anchor(&self.session_id).await?;
        self.events = self.tape_store.load_since_anchor(&self.session_id).await?;
        Ok(())
    }

    /// Return the current SOUL.md content.
    pub async fn soul_content(&self) -> String {
        self.soul.read().await.content().to_string()
    }

    /// Return the latest SOUL.md content, reloading it from disk when needed.
    pub async fn latest_soul_content(&self) -> Result<String, SoulError> {
        let mut soul = self.soul.write().await;
        soul.reload_if_changed()?;
        Ok(soul.content().to_string())
    }

    /// Return the session identifier.
    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }
}

fn events_to_messages(events: &[EventEnvelope]) -> Vec<ChatMessage> {
    events
        .iter()
        .filter_map(|event| match event.kind {
            EventKind::UserInput => event
                .payload
                .get("text")
                .and_then(serde_json::Value::as_str)
                .map(ChatMessage::user),
            EventKind::ModelResponse => event
                .payload
                .get("text")
                .and_then(serde_json::Value::as_str)
                .map(ChatMessage::assistant),
            EventKind::AssistantToolCalls => assistant_tool_calls_message(event),
            EventKind::ToolCompleted => tool_event_message(event, false),
            EventKind::ToolFailed | EventKind::ToolDenied | EventKind::ToolTimedOut => {
                tool_event_message(event, true)
            }
            _ => None,
        })
        .collect()
}

fn assistant_tool_calls_message(event: &EventEnvelope) -> Option<ChatMessage> {
    let payload =
        serde_json::from_value::<AssistantToolCallsPayload>(event.payload.clone()).ok()?;
    let tool_calls = payload
        .tool_calls
        .into_iter()
        .map(|call| ToolCall {
            id: call.call_id,
            name: call.name,
            arguments: serde_json::to_string(&call.arguments)
                .expect("tool arguments should serialize"),
        })
        .collect::<Vec<_>>();

    Some(
        ChatMessage::assistant(payload.text.unwrap_or_default()).with_metadata(
            "tool_calls",
            serde_json::to_string(&tool_calls).expect("tool calls should serialize"),
        ),
    )
}

fn tool_event_message(event: &EventEnvelope, is_error: bool) -> Option<ChatMessage> {
    let payload = serde_json::from_value::<ToolResultPayload>(event.payload.clone()).ok();
    if let Some(payload) = payload {
        let status = if payload.is_error { "error" } else { "ok" };
        return Some(
            ChatMessage::user(format!(
                "[Tool results]\n<tool_result name=\"{}\" status=\"{}\">{}</tool_result>",
                payload.name, status, payload.content
            ))
            .with_metadata("tool_result_for", payload.call_id)
            .with_metadata("tool_result_content", payload.content)
            .with_metadata("is_error", payload.is_error.to_string()),
        );
    }

    let name = event
        .payload
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("tool");
    let content = event
        .payload
        .get("result")
        .or_else(|| event.payload.get("error"))
        .or_else(|| event.payload.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let status = if is_error { "error" } else { "result" };

    Some(ChatMessage::user(format!(
        "[Tool {status} from {name}]: {content}"
    )))
}

fn estimated_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| message.content.chars().count().div_ceil(4))
        .sum()
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::{Value, json};
    use tokio::sync::RwLock;

    use super::{
        ContextBuildResult, ContextManager, InMemoryTapeStore, SoulManager, TapeStore, TokenBudget,
        estimated_tokens, events_to_messages,
    };
    use crate::message::ChatMessage;
    use crate::protocol::{
        AssistantToolCallsPayload, EventEnvelope, EventKind, SessionId, ToolCallRecord,
        ToolResultPayload,
    };

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = env::temp_dir().join(format!(
            "gemenr-context-{prefix}-{}-{timestamp}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    fn event(session_id: &SessionId, kind: EventKind, payload: Value) -> EventEnvelope {
        EventEnvelope::new(session_id.clone(), None, kind, payload)
    }

    fn manager(
        session_id: SessionId,
        tape_store: Arc<dyn TapeStore>,
    ) -> (ContextManager, std::path::PathBuf) {
        let workspace = temp_dir("workspace");
        let soul = SoulManager::load(&workspace).expect("SOUL.md should load");
        (
            ContextManager::new(session_id, tape_store, Arc::new(RwLock::new(soul))),
            workspace,
        )
    }

    #[tokio::test]
    async fn build_context_returns_ready_for_small_event_sets() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut manager, workspace) = manager(session_id.clone(), tape_store);

        manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": "hello"}),
            ))
            .await
            .expect("append should succeed");
        manager
            .append(event(
                &session_id,
                EventKind::ModelResponse,
                json!({"text": "hi there"}),
            ))
            .await
            .expect("append should succeed");

        let result = manager.build_context(&TokenBudget::default());

        assert_eq!(
            result,
            ContextBuildResult::Ready(vec![
                ChatMessage::user("hello"),
                ChatMessage::assistant("hi there"),
            ])
        );
        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn build_context_requests_summary_when_budget_threshold_is_exceeded() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut manager, workspace) = manager(session_id.clone(), tape_store);
        let long_text = "a".repeat(64);

        manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": long_text}),
            ))
            .await
            .expect("append should succeed");

        let result = manager.build_context(&TokenBudget {
            max_tokens: 16,
            threshold: 0.5,
        });

        match result {
            ContextBuildResult::NeedsSummary { messages } => {
                assert_eq!(messages, vec![ChatMessage::user("a".repeat(64))]);
            }
            other => panic!("expected NeedsSummary, got {other:?}"),
        }

        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn append_adds_events_to_the_built_context() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut manager, workspace) = manager(session_id.clone(), tape_store);

        manager
            .append(event(
                &session_id,
                EventKind::ToolCompleted,
                json!({"name": "shell", "result": "ok"}),
            ))
            .await
            .expect("append should succeed");

        assert_eq!(
            manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![ChatMessage::user("[Tool result from shell]: ok")])
        );

        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn build_context_includes_failed_tool_results() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut manager, workspace) = manager(session_id.clone(), tape_store);

        manager
            .append(event(
                &session_id,
                EventKind::ToolFailed,
                json!({"name": "shell", "result": "Denied: policy blocked"}),
            ))
            .await
            .expect("append should succeed");

        assert_eq!(
            manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![ChatMessage::user(
                "[Tool error from shell]: Denied: policy blocked",
            )])
        );

        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn create_anchor_clears_buffered_events() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut manager, workspace) = manager(session_id.clone(), Arc::clone(&tape_store));

        manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": "before anchor"}),
            ))
            .await
            .expect("append should succeed");
        let anchor_id = manager
            .create_anchor("summary".to_string())
            .await
            .expect("anchor creation should succeed");
        manager
            .append(event(
                &session_id,
                EventKind::ModelResponse,
                json!({"text": "after anchor"}),
            ))
            .await
            .expect("append should succeed");

        assert!(!anchor_id.0.is_empty());
        assert_eq!(
            manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![
                ChatMessage::system("Summary of earlier context:\nsummary"),
                ChatMessage::assistant("after anchor"),
            ])
        );

        let stored_anchor = tape_store
            .load_last_anchor(&session_id)
            .await
            .expect("anchor should load")
            .expect("anchor should exist");
        assert_eq!(stored_anchor.summary, "summary");

        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn apply_summary_creates_anchor_and_clears_buffer() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut manager, workspace) = manager(session_id.clone(), Arc::clone(&tape_store));

        manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": "before summary"}),
            ))
            .await
            .expect("append should succeed");

        let anchor_id = manager
            .apply_summary("summarized context".to_string())
            .await
            .expect("summary application should succeed");

        assert!(!anchor_id.0.is_empty());
        assert_eq!(
            manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![ChatMessage::system(
                "Summary of earlier context:\nsummarized context",
            )])
        );

        let stored_anchor = tape_store
            .load_last_anchor(&session_id)
            .await
            .expect("anchor should load")
            .expect("anchor should exist");
        assert_eq!(stored_anchor.summary, "summarized context");

        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn restore_from_tape_recovers_latest_anchor_and_events() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut first_manager, first_workspace) =
            manager(session_id.clone(), Arc::clone(&tape_store));

        first_manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": "before anchor"}),
            ))
            .await
            .expect("append should succeed");
        first_manager
            .create_anchor("phase summary".to_string())
            .await
            .expect("anchor creation should succeed");
        first_manager
            .append(event(
                &session_id,
                EventKind::ModelResponse,
                json!({"text": "after anchor"}),
            ))
            .await
            .expect("append should succeed");

        let (mut restored_manager, restored_workspace) = manager(session_id.clone(), tape_store);
        restored_manager
            .restore_from_tape()
            .await
            .expect("restore should succeed");

        assert_eq!(
            restored_manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![
                ChatMessage::system("Summary of earlier context:\nphase summary"),
                ChatMessage::assistant("after anchor"),
            ])
        );

        fs::remove_dir_all(first_workspace).expect("temp directory should be removed");
        fs::remove_dir_all(restored_workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn restore_from_tape_keeps_failed_tool_context_after_anchor() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut first_manager, first_workspace) =
            manager(session_id.clone(), Arc::clone(&tape_store));

        first_manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": "before anchor"}),
            ))
            .await
            .expect("append should succeed");
        first_manager
            .create_anchor("phase summary".to_string())
            .await
            .expect("anchor creation should succeed");
        first_manager
            .append(event(
                &session_id,
                EventKind::ToolFailed,
                json!({"name": "shell", "result": "command exited with status 1"}),
            ))
            .await
            .expect("append should succeed");

        let (mut restored_manager, restored_workspace) = manager(session_id.clone(), tape_store);
        restored_manager
            .restore_from_tape()
            .await
            .expect("restore should succeed");

        assert_eq!(
            restored_manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![
                ChatMessage::system("Summary of earlier context:\nphase summary"),
                ChatMessage::user("[Tool error from shell]: command exited with status 1"),
            ])
        );

        fs::remove_dir_all(first_workspace).expect("temp directory should be removed");
        fs::remove_dir_all(restored_workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn restore_from_tape_keeps_summary_context_after_apply_summary() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut first_manager, first_workspace) =
            manager(session_id.clone(), Arc::clone(&tape_store));

        first_manager
            .append(event(
                &session_id,
                EventKind::UserInput,
                json!({"text": "before summary"}),
            ))
            .await
            .expect("append should succeed");
        first_manager
            .apply_summary("summarized context".to_string())
            .await
            .expect("summary application should succeed");
        first_manager
            .append(event(
                &session_id,
                EventKind::ContextSummarized,
                json!({"summary": "summarized context"}),
            ))
            .await
            .expect("append should succeed");
        first_manager
            .append(event(
                &session_id,
                EventKind::ToolFailed,
                json!({"name": "shell", "result": "Tool execution cancelled"}),
            ))
            .await
            .expect("append should succeed");

        let (mut restored_manager, restored_workspace) = manager(session_id.clone(), tape_store);
        restored_manager
            .restore_from_tape()
            .await
            .expect("restore should succeed");

        assert_eq!(
            restored_manager.build_context(&TokenBudget::default()),
            ContextBuildResult::Ready(vec![
                ChatMessage::system("Summary of earlier context:\nsummarized context"),
                ChatMessage::user("[Tool error from shell]: Tool execution cancelled"),
            ])
        );

        fs::remove_dir_all(first_workspace).expect("temp directory should be removed");
        fs::remove_dir_all(restored_workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn soul_content_returns_current_soul_markdown() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (manager, workspace) = manager(session_id, tape_store);

        let content = manager.soul_content().await;

        assert!(content.contains("# Identity"));
        assert!(content.contains("# Preferences"));

        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn latest_soul_content_observes_external_edit() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (manager, workspace) = manager(session_id, tape_store);
        let soul_path = workspace.join("SOUL.md");
        let updated_content = "# Identity\nexternal update\n\n# Preferences\nprefs\n\n# Experiences\nexp\n\n# Notes\nnotes\n";

        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&soul_path, updated_content).expect("SOUL.md should be rewritten");

        let content = manager
            .latest_soul_content()
            .await
            .expect("latest soul content should reload");

        assert_eq!(content, updated_content);
        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn restore_from_tape_rebuilds_native_tool_metadata() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut first_manager, first_workspace) =
            manager(session_id.clone(), Arc::clone(&tape_store));

        first_manager
            .append(event(
                &session_id,
                EventKind::AssistantToolCalls,
                serde_json::to_value(AssistantToolCallsPayload {
                    text: Some("working".to_string()),
                    tool_calls: vec![ToolCallRecord {
                        call_id: "call-1".to_string(),
                        name: "shell".to_string(),
                        arguments: json!({"command": "pwd"}),
                    }],
                })
                .expect("assistant tool calls payload should serialize"),
            ))
            .await
            .expect("append should succeed");
        first_manager
            .append(event(
                &session_id,
                EventKind::ToolCompleted,
                serde_json::to_value(ToolResultPayload {
                    call_id: "call-1".to_string(),
                    name: "shell".to_string(),
                    content: "pwd output".to_string(),
                    is_error: false,
                })
                .expect("tool result payload should serialize"),
            ))
            .await
            .expect("append should succeed");

        let (mut restored_manager, restored_workspace) = manager(session_id, tape_store);
        restored_manager
            .restore_from_tape()
            .await
            .expect("restore should succeed");

        let ContextBuildResult::Ready(messages) =
            restored_manager.build_context(&TokenBudget::default())
        else {
            panic!("expected ready context after restore");
        };

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, crate::message::ChatRole::Assistant);
        assert_eq!(messages[0].content, "working");
        let tool_calls = serde_json::from_str::<Vec<crate::model::ToolCall>>(
            messages[0]
                .metadata
                .get("tool_calls")
                .expect("tool metadata should exist"),
        )
        .expect("tool metadata should deserialize");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call-1");
        assert_eq!(tool_calls[0].name, "shell");

        assert_eq!(messages[1].role, crate::message::ChatRole::User);
        assert_eq!(
            messages[1]
                .metadata
                .get("tool_result_for")
                .map(String::as_str),
            Some("call-1")
        );
        assert_eq!(
            messages[1].metadata.get("is_error").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            messages[1]
                .metadata
                .get("tool_result_content")
                .map(String::as_str),
            Some("pwd output")
        );

        fs::remove_dir_all(first_workspace).expect("temp directory should be removed");
        fs::remove_dir_all(restored_workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn restore_from_tape_keeps_xml_tool_result_projection() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut first_manager, first_workspace) =
            manager(session_id.clone(), Arc::clone(&tape_store));

        first_manager
            .append(event(
                &session_id,
                EventKind::ToolCompleted,
                serde_json::to_value(ToolResultPayload {
                    call_id: "call-9".to_string(),
                    name: "shell".to_string(),
                    content: "done".to_string(),
                    is_error: false,
                })
                .expect("tool result payload should serialize"),
            ))
            .await
            .expect("append should succeed");

        let (mut restored_manager, restored_workspace) = manager(session_id, tape_store);
        restored_manager
            .restore_from_tape()
            .await
            .expect("restore should succeed");

        let ContextBuildResult::Ready(messages) =
            restored_manager.build_context(&TokenBudget::default())
        else {
            panic!("expected ready context after restore");
        };

        assert_eq!(messages.len(), 1);
        assert!(
            messages[0]
                .content
                .contains(r#"<tool_result name="shell" status="ok">done</tool_result>"#)
        );

        fs::remove_dir_all(first_workspace).expect("temp directory should be removed");
        fs::remove_dir_all(restored_workspace).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn restore_from_tape_still_respects_last_anchor_boundary() {
        let session_id = SessionId::new();
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let (mut first_manager, first_workspace) =
            manager(session_id.clone(), Arc::clone(&tape_store));

        first_manager
            .append(event(
                &session_id,
                EventKind::AssistantToolCalls,
                serde_json::to_value(AssistantToolCallsPayload {
                    text: Some("before anchor".to_string()),
                    tool_calls: vec![ToolCallRecord {
                        call_id: "call-before".to_string(),
                        name: "shell".to_string(),
                        arguments: json!({"command": "ls"}),
                    }],
                })
                .expect("assistant tool calls payload should serialize"),
            ))
            .await
            .expect("append should succeed");
        first_manager
            .create_anchor("phase summary".to_string())
            .await
            .expect("anchor creation should succeed");
        first_manager
            .append(event(
                &session_id,
                EventKind::ToolCompleted,
                serde_json::to_value(ToolResultPayload {
                    call_id: "call-after".to_string(),
                    name: "shell".to_string(),
                    content: "post-anchor".to_string(),
                    is_error: false,
                })
                .expect("tool result payload should serialize"),
            ))
            .await
            .expect("append should succeed");

        let (mut restored_manager, restored_workspace) = manager(session_id, tape_store);
        restored_manager
            .restore_from_tape()
            .await
            .expect("restore should succeed");

        let ContextBuildResult::Ready(messages) =
            restored_manager.build_context(&TokenBudget::default())
        else {
            panic!("expected ready context after restore");
        };

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0],
            ChatMessage::system(
                "Summary of earlier context:
phase summary"
            )
        );
        assert!(
            messages[1]
                .metadata
                .get("tool_result_for")
                .is_some_and(|value| value == "call-after")
        );
        assert!(
            messages
                .iter()
                .all(|message| !message.content.contains("call-before"))
        );

        fs::remove_dir_all(first_workspace).expect("temp directory should be removed");
        fs::remove_dir_all(restored_workspace).expect("temp directory should be removed");
    }

    #[test]
    fn events_to_messages_filters_supported_event_kinds() {
        let session_id = SessionId::new();
        let messages = events_to_messages(&[
            event(&session_id, EventKind::UserInput, json!({"text": "hello"})),
            event(
                &session_id,
                EventKind::ModelResponse,
                json!({"text": "world"}),
            ),
            event(
                &session_id,
                EventKind::ToolCompleted,
                json!({"name": "shell", "result": "ok"}),
            ),
            event(
                &session_id,
                EventKind::ToolFailed,
                json!({"name": "shell", "error": "boom"}),
            ),
        ]);

        assert_eq!(
            messages,
            vec![
                ChatMessage::user("hello"),
                ChatMessage::assistant("world"),
                ChatMessage::user("[Tool result from shell]: ok"),
                ChatMessage::user("[Tool error from shell]: boom"),
            ]
        );
    }

    #[test]
    fn token_estimation_distinguishes_short_and_long_messages() {
        let short_messages = vec![ChatMessage::user("short")];
        let long_messages = vec![ChatMessage::assistant("x".repeat(128))];

        assert!(estimated_tokens(&short_messages) <= 16);
        assert!(estimated_tokens(&long_messages) > 16);
    }
}
