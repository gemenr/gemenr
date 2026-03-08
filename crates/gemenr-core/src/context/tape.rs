use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::protocol::{EventEnvelope, EventKind, SessionId};

/// An anchor entry in the tape — marks a stage boundary with a summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorEntry {
    /// Unique identifier for this anchor.
    pub anchor_id: String,
    /// Summary text for this anchor point.
    pub summary: String,
    /// The event envelope that created this anchor.
    pub event: EventEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AnchorPayload {
    anchor_id: String,
    summary: String,
}

/// Pluggable storage backend for the event tape.
///
/// Each session has its own tape. Events are appended sequentially
/// and can be loaded for context reconstruction.
#[async_trait]
pub trait TapeStore: Send + Sync {
    /// Append an event to the session's tape.
    async fn append(&self, session_id: &SessionId, event: EventEnvelope) -> Result<(), TapeError>;

    /// Load all events since the last anchor, or all events if no anchor exists.
    async fn load_since_anchor(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<EventEnvelope>, TapeError>;

    /// Load the most recent anchor entry for a session.
    async fn load_last_anchor(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<AnchorEntry>, TapeError>;

    /// Load all events for a session.
    async fn load_all(&self, session_id: &SessionId) -> Result<Vec<EventEnvelope>, TapeError>;
}

/// Errors from tape storage operations.
#[derive(Debug, thiserror::Error)]
pub enum TapeError {
    /// I/O error during tape read/write.
    #[error("tape I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to serialize or deserialize event data.
    #[error("tape serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// JSONL file-based tape store — one `.jsonl` file per session.
///
/// Events are serialized as JSON and appended one per line.
/// Anchors are identified by [`EventKind::AnchorCreated`] in the event stream.
#[derive(Debug, Clone)]
pub struct JsonlTapeStore {
    /// Base directory for tape files.
    base_dir: PathBuf,
}

impl JsonlTapeStore {
    /// Create a new JSONL tape store with the given base directory.
    ///
    /// The directory is created if it does not already exist.
    pub fn new(base_dir: PathBuf) -> Result<Self, TapeError> {
        fs::create_dir_all(&base_dir)?;

        Ok(Self { base_dir })
    }

    /// Get the file path for a session's tape.
    #[must_use]
    fn session_path(&self, session_id: &SessionId) -> PathBuf {
        self.base_dir.join(format!("{}.jsonl", session_id.0))
    }
}

#[async_trait]
impl TapeStore for JsonlTapeStore {
    async fn append(&self, session_id: &SessionId, event: EventEnvelope) -> Result<(), TapeError> {
        let path = self.session_path(session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        let line = serde_json::to_string(&event)?;

        writeln!(file, "{line}")?;
        debug!(session_id = %session_id.0, path = %path.display(), "appended event to tape");
        Ok(())
    }

    async fn load_since_anchor(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<EventEnvelope>, TapeError> {
        let events = self.load_all(session_id).await?;
        let anchor_index = events
            .iter()
            .rposition(|event| matches!(event.kind, EventKind::AnchorCreated));

        Ok(match anchor_index {
            Some(index) => events.into_iter().skip(index + 1).collect(),
            None => events,
        })
    }

    async fn load_last_anchor(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<AnchorEntry>, TapeError> {
        let events = self.load_all(session_id).await?;

        for event in events.iter().rev() {
            if let Some(anchor) = anchor_from_event(event)? {
                return Ok(Some(anchor));
            }
        }

        Ok(None)
    }

    async fn load_all(&self, session_id: &SessionId) -> Result<Vec<EventEnvelope>, TapeError> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            events.push(serde_json::from_str::<EventEnvelope>(&line)?);
        }

        debug!(session_id = %session_id.0, path = %path.display(), count = events.len(), "loaded events from tape");
        Ok(events)
    }
}

/// In-memory tape store for testing.
///
/// Stores events in a `HashMap<SessionId, Vec<EventEnvelope>>` behind a mutex.
#[derive(Debug, Default)]
pub struct InMemoryTapeStore {
    events: Mutex<HashMap<String, Vec<EventEnvelope>>>,
}

impl InMemoryTapeStore {
    /// Create a new empty in-memory tape store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TapeStore for InMemoryTapeStore {
    async fn append(&self, session_id: &SessionId, event: EventEnvelope) -> Result<(), TapeError> {
        let mut events = self.events.lock().map_err(|error| {
            std::io::Error::other(format!("in-memory tape store lock poisoned: {error}"))
        })?;

        events.entry(session_id.0.clone()).or_default().push(event);
        debug!(session_id = %session_id.0, "appended event to in-memory tape");
        Ok(())
    }

    async fn load_since_anchor(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<EventEnvelope>, TapeError> {
        let events = self.load_all(session_id).await?;
        let anchor_index = events
            .iter()
            .rposition(|event| matches!(event.kind, EventKind::AnchorCreated));

        Ok(match anchor_index {
            Some(index) => events.into_iter().skip(index + 1).collect(),
            None => events,
        })
    }

    async fn load_last_anchor(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<AnchorEntry>, TapeError> {
        let events = self.load_all(session_id).await?;

        for event in events.iter().rev() {
            if let Some(anchor) = anchor_from_event(event)? {
                return Ok(Some(anchor));
            }
        }

        Ok(None)
    }

    async fn load_all(&self, session_id: &SessionId) -> Result<Vec<EventEnvelope>, TapeError> {
        let events = self.events.lock().map_err(|error| {
            std::io::Error::other(format!("in-memory tape store lock poisoned: {error}"))
        })?;

        Ok(events.get(&session_id.0).cloned().unwrap_or_default())
    }
}

fn anchor_from_event(event: &EventEnvelope) -> Result<Option<AnchorEntry>, TapeError> {
    if !matches!(event.kind, EventKind::AnchorCreated) {
        return Ok(None);
    }

    let payload: AnchorPayload = serde_json::from_value(event.payload.clone())?;
    Ok(Some(AnchorEntry {
        anchor_id: payload.anchor_id,
        summary: payload.summary,
        event: event.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::{env, fs};

    use serde_json::{Value, json};

    use super::{InMemoryTapeStore, JsonlTapeStore, TapeStore};
    use crate::protocol::{EventEnvelope, EventKind, SessionId};

    fn event(session_id: &SessionId, kind: EventKind, payload: Value) -> EventEnvelope {
        EventEnvelope::new(session_id.clone(), None, kind, payload)
    }

    fn anchor_event(session_id: &SessionId, anchor_id: &str, summary: &str) -> EventEnvelope {
        event(
            session_id,
            EventKind::AnchorCreated,
            json!({
                "anchor_id": anchor_id,
                "summary": summary,
            }),
        )
    }

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = env::temp_dir().join(format!("gemenr-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[tokio::test]
    async fn in_memory_tape_store_appends_and_loads_events_in_order() {
        let store = InMemoryTapeStore::new();
        let session_id = SessionId::new();
        let first = event(&session_id, EventKind::UserInput, json!({"text": "hello"}));
        let second = event(
            &session_id,
            EventKind::ModelResponse,
            json!({"text": "world"}),
        );

        store
            .append(&session_id, first.clone())
            .await
            .expect("append should succeed");
        store
            .append(&session_id, second.clone())
            .await
            .expect("append should succeed");

        let events = store
            .load_all(&session_id)
            .await
            .expect("load should succeed");

        assert_eq!(events, vec![first, second]);
    }

    #[tokio::test]
    async fn in_memory_tape_store_returns_empty_for_unknown_session() {
        let store = InMemoryTapeStore::new();
        let events = store
            .load_all(&SessionId::new())
            .await
            .expect("load should succeed");

        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn in_memory_tape_store_finds_latest_anchor() {
        let store = InMemoryTapeStore::new();
        let session_id = SessionId::new();

        store
            .append(
                &session_id,
                event(&session_id, EventKind::UserInput, json!({"text": "before"})),
            )
            .await
            .expect("append should succeed");
        store
            .append(&session_id, anchor_event(&session_id, "anchor-1", "first"))
            .await
            .expect("append should succeed");
        store
            .append(&session_id, anchor_event(&session_id, "anchor-2", "second"))
            .await
            .expect("append should succeed");

        let anchor = store
            .load_last_anchor(&session_id)
            .await
            .expect("anchor load should succeed")
            .expect("anchor should exist");

        assert_eq!(anchor.anchor_id, "anchor-2");
        assert_eq!(anchor.summary, "second");
        assert!(matches!(anchor.event.kind, EventKind::AnchorCreated));
    }

    #[tokio::test]
    async fn in_memory_tape_store_loads_only_events_after_last_anchor() {
        let store = InMemoryTapeStore::new();
        let session_id = SessionId::new();
        let after_anchor = event(
            &session_id,
            EventKind::ToolCompleted,
            json!({"name": "shell", "result": "ok"}),
        );

        store
            .append(
                &session_id,
                event(&session_id, EventKind::UserInput, json!({"text": "before"})),
            )
            .await
            .expect("append should succeed");
        store
            .append(&session_id, anchor_event(&session_id, "anchor-1", "cut"))
            .await
            .expect("append should succeed");
        store
            .append(&session_id, after_anchor.clone())
            .await
            .expect("append should succeed");

        let events = store
            .load_since_anchor(&session_id)
            .await
            .expect("load since anchor should succeed");

        assert_eq!(events, vec![after_anchor]);
    }

    #[tokio::test]
    async fn in_memory_tape_store_returns_all_events_when_no_anchor_exists() {
        let store = InMemoryTapeStore::new();
        let session_id = SessionId::new();
        let event = event(&session_id, EventKind::UserInput, json!({"text": "hello"}));

        store
            .append(&session_id, event.clone())
            .await
            .expect("append should succeed");

        let events = store
            .load_since_anchor(&session_id)
            .await
            .expect("load since anchor should succeed");

        assert_eq!(events, vec![event]);
    }

    #[tokio::test]
    async fn jsonl_tape_store_appends_and_loads_events() {
        let directory = temp_dir("jsonl-append-load");
        let store = JsonlTapeStore::new(directory.clone()).expect("store should be created");
        let session_id = SessionId::new();
        let first = event(&session_id, EventKind::UserInput, json!({"text": "hello"}));
        let second = event(
            &session_id,
            EventKind::ModelResponse,
            json!({"text": "world"}),
        );

        store
            .append(&session_id, first.clone())
            .await
            .expect("append should succeed");
        store
            .append(&session_id, second.clone())
            .await
            .expect("append should succeed");

        let events = store
            .load_all(&session_id)
            .await
            .expect("load should succeed");

        assert_eq!(events, vec![first, second]);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn jsonl_tape_store_persists_one_json_object_per_line() {
        let directory = temp_dir("jsonl-lines");
        let store = JsonlTapeStore::new(directory.clone()).expect("store should be created");
        let session_id = SessionId::new();

        store
            .append(
                &session_id,
                event(&session_id, EventKind::UserInput, json!({"text": "first"})),
            )
            .await
            .expect("append should succeed");
        store
            .append(
                &session_id,
                event(
                    &session_id,
                    EventKind::ModelResponse,
                    json!({"text": "second"}),
                ),
            )
            .await
            .expect("append should succeed");

        let path = directory.join(format!("{}.jsonl", session_id.0));
        let content = fs::read_to_string(&path).expect("jsonl file should be readable");
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines.len(), 2);
        for line in lines {
            let value: Value = serde_json::from_str(line).expect("line should be valid json");
            assert!(value.is_object());
        }

        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn jsonl_tape_store_returns_empty_for_missing_file() {
        let directory = temp_dir("jsonl-missing");
        let store = JsonlTapeStore::new(directory.clone()).expect("store should be created");
        let events = store
            .load_all(&SessionId::new())
            .await
            .expect("load should succeed");

        assert!(events.is_empty());
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn jsonl_tape_store_finds_latest_anchor() {
        let directory = temp_dir("jsonl-anchor");
        let store = JsonlTapeStore::new(directory.clone()).expect("store should be created");
        let session_id = SessionId::new();

        store
            .append(&session_id, anchor_event(&session_id, "anchor-1", "first"))
            .await
            .expect("append should succeed");
        store
            .append(&session_id, anchor_event(&session_id, "anchor-2", "second"))
            .await
            .expect("append should succeed");

        let anchor = store
            .load_last_anchor(&session_id)
            .await
            .expect("anchor load should succeed")
            .expect("anchor should exist");

        assert_eq!(anchor.anchor_id, "anchor-2");
        assert_eq!(anchor.summary, "second");
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
