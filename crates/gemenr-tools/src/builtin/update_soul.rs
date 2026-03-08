//! Built-in SOUL.md update tool.

use std::sync::Arc;

use async_trait::async_trait;
use gemenr_core::{RiskLevel, SoulManager, ToolSpec};
use tokio::sync::RwLock;

use crate::handler::{ToolError, ToolHandler, ToolOutput};

/// Tool handler for updating persistent `SOUL.md` memory.
pub struct UpdateSoulHandler {
    soul: Arc<RwLock<SoulManager>>,
}

impl UpdateSoulHandler {
    /// Create a new SOUL update handler.
    #[must_use]
    pub fn new(soul: Arc<RwLock<SoulManager>>) -> Self {
        Self { soul }
    }
}

#[async_trait]
impl ToolHandler for UpdateSoulHandler {
    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let section = args
            .get("section")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'section'".to_string(),
            })?;
        let action = args
            .get("action")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'action'".to_string(),
            })?;
        let content = args
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'content'".to_string(),
            })?;

        let section = validate_section(section)?;
        let mut soul = self.soul.write().await;

        match action {
            "append" => soul.append(section, content),
            "replace" => soul.update(section, content),
            other => {
                return Err(ToolError::Input {
                    message: format!("invalid action '{other}'"),
                });
            }
        }
        .map_err(|error| ToolError::Execution {
            exit_code: None,
            stderr: error.to_string(),
        })?;

        Ok(ToolOutput {
            content: format!("updated SOUL.md section '{section}' with action '{action}'"),
        })
    }
}

/// Create the tool specification for the built-in SOUL update tool.
#[must_use]
pub fn update_soul_spec() -> ToolSpec {
    ToolSpec {
        name: "update_soul".to_string(),
        description: "Update a section of SOUL.md to record preferences, experiences, or notes for future reference.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["Identity", "Preferences", "Experiences", "Notes"],
                    "description": "Which section to update"
                },
                "action": {
                    "type": "string",
                    "enum": ["append", "replace"],
                    "description": "Whether to append to or replace the section content"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["section", "action", "content"]
        }),
        risk_level: RiskLevel::Low,
    }
}

fn validate_section(section: &str) -> Result<&str, ToolError> {
    match section {
        "Identity" | "Preferences" | "Experiences" | "Notes" => Ok(section),
        other => Err(ToolError::Input {
            message: format!("invalid section '{other}'"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use gemenr_core::SoulManager;
    use serde_json::json;
    use tokio::sync::RwLock;

    use super::UpdateSoulHandler;
    use crate::{ToolError, ToolHandler};

    fn temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("gemenr-update-soul-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[tokio::test]
    async fn appends_section_content() {
        let directory = temp_dir("append");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let handler = UpdateSoulHandler::new(Arc::clone(&soul));

        handler
            .execute(json!({
                "section": "Experiences",
                "action": "append",
                "content": "- Validate after each step"
            }))
            .await
            .expect("append should succeed");

        let guard = soul.read().await;
        assert!(guard.content().contains("- Validate after each step"));
        drop(guard);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn replaces_section_content() {
        let directory = temp_dir("replace");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let handler = UpdateSoulHandler::new(Arc::clone(&soul));

        handler
            .execute(json!({
                "section": "Notes",
                "action": "replace",
                "content": "Keep commits focused."
            }))
            .await
            .expect("replace should succeed");

        let guard = soul.read().await;
        assert!(guard.content().contains("# Notes\nKeep commits focused."));
        drop(guard);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn rejects_invalid_section() {
        let directory = temp_dir("invalid-section");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let handler = UpdateSoulHandler::new(soul);

        let error = handler
            .execute(json!({
                "section": "Unknown",
                "action": "append",
                "content": "test"
            }))
            .await
            .expect_err("invalid section should fail");

        assert_eq!(
            error,
            ToolError::Input {
                message: "invalid section 'Unknown'".to_string(),
            }
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn rejects_invalid_action() {
        let directory = temp_dir("invalid-action");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let handler = UpdateSoulHandler::new(soul);

        let error = handler
            .execute(json!({
                "section": "Notes",
                "action": "merge",
                "content": "test"
            }))
            .await
            .expect_err("invalid action should fail");

        assert_eq!(
            error,
            ToolError::Input {
                message: "invalid action 'merge'".to_string(),
            }
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
