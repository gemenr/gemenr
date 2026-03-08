//! Built-in file read tool.

use async_trait::async_trait;
use gemenr_core::{RiskLevel, ToolSpec};

use crate::handler::{ToolError, ToolHandler, ToolOutput};

const FILE_READ_LIMIT: usize = 50_000;

/// Tool handler for reading text files from disk.
pub struct FsReadHandler;

#[async_trait]
impl ToolHandler for FsReadHandler {
    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let path = args
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'path'".to_string(),
            })?;

        let content =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|error| ToolError::Execution {
                    exit_code: None,
                    stderr: error.to_string(),
                })?;

        Ok(ToolOutput {
            content: truncate_output(&content, FILE_READ_LIMIT),
        })
    }
}

/// Create the tool specification for the built-in file read tool.
#[must_use]
pub fn fs_read_spec() -> ToolSpec {
    ToolSpec {
        name: "fs.read".to_string(),
        description: "Read the content of a file at the given path.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }),
        risk_level: RiskLevel::Low,
    }
}

fn truncate_output(content: &str, limit: usize) -> String {
    if content.chars().count() <= limit {
        return content.to_string();
    }

    let truncated = content.chars().take(limit).collect::<String>();
    format!("{truncated}\n...[truncated]")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::FsReadHandler;
    use crate::{ToolError, ToolHandler};

    fn temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("gemenr-fs-read-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[tokio::test]
    async fn reads_existing_file() {
        let directory = temp_dir("existing");
        let file_path = directory.join("sample.txt");
        fs::write(&file_path, "hello from file").expect("sample file should be written");

        let handler = FsReadHandler;
        let output = handler
            .execute(json!({"path": file_path}))
            .await
            .expect("read should succeed");

        assert_eq!(output.content, "hello from file");
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn returns_execution_error_for_missing_file() {
        let directory = temp_dir("missing");
        let file_path = directory.join("missing.txt");

        let handler = FsReadHandler;
        let error = handler
            .execute(json!({"path": file_path}))
            .await
            .expect_err("missing file should fail");

        assert!(matches!(error, ToolError::Execution { .. }));
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn returns_input_error_when_path_is_missing() {
        let handler = FsReadHandler;
        let error = handler
            .execute(json!({}))
            .await
            .expect_err("missing path should fail");

        assert_eq!(
            error,
            ToolError::Input {
                message: "missing required field 'path'".to_string(),
            }
        );
    }
}
