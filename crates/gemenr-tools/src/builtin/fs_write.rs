//! Built-in file write tool.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use gemenr_core::{RiskLevel, ToolSpec};

use crate::handler::{ExecContext, ToolError, ToolHandler, ToolOutput};

/// Tool handler for writing text files to disk.
pub struct FsWriteHandler;

#[async_trait]
impl ToolHandler for FsWriteHandler {
    async fn execute(
        &self,
        ctx: &ExecContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let path = args
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'path'".to_string(),
            })?;
        let content = args
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'content'".to_string(),
            })?;
        let resolved_path = resolve_path(ctx, path);

        if let Some(parent) = resolved_path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| ToolError::Execution {
                    exit_code: None,
                    stderr: error.to_string(),
                })?;
        }

        tokio::fs::write(&resolved_path, content)
            .await
            .map_err(|error| ToolError::Execution {
                exit_code: None,
                stderr: error.to_string(),
            })?;

        Ok(ToolOutput {
            content: format!("wrote {} bytes to {}", content.len(), path),
        })
    }
}

fn resolve_path(ctx: &ExecContext, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        ctx.working_dir.join(path)
    }
}

/// Create the tool specification for the built-in file write tool.
#[must_use]
pub fn fs_write_spec() -> ToolSpec {
    ToolSpec {
        name: "fs.write".to_string(),
        description:
            "Write content to a file at the given path. Creates the file if it doesn't exist."
                .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }),
        risk_level: RiskLevel::Medium,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::FsWriteHandler;
    use crate::{ExecContext, ToolHandler};

    fn temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("gemenr-fs-write-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[tokio::test]
    async fn writes_new_file() {
        let directory = temp_dir("new-file");
        let file_path = directory.join("sample.txt");
        let handler = FsWriteHandler;
        let context = ExecContext::default();

        handler
            .execute(&context, json!({"path": file_path, "content": "hello"}))
            .await
            .expect("write should succeed");

        assert_eq!(
            fs::read_to_string(&file_path).expect("file should exist"),
            "hello"
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn overwrites_existing_file() {
        let directory = temp_dir("overwrite");
        let file_path = directory.join("sample.txt");
        fs::write(&file_path, "old").expect("initial file should be written");
        let handler = FsWriteHandler;
        let context = ExecContext::default();

        handler
            .execute(
                &context,
                json!({"path": file_path, "content": "new content"}),
            )
            .await
            .expect("overwrite should succeed");

        assert_eq!(
            fs::read_to_string(&file_path).expect("file should exist"),
            "new content"
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn creates_parent_directories_automatically() {
        let directory = temp_dir("nested");
        let file_path = directory.join("nested/path/sample.txt");
        let handler = FsWriteHandler;
        let context = ExecContext::default();

        handler
            .execute(
                &context,
                json!({"path": file_path, "content": "hello nested"}),
            )
            .await
            .expect("nested write should succeed");

        assert_eq!(
            fs::read_to_string(&file_path).expect("nested file should exist"),
            "hello nested"
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn resolves_relative_paths_from_context_working_directory() {
        let directory = temp_dir("relative");
        let file_path = directory.join("nested/path/sample.txt");
        let handler = FsWriteHandler;
        let context = ExecContext {
            working_dir: directory.clone(),
            timeout: Duration::from_secs(5),
            ..ExecContext::default()
        };

        handler
            .execute(
                &context,
                json!({"path": "nested/path/sample.txt", "content": "hello relative"}),
            )
            .await
            .expect("relative write should succeed");

        assert_eq!(
            fs::read_to_string(&file_path).expect("nested file should exist"),
            "hello relative"
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
