//! Built-in shell command execution tool.

use async_trait::async_trait;
use gemenr_core::{RiskLevel, ToolSpec};
use tokio::process::Command;

use crate::handler::{ExecContext, ToolError, ToolHandler, ToolOutput};

const SHELL_OUTPUT_LIMIT: usize = 10_000;

/// Tool handler for shell command execution.
pub struct ShellHandler;

#[async_trait]
impl ToolHandler for ShellHandler {
    async fn execute(
        &self,
        ctx: &ExecContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let command = args
            .get("command")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::Input {
                message: "missing required field 'command'".to_string(),
            })?;

        let output = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .arg("/C")
                .arg(command)
                .current_dir(&ctx.working_dir)
                .output()
                .await
        } else {
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&ctx.working_dir)
                .output()
                .await
        }
        .map_err(|error| ToolError::Execution {
            exit_code: None,
            stderr: error.to_string(),
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let content = truncate_output(
            &format!("stdout:\n{}\n\nstderr:\n{}", stdout, stderr),
            SHELL_OUTPUT_LIMIT,
        );

        if output.status.success() {
            Ok(ToolOutput { content })
        } else {
            Err(ToolError::Execution {
                exit_code: output.status.code(),
                stderr: content,
            })
        }
    }
}

/// Create the tool specification for the built-in shell tool.
#[must_use]
pub fn shell_spec() -> ToolSpec {
    ToolSpec {
        name: "shell".to_string(),
        description: "Execute a shell command and return its output.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        }),
        risk_level: RiskLevel::Medium,
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
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::ShellHandler;
    use crate::{ExecContext, ToolError, ToolHandler};

    fn temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("gemenr-shell-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[tokio::test]
    async fn executes_simple_command() {
        let handler = ShellHandler;
        let context = ExecContext::default();

        let output = handler
            .execute(&context, json!({"command": "echo hello"}))
            .await
            .expect("command should succeed");

        assert!(output.content.contains("stdout:"));
        assert!(output.content.contains("hello"));
        assert!(output.content.contains("stderr:"));
    }

    #[tokio::test]
    async fn returns_execution_error_for_failing_command() {
        let handler = ShellHandler;
        let context = ExecContext::default();

        let error = handler
            .execute(
                &context,
                json!({"command": "command-that-should-not-exist-12345"}),
            )
            .await
            .expect_err("command should fail");

        assert!(matches!(error, ToolError::Execution { .. }));
    }

    #[tokio::test]
    async fn returns_input_error_when_command_is_missing() {
        let handler = ShellHandler;
        let context = ExecContext::default();

        let error = handler
            .execute(&context, json!({}))
            .await
            .expect_err("missing command should fail");

        assert_eq!(
            error,
            ToolError::Input {
                message: "missing required field 'command'".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn executes_command_in_context_working_directory() {
        let directory = temp_dir("working-dir");
        let handler = ShellHandler;
        let context = ExecContext {
            working_dir: directory.clone(),
            timeout: Duration::from_secs(5),
        };
        let command = if cfg!(target_os = "windows") {
            "cd"
        } else {
            "pwd"
        };

        let output = handler
            .execute(&context, json!({"command": command}))
            .await
            .expect("command should succeed");

        assert!(
            output
                .content
                .contains(directory.to_string_lossy().as_ref())
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
