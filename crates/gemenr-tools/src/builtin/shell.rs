//! Built-in shell command execution tool.

use std::sync::Arc;

use async_trait::async_trait;
use gemenr_core::{RiskLevel, ToolSpec};

use crate::SandboxKind;
use crate::handler::{ExecContext, ToolError, ToolHandler, ToolOutput};
use crate::sandbox::{self, SandboxRunner, ShellCommand};

type RunnerSelector =
    dyn Fn(SandboxKind) -> Result<Box<dyn SandboxRunner>, ToolError> + Send + Sync + 'static;

/// Tool handler for shell command execution.
pub struct ShellHandler {
    runner_selector: Arc<RunnerSelector>,
}

impl ShellHandler {
    /// Create a shell handler with the default sandbox backend selector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runner_selector: Arc::new(sandbox::runner_for),
        }
    }

    /// Create a shell handler with a custom backend selector.
    #[must_use]
    pub fn with_runner_selector(runner_selector: Arc<RunnerSelector>) -> Self {
        Self { runner_selector }
    }
}

impl Default for ShellHandler {
    fn default() -> Self {
        Self::new()
    }
}

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
        let shell_command = ShellCommand {
            command: command.to_string(),
        };

        match ctx.sandbox {
            SandboxKind::None => sandbox::run_without_sandbox(&shell_command, ctx).await,
            kind => {
                let runner = (self.runner_selector)(kind)?;
                runner.run(&shell_command, ctx).await
            }
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use async_trait::async_trait;
    use serde_json::json;

    use super::{ShellHandler, shell_spec};
    use crate::sandbox::{SandboxRunner, ShellCommand};
    use crate::{ExecContext, SandboxKind, ToolError, ToolHandler, ToolOutput};

    fn temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("gemenr-shell-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[derive(Clone)]
    struct RecordingRunner {
        content: &'static str,
    }

    #[async_trait]
    impl SandboxRunner for RecordingRunner {
        async fn run(
            &self,
            _command: &ShellCommand,
            _ctx: &ExecContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                content: self.content.to_string(),
            })
        }
    }

    #[tokio::test]
    async fn none_sandbox_keeps_existing_behavior() {
        let handler = ShellHandler::default();
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
    async fn policy_selects_requested_backend() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let selector_seen = Arc::clone(&seen);
        let handler = ShellHandler::with_runner_selector(Arc::new(move |kind| {
            selector_seen
                .lock()
                .expect("selector mutex should not be poisoned")
                .push(kind);
            Ok(Box::new(RecordingRunner {
                content: "sandboxed",
            }) as Box<dyn SandboxRunner>)
        }));
        let context = ExecContext {
            sandbox: SandboxKind::Seatbelt,
            ..ExecContext::default()
        };

        let output = handler
            .execute(&context, json!({"command": "echo hello"}))
            .await
            .expect("command should succeed");

        assert_eq!(output.content, "sandboxed");
        assert_eq!(
            seen.lock()
                .expect("selector mutex should not be poisoned")
                .as_slice(),
            &[SandboxKind::Seatbelt]
        );
    }

    #[tokio::test]
    async fn unavailable_backend_error_is_returned() {
        let handler = ShellHandler::with_runner_selector(Arc::new(|kind| {
            Err(ToolError::SandboxUnavailable {
                backend: format!("{kind:?}"),
                reason: "backend unavailable".to_string(),
            })
        }));
        let context = ExecContext {
            sandbox: SandboxKind::Landlock,
            ..ExecContext::default()
        };

        let error = handler
            .execute(&context, json!({"command": "echo hello"}))
            .await
            .expect_err("backend selection should fail");

        assert!(matches!(error, ToolError::SandboxUnavailable { .. }));
    }

    #[tokio::test]
    async fn returns_input_error_when_command_is_missing() {
        let handler = ShellHandler::default();
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
        let handler = ShellHandler::default();
        let context = ExecContext {
            working_dir: directory.clone(),
            timeout: Duration::from_secs(5),
            ..ExecContext::default()
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

    #[test]
    fn shell_tool_spec_keeps_medium_risk() {
        assert_eq!(shell_spec().risk_level, gemenr_core::RiskLevel::Medium);
    }
}
