//! Sandbox backend selection and shared helpers for shell execution.

use std::path::Path;

use async_trait::async_trait;
use tokio::process::Command;

use crate::SandboxKind;
use crate::handler::{ExecContext, ToolError, ToolOutput, trace_tool_failure};

#[cfg(target_os = "linux")]
mod landlock;
#[cfg(not(target_os = "linux"))]
mod landlock;
#[cfg(target_os = "macos")]
mod seatbelt;
#[cfg(not(target_os = "macos"))]
mod seatbelt;

const SHELL_OUTPUT_LIMIT: usize = 10_000;

/// Parsed shell command passed to a sandbox backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    /// Raw shell command string.
    pub command: String,
}

/// Executes a shell command inside a selected sandbox backend.
#[async_trait]
pub trait SandboxRunner: Send + Sync {
    /// Run `command` using the current backend.
    async fn run(&self, command: &ShellCommand, ctx: &ExecContext)
    -> Result<ToolOutput, ToolError>;
}

/// Create a backend runner for the requested sandbox kind.
pub fn runner_for(kind: SandboxKind) -> Result<Box<dyn SandboxRunner>, ToolError> {
    match kind {
        SandboxKind::None => Err(ToolError::SandboxUnavailable {
            backend: "none".to_string(),
            reason: "no sandbox backend requested".to_string(),
        }),
        SandboxKind::Seatbelt => seatbelt::runner(),
        SandboxKind::Landlock => landlock::runner(),
    }
}

/// Run a shell command without any sandbox wrapper.
pub async fn run_without_sandbox(
    command: &ShellCommand,
    ctx: &ExecContext,
) -> Result<ToolOutput, ToolError> {
    collect_output(build_shell_command(&command.command, &ctx.working_dir)).await
}

pub(crate) fn build_shell_command(command: &str, working_dir: &Path) -> Command {
    let mut process = if cfg!(target_os = "windows") {
        let mut command_builder = Command::new("cmd");
        command_builder.arg("/C").arg(command);
        command_builder
    } else {
        let mut command_builder = Command::new("sh");
        command_builder.arg("-c").arg(command);
        command_builder
    };
    process.current_dir(working_dir);
    process
}

pub(crate) fn push_shell_invocation(command: &mut Command, shell_command: &str) {
    if cfg!(target_os = "windows") {
        command.arg("cmd").arg("/C").arg(shell_command);
    } else {
        command.arg("sh").arg("-c").arg(shell_command);
    }
}

pub(crate) async fn collect_output(mut command: Command) -> Result<ToolOutput, ToolError> {
    let output = command.output().await.map_err(|error| {
        trace_tool_failure("shell", "spawn", &error);
        ToolError::Execution {
            exit_code: None,
            stderr: error.to_string(),
        }
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

fn truncate_output(content: &str, limit: usize) -> String {
    if content.chars().count() <= limit {
        return content.to_string();
    }

    let truncated = content.chars().take(limit).collect::<String>();
    format!("{truncated}\n...[truncated]")
}

#[cfg(test)]
mod tests {
    use super::runner_for;
    use crate::{SandboxKind, ToolError};

    #[test]
    fn backend_selection_matches_current_platform() {
        #[cfg(target_os = "macos")]
        assert!(runner_for(SandboxKind::Seatbelt).is_ok());
        #[cfg(not(target_os = "macos"))]
        assert!(matches!(
            runner_for(SandboxKind::Seatbelt),
            Err(ToolError::SandboxUnavailable { .. })
        ));

        #[cfg(target_os = "linux")]
        assert!(runner_for(SandboxKind::Landlock).is_ok());
        #[cfg(not(target_os = "linux"))]
        assert!(matches!(
            runner_for(SandboxKind::Landlock),
            Err(ToolError::SandboxUnavailable { .. })
        ));
    }

    #[test]
    fn unavailable_backend_is_explicit_on_other_platforms() {
        #[cfg(not(target_os = "macos"))]
        assert!(matches!(
            runner_for(SandboxKind::Seatbelt),
            Err(ToolError::SandboxUnavailable { .. })
        ));

        #[cfg(not(target_os = "linux"))]
        assert!(matches!(
            runner_for(SandboxKind::Landlock),
            Err(ToolError::SandboxUnavailable { .. })
        ));
    }
}
