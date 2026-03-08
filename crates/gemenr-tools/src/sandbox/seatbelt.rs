use std::path::Path;

use async_trait::async_trait;
use tokio::process::Command;

use super::{SandboxRunner, ShellCommand, collect_output, push_shell_invocation};
use crate::handler::{ExecContext, ToolError, ToolOutput};

/// macOS Seatbelt sandbox runner.
pub struct SeatbeltRunner;

pub(super) fn runner() -> Result<Box<dyn SandboxRunner>, ToolError> {
    if cfg!(target_os = "macos") {
        Ok(Box::new(SeatbeltRunner))
    } else {
        Err(ToolError::SandboxUnavailable {
            backend: "seatbelt".to_string(),
            reason: "Seatbelt is only available on macOS".to_string(),
        })
    }
}

#[async_trait]
impl SandboxRunner for SeatbeltRunner {
    async fn run(
        &self,
        command: &ShellCommand,
        ctx: &ExecContext,
    ) -> Result<ToolOutput, ToolError> {
        run_seatbelt(command, ctx).await
    }
}

#[cfg(target_os = "macos")]
async fn run_seatbelt(command: &ShellCommand, ctx: &ExecContext) -> Result<ToolOutput, ToolError> {
    let mut process = Command::new("sandbox-exec");
    process.arg("-p").arg(profile_for(&ctx.working_dir));
    push_shell_invocation(&mut process, &command.command);
    process.current_dir(&ctx.working_dir);
    collect_output(process).await
}

#[cfg(not(target_os = "macos"))]
async fn run_seatbelt(
    _command: &ShellCommand,
    _ctx: &ExecContext,
) -> Result<ToolOutput, ToolError> {
    Err(ToolError::SandboxUnavailable {
        backend: "seatbelt".to_string(),
        reason: "Seatbelt is only available on macOS".to_string(),
    })
}

#[cfg(target_os = "macos")]
fn profile_for(working_dir: &Path) -> String {
    let escaped = working_dir
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!(
        "(version 1)\n(allow default)\n(deny network*)\n(allow file-write* (subpath \"{escaped}\"))"
    )
}
