use async_trait::async_trait;

use super::{SandboxRunner, ShellCommand};
use crate::handler::{ExecContext, ToolError, ToolOutput};

/// Linux Landlock sandbox runner.
pub struct LandlockRunner;

pub(super) fn runner() -> Result<Box<dyn SandboxRunner>, ToolError> {
    if cfg!(target_os = "linux") {
        Ok(Box::new(LandlockRunner))
    } else {
        Err(ToolError::SandboxUnavailable {
            backend: "landlock".to_string(),
            reason: "Landlock is only available on Linux".to_string(),
        })
    }
}

#[async_trait]
impl SandboxRunner for LandlockRunner {
    async fn run(
        &self,
        command: &ShellCommand,
        ctx: &ExecContext,
    ) -> Result<ToolOutput, ToolError> {
        run_landlock(command, ctx).await
    }
}

#[cfg(target_os = "linux")]
async fn run_landlock(command: &ShellCommand, ctx: &ExecContext) -> Result<ToolOutput, ToolError> {
    use std::os::unix::process::CommandExt;

    use landlock::{
        ABI, Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetCreatedAttr,
        path_beneath_rules,
    };

    let mut process = super::build_shell_command(&command.command, &ctx.working_dir);
    let working_dir = ctx.working_dir.clone();

    process.pre_exec(move || {
        let abi = ABI::V1;
        let read_access = AccessFs::from_read(abi);
        let write_access = AccessFs::from_write(abi);
        let ruleset = Ruleset::default()
            .handle_access(AccessFs::from_all(abi))
            .map_err(to_io_error)?
            .create()
            .map_err(to_io_error)?
            .set_compatibility(CompatLevel::HardRequirement)
            .map_err(to_io_error)?
            .add_rules(path_beneath_rules(
                [
                    working_dir.as_path(),
                    std::path::Path::new("/bin"),
                    std::path::Path::new("/usr"),
                    std::path::Path::new("/lib"),
                    std::path::Path::new("/lib64"),
                    std::path::Path::new("/etc"),
                    std::path::Path::new("/tmp"),
                ],
                read_access,
            ))
            .map_err(to_io_error)?
            .add_rules(path_beneath_rules([working_dir.as_path()], write_access))
            .map_err(to_io_error)?;

        ruleset.restrict_self().map_err(to_io_error)?;
        Ok(())
    });

    super::collect_output(process).await
}

#[cfg(target_os = "linux")]
fn to_io_error(error: landlock::RulesetError) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

#[cfg(not(target_os = "linux"))]
async fn run_landlock(
    _command: &ShellCommand,
    _ctx: &ExecContext,
) -> Result<ToolOutput, ToolError> {
    Err(ToolError::SandboxUnavailable {
        backend: "landlock".to_string(),
        reason: "Landlock is only available on Linux".to_string(),
    })
}
