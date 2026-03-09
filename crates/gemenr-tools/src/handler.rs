use std::error::Error as StdError;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::{PolicyContext, ToolCallRequest};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::policy::SandboxKind;

/// Handler for executing a registered tool.
///
/// Each tool implements this trait to define its execution behavior.
/// The handler receives the execution context plus JSON arguments and returns
/// a structured result.
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with the given context and arguments.
    async fn execute(
        &self,
        ctx: &ExecContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError>;
}

/// Successful output from a tool execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The output content produced by the tool execution.
    pub content: String,
}

/// Backward-compatible alias for the shared tool call request type.
pub type ToolCallSpec = ToolCallRequest;

/// Context for tool execution.
#[derive(Debug, Clone)]
pub struct ExecContext {
    /// Working directory for the tool execution.
    pub working_dir: PathBuf,
    /// Timeout applied to the tool execution.
    pub timeout: Duration,
    /// Policy context associated with the invocation.
    pub policy_context: PolicyContext,
    /// Sandbox backend selected for the prepared call.
    pub sandbox: SandboxKind,
    /// Shared cancellation flag propagated from the runtime.
    pub cancelled: Arc<AtomicBool>,
}

impl Default for ExecContext {
    fn default() -> Self {
        Self {
            working_dir: std::env::current_dir().unwrap_or_default(),
            timeout: Duration::from_secs(120),
            policy_context: PolicyContext::default(),
            sandbox: SandboxKind::None,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Errors that can occur during tool execution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    /// Invalid or malformed input arguments.
    #[error("invalid input: {message}")]
    Input { message: String },

    /// Tool execution failed.
    #[error("execution failed (exit code {exit_code:?}): {stderr}")]
    Execution {
        /// Optional process exit code associated with the failure.
        exit_code: Option<i32>,
        /// Standard error output collected from the tool.
        stderr: String,
    },

    /// Tool execution timed out.
    #[error("execution timed out after {0:?}")]
    Timeout(Duration),

    /// Tool execution was cancelled before completion.
    #[error("execution cancelled")]
    Cancelled,

    /// Tool was not found in the registry.
    #[error("tool not found: {0}")]
    NotFound(String),

    /// Requested sandbox backend is unavailable on this platform or host.
    #[error("sandbox backend `{backend}` is unavailable: {reason}")]
    SandboxUnavailable {
        /// Backend name that was requested.
        backend: String,
        /// Human-readable reason for the failure.
        reason: String,
    },
}

pub(crate) fn trace_tool_failure<E>(tool_name: &str, action: &str, error: &E)
where
    E: StdError + 'static,
{
    warn!(
        tool_name,
        action,
        error = %error,
        error_debug = ?error,
        error_chain = ?error_chain(error),
        "Tool execution failed"
    );
}

fn error_chain(error: &(dyn StdError + 'static)) -> Vec<String> {
    let mut chain = vec![error.to_string()];
    let mut source = error.source();

    while let Some(next) = source {
        chain.push(next.to_string());
        source = next.source();
    }

    chain
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use serde_json::json;

    use super::{ExecContext, ToolCallSpec, ToolError};
    use crate::policy::SandboxKind;

    #[test]
    fn tool_call_spec_exposes_fields() {
        let call = ToolCallSpec {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            arguments: json!({"command": "pwd"}),
        };

        assert_eq!(call.call_id, "call-1");
        assert_eq!(call.name, "shell");
        assert_eq!(call.arguments, json!({"command": "pwd"}));
    }

    #[test]
    fn exec_context_defaults_to_two_minute_timeout() {
        let context = ExecContext::default();

        assert_eq!(context.timeout, Duration::from_secs(120));
    }

    #[test]
    fn exec_context_defaults_to_unsandboxed_execution() {
        let context = ExecContext::default();

        assert_eq!(context.sandbox, SandboxKind::None);
    }

    #[test]
    fn exec_context_defaults_to_uncancelled_flag() {
        let context = ExecContext::default();

        assert!(!context.cancelled.load(Ordering::Relaxed));
    }

    #[test]
    fn tool_error_display_is_human_readable() {
        let input = ToolError::Input {
            message: "missing field".to_string(),
        };
        let execution = ToolError::Execution {
            exit_code: Some(2),
            stderr: "permission denied".to_string(),
        };
        let timeout = ToolError::Timeout(Duration::from_secs(5));
        let cancelled = ToolError::Cancelled;
        let not_found = ToolError::NotFound("shell".to_string());
        let unavailable = ToolError::SandboxUnavailable {
            backend: "seatbelt".to_string(),
            reason: "unsupported host".to_string(),
        };

        assert_eq!(input.to_string(), "invalid input: missing field");
        assert_eq!(
            execution.to_string(),
            "execution failed (exit code Some(2)): permission denied"
        );
        assert_eq!(timeout.to_string(), "execution timed out after 5s");
        assert_eq!(cancelled.to_string(), "execution cancelled");
        assert_eq!(not_found.to_string(), "tool not found: shell");
        assert_eq!(
            unavailable.to_string(),
            "sandbox backend `seatbelt` is unavailable: unsupported host"
        );
    }
}
