use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::{ExecutionPolicy, PolicyContext};
use serde::{Deserialize, Serialize};

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

/// A request to invoke a specific tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallSpec {
    /// Unique identifier for this tool call.
    pub call_id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments serialized as JSON.
    pub arguments: serde_json::Value,
}

/// Context for tool execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecContext {
    /// Working directory for the tool execution.
    pub working_dir: PathBuf,
    /// Timeout applied to the tool execution.
    pub timeout: Duration,
    /// Policy context associated with the invocation.
    pub policy_context: PolicyContext,
    /// Effective execution policy selected for the call.
    pub execution_policy: Option<ExecutionPolicy>,
}

impl Default for ExecContext {
    fn default() -> Self {
        Self {
            working_dir: std::env::current_dir().unwrap_or_default(),
            timeout: Duration::from_secs(120),
            policy_context: PolicyContext::default(),
            execution_policy: None,
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;

    use super::{ExecContext, ToolCallSpec, ToolError};

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
