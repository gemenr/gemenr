use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::tool_spec::ToolSpec;

/// Result of a successful tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolInvokeResult {
    /// Output content from the tool.
    pub content: String,
    /// Whether the tool reported a non-fatal error condition.
    pub is_error: bool,
}

/// Errors that can occur while invoking a tool.
#[derive(Debug, thiserror::Error)]
pub enum ToolInvokeError {
    /// The requested tool was not found.
    #[error("tool not found: {0}")]
    NotFound(String),
    /// The tool execution failed.
    #[error("execution error: {0}")]
    Execution(String),
    /// The tool execution timed out.
    #[error("tool execution timed out")]
    Timeout,
    /// The tool execution was cancelled.
    #[error("tool execution cancelled")]
    Cancelled,
}

/// Policy decision for a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Tool call is allowed without confirmation.
    Allow,
    /// Tool call needs user confirmation with the provided message.
    NeedConfirmation(String),
    /// Tool call is denied with the provided reason.
    Deny(String),
}

/// Abstract interface for tool lookup, policy checks, and execution.
///
/// This trait is defined in `gemenr-core` so the runtime can talk to the tool
/// system without depending on a concrete tool crate.
#[async_trait]
pub trait ToolInvoker: Send + Sync {
    /// Look up a registered tool definition by name.
    fn lookup(&self, name: &str) -> Option<&ToolSpec>;

    /// List all registered tool specifications.
    fn list_specs(&self) -> Vec<ToolSpec>;

    /// Evaluate the policy for a tool call.
    fn check_policy(&self, name: &str, arguments: &serde_json::Value) -> PolicyDecision;

    /// Execute a tool call.
    async fn invoke(
        &self,
        call_id: &str,
        name: &str,
        arguments: serde_json::Value,
        cancelled: Arc<AtomicBool>,
    ) -> Result<ToolInvokeResult, ToolInvokeError>;
}
