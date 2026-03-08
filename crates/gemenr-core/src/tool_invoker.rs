use async_trait::async_trait;
use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolInvokeError {
    /// The requested tool was not found.
    #[error("tool not found: {0}")]
    NotFound(String),
    /// The tool execution was blocked by policy.
    #[error("tool execution denied: {reason}")]
    Denied {
        /// Human-readable reason explaining why the tool was denied.
        reason: String,
    },
    /// The tool execution was not approved by the user.
    #[error("tool execution not approved: {message}")]
    ApprovalDenied {
        /// Human-readable approval message shown to the user.
        message: String,
    },
    /// The tool execution timed out.
    #[error("tool execution timed out")]
    Timeout,
    /// The tool execution was cancelled.
    #[error("tool execution cancelled")]
    Cancelled,
    /// The tool execution failed after it started running.
    #[error("tool execution failed: {message}")]
    Execution {
        /// Human-readable execution failure message.
        message: String,
    },
}

/// Backward-compatible Phase 1 policy decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Tool call is allowed without confirmation.
    Allow,
    /// Tool call needs user confirmation with the provided message.
    NeedConfirmation(String),
    /// Tool call is denied with the provided reason.
    Deny(String),
}

/// Context used when evaluating a tool execution policy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PolicyContext {
    /// Organization identifier associated with the turn.
    pub organization_id: Option<String>,
    /// Workspace identifier associated with the turn.
    pub workspace_id: Option<String>,
    /// Conversation identifier associated with the turn.
    pub conversation_id: Option<String>,
}

/// Sandbox backend selected by policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxKind {
    /// Run the tool without a sandbox wrapper.
    None,
    /// Run inside a macOS Seatbelt sandbox.
    Seatbelt,
    /// Run inside a Linux Landlock sandbox.
    Landlock,
}

/// Final execution plan returned by policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionPolicy {
    /// Allow execution with the selected sandbox backend.
    Allow {
        /// Sandbox backend requested by policy.
        sandbox: SandboxKind,
    },
    /// Require confirmation before execution.
    NeedConfirmation {
        /// User-facing confirmation prompt.
        message: String,
        /// Sandbox backend requested by policy.
        sandbox: SandboxKind,
    },
    /// Deny execution with a human-readable reason.
    Deny {
        /// Reason for the denial.
        reason: String,
    },
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
    fn check_policy(
        &self,
        name: &str,
        arguments: &serde_json::Value,
        context: &PolicyContext,
    ) -> ExecutionPolicy;

    /// Execute a tool call.
    async fn invoke(
        &self,
        call_id: &str,
        name: &str,
        arguments: serde_json::Value,
        cancelled: Arc<AtomicBool>,
    ) -> Result<ToolInvokeResult, ToolInvokeError>;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ExecutionPolicy, SandboxKind, ToolInvokeError};

    #[test]
    fn execution_policy_variants_are_comparable() {
        assert_eq!(
            ExecutionPolicy::Allow {
                sandbox: SandboxKind::Seatbelt,
            },
            ExecutionPolicy::Allow {
                sandbox: SandboxKind::Seatbelt,
            }
        );
        assert_eq!(
            ExecutionPolicy::NeedConfirmation {
                message: "confirm".to_string(),
                sandbox: SandboxKind::Landlock,
            },
            ExecutionPolicy::NeedConfirmation {
                message: "confirm".to_string(),
                sandbox: SandboxKind::Landlock,
            }
        );
        assert_eq!(
            ExecutionPolicy::Deny {
                reason: "nope".to_string(),
            },
            ExecutionPolicy::Deny {
                reason: "nope".to_string(),
            }
        );
    }

    #[test]
    fn sandbox_kind_serializes_to_stable_values() {
        assert_eq!(
            serde_json::to_value(SandboxKind::None).expect("serialize"),
            json!("none")
        );
        assert_eq!(
            serde_json::to_value(SandboxKind::Seatbelt).expect("serialize"),
            json!("seatbelt")
        );
        assert_eq!(
            serde_json::to_value(SandboxKind::Landlock).expect("serialize"),
            json!("landlock")
        );
    }

    #[test]
    fn tool_invoke_error_variants_are_distinguishable() {
        assert!(matches!(
            ToolInvokeError::Denied {
                reason: "policy blocked".to_string(),
            },
            ToolInvokeError::Denied { .. }
        ));
        assert!(matches!(
            ToolInvokeError::ApprovalDenied {
                message: "approval required".to_string(),
            },
            ToolInvokeError::ApprovalDenied { .. }
        ));
        assert!(matches!(ToolInvokeError::Timeout, ToolInvokeError::Timeout));
        assert!(matches!(
            ToolInvokeError::Execution {
                message: "handler crashed".to_string(),
            },
            ToolInvokeError::Execution { .. }
        ));
    }

    #[test]
    fn tool_invoke_error_display_is_actionable() {
        assert!(
            ToolInvokeError::Denied {
                reason: "policy blocked shell access".to_string(),
            }
            .to_string()
            .contains("policy blocked shell access")
        );
        assert!(
            ToolInvokeError::ApprovalDenied {
                message: "user rejected confirmation".to_string(),
            }
            .to_string()
            .contains("user rejected confirmation")
        );
        assert_eq!(
            ToolInvokeError::Timeout.to_string(),
            "tool execution timed out"
        );
        assert!(
            ToolInvokeError::Execution {
                message: "process exited with status 1".to_string(),
            }
            .to_string()
            .contains("process exited with status 1")
        );
    }
}
