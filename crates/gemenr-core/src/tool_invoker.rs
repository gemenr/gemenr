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

/// Tool call payload used during the authorization phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRequest {
    /// Unique identifier for this tool call.
    pub call_id: String,
    /// Registered tool name.
    pub name: String,
    /// Tool arguments serialized as JSON.
    pub arguments: serde_json::Value,
}

/// Authorized tool call that carries the resolved execution policy forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedToolCall {
    /// Original tool call request.
    pub request: ToolCallRequest,
    /// Execution policy resolved during authorization.
    pub policy: ExecutionPolicy,
}

/// Result of authorizing a tool call request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizationDecision {
    /// The tool call is ready to execute immediately.
    Prepared(PreparedToolCall),
    /// The tool call is authorized pending user confirmation.
    NeedConfirmation {
        /// Prepared tool call to execute after approval.
        prepared: PreparedToolCall,
        /// User-facing confirmation message.
        message: String,
    },
    /// The tool call is denied before execution starts.
    Denied {
        /// Human-readable reason for the denial.
        reason: String,
    },
}

/// Abstract interface for registered tool discovery.
pub trait ToolCatalog: Send + Sync {
    /// Look up a registered tool definition by name.
    fn lookup(&self, name: &str) -> Option<&ToolSpec>;

    /// Borrow all registered tool specifications.
    ///
    /// Implementations should retain an internal cache so callers can inspect
    /// tool metadata without forcing a fresh allocation on every turn.
    fn list_specs(&self) -> &[ToolSpec];
}

/// Abstract interface for policy authorization.
pub trait ToolAuthorizer: Send + Sync {
    /// Resolve whether a tool call may execute in the given context.
    fn authorize(
        &self,
        request: &ToolCallRequest,
        context: &PolicyContext,
    ) -> AuthorizationDecision;
}

/// Abstract interface for executing already-authorized tool calls.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool call that already passed authorization.
    async fn invoke(
        &self,
        prepared: PreparedToolCall,
        cancelled: Arc<AtomicBool>,
    ) -> Result<ToolInvokeResult, ToolInvokeError>;
}

/// Aggregated tool contract used by the runtime.
///
/// This trait stays available as a compatibility layer so callers can inject a
/// single trait object while implementations keep catalog, authorization, and
/// execution responsibilities separated.
pub trait ToolInvoker: ToolCatalog + ToolAuthorizer + ToolExecutor {}

impl<T> ToolInvoker for T where T: ToolCatalog + ToolAuthorizer + ToolExecutor + ?Sized {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use async_trait::async_trait;
    use serde_json::json;

    use super::{
        AuthorizationDecision, ExecutionPolicy, PolicyContext, PreparedToolCall, SandboxKind,
        ToolAuthorizer, ToolCallRequest, ToolCatalog, ToolExecutor, ToolInvokeError,
        ToolInvokeResult, ToolInvoker,
    };

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

    #[test]
    fn authorization_decision_preserves_prepared_call() {
        let request = ToolCallRequest {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            arguments: json!({"command": "pwd"}),
        };
        let prepared = PreparedToolCall {
            request: request.clone(),
            policy: ExecutionPolicy::NeedConfirmation {
                message: "confirm shell".to_string(),
                sandbox: SandboxKind::Seatbelt,
            },
        };

        assert_eq!(
            AuthorizationDecision::Prepared(prepared.clone()),
            AuthorizationDecision::Prepared(prepared.clone())
        );
        assert_eq!(prepared.request, request);
        assert_eq!(prepared.request.call_id, "call-1");
        assert_eq!(prepared.request.name, "shell");
        assert_eq!(prepared.request.arguments, json!({"command": "pwd"}));
    }

    #[test]
    fn tool_invoker_blanket_trait_is_object_safe() {
        struct DummyInvoker;

        impl ToolCatalog for DummyInvoker {
            fn lookup(&self, _name: &str) -> Option<&crate::tool_spec::ToolSpec> {
                None
            }

            fn list_specs(&self) -> &[crate::tool_spec::ToolSpec] {
                &[]
            }
        }

        impl ToolAuthorizer for DummyInvoker {
            fn authorize(
                &self,
                request: &ToolCallRequest,
                _context: &PolicyContext,
            ) -> AuthorizationDecision {
                AuthorizationDecision::Prepared(PreparedToolCall {
                    request: request.clone(),
                    policy: ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    },
                })
            }
        }

        #[async_trait]
        impl ToolExecutor for DummyInvoker {
            async fn invoke(
                &self,
                _prepared: PreparedToolCall,
                _cancelled: Arc<AtomicBool>,
            ) -> Result<ToolInvokeResult, ToolInvokeError> {
                Ok(ToolInvokeResult {
                    content: String::new(),
                    is_error: false,
                })
            }
        }

        let invoker: Arc<dyn ToolInvoker> = Arc::new(DummyInvoker);
        assert!(invoker.lookup("missing").is_none());
        assert!(invoker.list_specs().is_empty());
    }
}
