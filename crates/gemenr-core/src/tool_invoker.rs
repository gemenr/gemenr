use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::tool_spec::ToolSpec;
use async_trait::async_trait;

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

/// Opaque execution context attached to a prepared tool call.
///
/// The authorizer populates this with implementation-specific data, and the
/// executor consumes it. Core does not inspect or interpret this value.
pub struct ExecutionContext(Box<dyn Any + Send + Sync>);

impl ExecutionContext {
    /// Create a new execution context wrapping the given value.
    #[must_use]
    pub fn new<T>(value: T) -> Self
    where
        T: Any + Send + Sync + 'static,
    {
        Self(Box::new(value))
    }

    /// Attempt to downcast the context to a concrete type by reference.
    #[must_use]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }

    /// Attempt to consume and downcast the context to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<T, Self> {
        match self.0.downcast::<T>() {
            Ok(value) => Ok(*value),
            Err(value) => Err(Self(value)),
        }
    }
}

impl fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("type_id", &(*self.0).type_id())
            .finish()
    }
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

/// A tool call that has been authorized and is ready for execution.
#[derive(Debug)]
pub struct PreparedToolCall {
    /// The original call request.
    pub request: ToolCallRequest,
    /// Opaque execution context from the authorizer.
    pub execution_context: ExecutionContext,
}

/// Result of authorizing a tool call request.
#[derive(Debug)]
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
        AuthorizationDecision, ExecutionContext, PolicyContext, PreparedToolCall, ToolAuthorizer,
        ToolCallRequest, ToolCatalog, ToolExecutor, ToolInvokeError, ToolInvokeResult, ToolInvoker,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestSandboxKind {
        None,
        Seatbelt,
        Landlock,
    }

    #[test]
    fn execution_context_supports_type_erasure_round_trip() {
        let context = ExecutionContext::new(TestSandboxKind::Landlock);

        assert_eq!(
            context.downcast_ref::<TestSandboxKind>(),
            Some(&TestSandboxKind::Landlock)
        );

        let restored = ExecutionContext::new(TestSandboxKind::Seatbelt)
            .downcast::<TestSandboxKind>()
            .expect("context should downcast");

        assert_eq!(restored, TestSandboxKind::Seatbelt);
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
        let decision = AuthorizationDecision::Prepared(PreparedToolCall {
            request: request.clone(),
            execution_context: ExecutionContext::new(TestSandboxKind::Seatbelt),
        });

        match decision {
            AuthorizationDecision::Prepared(prepared) => {
                assert_eq!(prepared.request, request);
                assert_eq!(prepared.request.call_id, "call-1");
                assert_eq!(prepared.request.name, "shell");
                assert_eq!(prepared.request.arguments, json!({"command": "pwd"}));
                assert_eq!(
                    prepared.execution_context.downcast_ref::<TestSandboxKind>(),
                    Some(&TestSandboxKind::Seatbelt)
                );
            }
            other => panic!("expected prepared decision, got {other:?}"),
        }
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
                    execution_context: ExecutionContext::new(TestSandboxKind::None),
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
