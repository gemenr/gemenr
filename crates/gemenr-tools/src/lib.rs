//! Tool registration, policy evaluation, and execution primitives.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::{ToolSpec, tool_invoker};
use tracing::{debug, warn};

/// Built-in tool implementations.
pub mod builtin;
/// Tool execution handler contracts.
pub mod handler;
/// MCP stdio integration.
pub mod mcp;
/// Scoped policy evaluation.
pub mod policy;
/// Sandbox backend selection.
pub mod sandbox;

pub use handler::{ExecContext, ToolCallSpec, ToolError, ToolHandler, ToolOutput};
pub use mcp::{
    McpClient, McpError, McpRemoteTool, McpToolAdapter, McpToolResult, mcp_tool_name,
    register_mcp_servers,
};
pub use policy::{
    ExecutionPolicy, PolicyEvaluator, PolicyRule, PolicyScope, RuleBasedPolicyEvaluator,
    SandboxKind,
};

/// Create a filtered tool invoker view that only exposes `allowed` tools.
#[must_use]
pub fn allowlist_tool_invoker(
    inner: Arc<dyn tool_invoker::ToolInvoker>,
    allowed: &[String],
) -> Arc<dyn tool_invoker::ToolInvoker> {
    let allowed_set: HashSet<String> = allowed.iter().cloned().collect();
    let allowed_specs = inner
        .list_specs()
        .iter()
        .filter(|spec| allowed_set.contains(&spec.name))
        .cloned()
        .collect();

    Arc::new(AllowlistedToolInvoker {
        inner,
        allowed: allowed_set,
        allowed_specs,
    })
}

struct AllowlistedToolInvoker {
    inner: Arc<dyn tool_invoker::ToolInvoker>,
    allowed: HashSet<String>,
    allowed_specs: Vec<ToolSpec>,
}

impl tool_invoker::ToolCatalog for AllowlistedToolInvoker {
    fn lookup(&self, name: &str) -> Option<&ToolSpec> {
        if self.allowed.contains(name) {
            self.inner.lookup(name)
        } else {
            None
        }
    }

    fn list_specs(&self) -> &[ToolSpec] {
        &self.allowed_specs
    }
}

impl tool_invoker::ToolAuthorizer for AllowlistedToolInvoker {
    fn authorize(
        &self,
        request: &tool_invoker::ToolCallRequest,
        context: &tool_invoker::PolicyContext,
    ) -> tool_invoker::AuthorizationDecision {
        if self.allowed.contains(&request.name) {
            self.inner.authorize(request, context)
        } else {
            tool_invoker::AuthorizationDecision::Denied {
                reason: format!("tool `{}` is not available for this job", request.name),
            }
        }
    }
}

#[async_trait]
impl tool_invoker::ToolExecutor for AllowlistedToolInvoker {
    async fn invoke(
        &self,
        prepared: tool_invoker::PreparedToolCall,
        cancelled: Arc<AtomicBool>,
    ) -> Result<tool_invoker::ToolInvokeResult, tool_invoker::ToolInvokeError> {
        if self.allowed.contains(&prepared.request.name) {
            self.inner.invoke(prepared, cancelled).await
        } else {
            Err(tool_invoker::ToolInvokeError::NotFound(
                prepared.request.name.clone(),
            ))
        }
    }
}

/// Catalog of all registered tools.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCatalog {
    /// All registered tool specifications.
    pub tools: Vec<ToolSpec>,
}

/// Central registry and execution engine for tools.
pub struct ToolPlane {
    tools: HashMap<String, (ToolSpec, Box<dyn ToolHandler>)>,
    specs_cache: Vec<ToolSpec>,
    policy_evaluator: Arc<dyn PolicyEvaluator>,
}

impl ToolPlane {
    /// Create an empty tool plane.
    #[must_use]
    pub fn new() -> Self {
        Self::with_policy_evaluator(Arc::new(RuleBasedPolicyEvaluator::default()))
    }

    /// Create an empty tool plane with a custom policy evaluator.
    #[must_use]
    pub fn with_policy_evaluator(policy_evaluator: Arc<dyn PolicyEvaluator>) -> Self {
        Self {
            tools: HashMap::new(),
            specs_cache: Vec::new(),
            policy_evaluator,
        }
    }

    /// Replace the policy evaluator used for subsequent checks.
    pub fn set_policy_evaluator(&mut self, policy_evaluator: Arc<dyn PolicyEvaluator>) {
        self.policy_evaluator = policy_evaluator;
    }

    /// Register enabled stdio MCP servers and their remote tools.
    pub async fn register_mcp_servers(
        &mut self,
        config: &gemenr_core::config::McpConfig,
    ) -> Result<(), crate::mcp::McpError> {
        crate::mcp::register_mcp_servers(self, config).await
    }

    /// Register a tool with its specification and handler.
    pub fn register(&mut self, spec: ToolSpec, handler: Box<dyn ToolHandler>) {
        debug!(tool = %spec.name, "registering tool");
        let cache_spec = spec.clone();

        if let Some(existing) = self
            .specs_cache
            .iter_mut()
            .find(|existing| existing.name == cache_spec.name)
        {
            *existing = cache_spec;
        } else {
            self.specs_cache.push(cache_spec);
            self.specs_cache
                .sort_by(|left, right| left.name.cmp(&right.name));
        }

        self.tools.insert(spec.name.clone(), (spec, handler));
    }

    /// Look up a tool specification by name.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.get(name).map(|(spec, _)| spec)
    }

    /// List all registered tools as a catalog.
    #[must_use]
    pub fn list(&self) -> ToolCatalog {
        ToolCatalog {
            tools: self.specs_cache.clone(),
        }
    }

    /// Authorize a tool call request against the configured policy evaluator.
    #[must_use]
    pub fn authorize(
        &self,
        request: &tool_invoker::ToolCallRequest,
        context: &tool_invoker::PolicyContext,
    ) -> tool_invoker::AuthorizationDecision {
        let Some(spec) = self.lookup(&request.name) else {
            return tool_invoker::AuthorizationDecision::Denied {
                reason: format!("Tool '{}' not found", request.name),
            };
        };

        let policy = self.policy_evaluator.evaluate(context, spec, request);
        match policy {
            ExecutionPolicy::Allow { sandbox } => {
                tool_invoker::AuthorizationDecision::Prepared(tool_invoker::PreparedToolCall {
                    request: request.clone(),
                    execution_context: tool_invoker::ExecutionContext::new(sandbox),
                })
            }
            ExecutionPolicy::NeedConfirmation { message, sandbox } => {
                tool_invoker::AuthorizationDecision::NeedConfirmation {
                    prepared: tool_invoker::PreparedToolCall {
                        request: request.clone(),
                        execution_context: tool_invoker::ExecutionContext::new(sandbox),
                    },
                    message,
                }
            }
            ExecutionPolicy::Deny { reason } => {
                tool_invoker::AuthorizationDecision::Denied { reason }
            }
        }
    }

    /// Invoke an already-authorized tool call with the provided execution context.
    pub async fn invoke(
        &self,
        prepared: &tool_invoker::PreparedToolCall,
        ctx: &ExecContext,
        cancelled: Arc<AtomicBool>,
    ) -> Result<ToolOutput, ToolError> {
        let Some((_, handler)) = self.tools.get(&prepared.request.name) else {
            return Err(ToolError::NotFound(prepared.request.name.clone()));
        };

        debug!(call_id = %prepared.request.call_id, tool = %prepared.request.name, "invoking tool");

        let mut execution_context = ctx.clone();
        execution_context.sandbox = prepared
            .execution_context
            .downcast_ref::<SandboxKind>()
            .copied()
            .unwrap_or_else(|| {
                warn!(
                    call_id = %prepared.request.call_id,
                    tool = %prepared.request.name,
                    "prepared tool call missing sandbox context; defaulting to unsandboxed execution"
                );
                SandboxKind::None
            });

        let execution = async {
            let tool_future =
                handler.execute(&execution_context, prepared.request.arguments.clone());
            tokio::pin!(tool_future);

            loop {
                tokio::select! {
                    result = &mut tool_future => return result,
                    _ = tokio::time::sleep(Duration::from_millis(25)) => {
                        if cancelled.load(Ordering::Relaxed) {
                            warn!(call_id = %prepared.request.call_id, tool = %prepared.request.name, "tool invocation cancelled");
                            return Err(ToolError::Cancelled);
                        }
                    }
                }
            }
        };

        match tokio::time::timeout(ctx.timeout, execution).await {
            Ok(result) => result,
            Err(_) => {
                warn!(call_id = %prepared.request.call_id, tool = %prepared.request.name, timeout = ?ctx.timeout, "tool invocation timed out");
                Err(ToolError::Timeout(ctx.timeout))
            }
        }
    }
}

impl Default for ToolPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl tool_invoker::ToolCatalog for ToolPlane {
    fn lookup(&self, name: &str) -> Option<&ToolSpec> {
        ToolPlane::lookup(self, name)
    }

    fn list_specs(&self) -> &[ToolSpec] {
        &self.specs_cache
    }
}

impl tool_invoker::ToolAuthorizer for ToolPlane {
    fn authorize(
        &self,
        request: &tool_invoker::ToolCallRequest,
        context: &tool_invoker::PolicyContext,
    ) -> tool_invoker::AuthorizationDecision {
        ToolPlane::authorize(self, request, context)
    }
}

#[async_trait]
impl tool_invoker::ToolExecutor for ToolPlane {
    async fn invoke(
        &self,
        prepared: tool_invoker::PreparedToolCall,
        cancelled: Arc<AtomicBool>,
    ) -> Result<tool_invoker::ToolInvokeResult, tool_invoker::ToolInvokeError> {
        let ctx = ExecContext {
            cancelled: Arc::clone(&cancelled),
            ..ExecContext::default()
        };

        match ToolPlane::invoke(self, &prepared, &ctx, cancelled).await {
            Ok(output) => Ok(tool_invoker::ToolInvokeResult {
                content: output.content,
                is_error: false,
            }),
            Err(ToolError::NotFound(name)) => Err(tool_invoker::ToolInvokeError::NotFound(name)),
            Err(ToolError::Timeout(_)) => Err(tool_invoker::ToolInvokeError::Timeout),
            Err(ToolError::Cancelled) => Err(tool_invoker::ToolInvokeError::Cancelled),
            Err(error) => {
                handler::trace_tool_failure(&prepared.request.name, "invoke", &error);
                Err(tool_invoker::ToolInvokeError::Execution {
                    message: error.to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use gemenr_core::{
        AuthorizationDecision, PolicyContext, PreparedToolCall, RiskLevel, ToolCatalog as _,
    };
    use serde_json::json;

    use super::{
        ExecContext, ExecutionPolicy, PolicyRule, PolicyScope, RuleBasedPolicyEvaluator,
        SandboxKind, ToolCallSpec, ToolError, ToolHandler, ToolOutput, ToolPlane,
    };

    struct StaticHandler {
        content: &'static str,
    }

    #[async_trait]
    impl ToolHandler for StaticHandler {
        async fn execute(
            &self,
            _ctx: &ExecContext,
            _args: serde_json::Value,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                content: self.content.to_string(),
            })
        }
    }

    struct SlowHandler;

    #[async_trait]
    impl ToolHandler for SlowHandler {
        async fn execute(
            &self,
            _ctx: &ExecContext,
            _args: serde_json::Value,
        ) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(ToolOutput {
                content: "done".to_string(),
            })
        }
    }

    struct ContextEchoHandler;

    #[async_trait]
    impl ToolHandler for ContextEchoHandler {
        async fn execute(
            &self,
            ctx: &ExecContext,
            _args: serde_json::Value,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                content: ctx.working_dir.display().to_string(),
            })
        }
    }

    struct SandboxEchoHandler;

    #[async_trait]
    impl ToolHandler for SandboxEchoHandler {
        async fn execute(
            &self,
            ctx: &ExecContext,
            _args: serde_json::Value,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                content: format!("{:?}", ctx.sandbox),
            })
        }
    }

    struct CountingPolicyEvaluator {
        evaluations: AtomicUsize,
    }

    impl CountingPolicyEvaluator {
        fn new() -> Self {
            Self {
                evaluations: AtomicUsize::new(0),
            }
        }

        fn count(&self) -> usize {
            self.evaluations.load(Ordering::Relaxed)
        }
    }

    impl super::PolicyEvaluator for CountingPolicyEvaluator {
        fn evaluate(
            &self,
            _ctx: &PolicyContext,
            _spec: &gemenr_core::ToolSpec,
            _call: &gemenr_core::ToolCallRequest,
        ) -> ExecutionPolicy {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            ExecutionPolicy::NeedConfirmation {
                message: "confirm shell".to_string(),
                sandbox: SandboxKind::Seatbelt,
            }
        }
    }

    fn spec(name: &str) -> gemenr_core::ToolSpec {
        gemenr_core::ToolSpec {
            name: name.to_string(),
            description: format!("{name} description"),
            input_schema: json!({"type": "object"}),
            risk_level: RiskLevel::Low,
        }
    }

    fn call(name: &str) -> ToolCallSpec {
        ToolCallSpec {
            call_id: "call-1".to_string(),
            name: name.to_string(),
            arguments: json!({}),
        }
    }

    fn prepared_call(name: &str, policy: ExecutionPolicy) -> PreparedToolCall {
        PreparedToolCall {
            request: call(name),
            execution_context: policy.into_execution_context(),
        }
    }

    #[test]
    fn register_and_lookup_tool() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(StaticHandler { content: "ok" }));

        let tool = plane.lookup("shell").expect("tool should be registered");

        assert_eq!(tool.name, "shell");
    }

    #[test]
    fn lookup_returns_none_for_missing_tool() {
        let plane = ToolPlane::new();

        assert!(plane.lookup("missing").is_none());
    }

    #[test]
    fn list_returns_all_registered_tools() {
        let mut plane = ToolPlane::new();
        plane.register(spec("fs.read"), Box::new(StaticHandler { content: "a" }));
        plane.register(spec("shell"), Box::new(StaticHandler { content: "b" }));

        let catalog = plane.list();
        let names = catalog
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(catalog.tools.len(), 2);
        assert_eq!(names, vec!["fs.read", "shell"]);
    }

    #[test]
    fn test_list_specs_returns_reference() {
        let mut plane = ToolPlane::new();
        plane.register(spec("fs.read"), Box::new(StaticHandler { content: "a" }));
        plane.register(spec("shell"), Box::new(StaticHandler { content: "b" }));

        let first = plane.list_specs();
        let second = plane.list_specs();

        assert_eq!(first.len(), 2);
        assert!(std::ptr::eq(first.as_ptr(), second.as_ptr()));
        assert_eq!(
            first
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["fs.read", "shell"]
        );
    }

    #[tokio::test]
    async fn invoke_executes_registered_tool() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(StaticHandler { content: "hello" }));

        let output = plane
            .invoke(
                &prepared_call(
                    "shell",
                    ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    },
                ),
                &ExecContext::default(),
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .expect("tool should execute");

        assert_eq!(output.content, "hello");
    }

    #[tokio::test]
    async fn invoke_passes_execution_context_to_handler() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(ContextEchoHandler));
        let context = ExecContext {
            working_dir: std::env::temp_dir().join("gemenr-context-test"),
            timeout: Duration::from_secs(1),
            ..ExecContext::default()
        };

        let output = plane
            .invoke(
                &prepared_call(
                    "shell",
                    ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    },
                ),
                &context,
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .expect("tool should receive context");

        assert_eq!(output.content, context.working_dir.display().to_string());
    }

    #[tokio::test]
    async fn invoke_downcasts_prepared_sandbox_context() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(SandboxEchoHandler));

        let output = plane
            .invoke(
                &prepared_call(
                    "shell",
                    ExecutionPolicy::Allow {
                        sandbox: SandboxKind::Seatbelt,
                    },
                ),
                &ExecContext::default(),
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .expect("sandbox should be forwarded");

        assert_eq!(output.content, "Seatbelt");
    }

    #[tokio::test]
    async fn invoke_returns_not_found_for_missing_tool() {
        let plane = ToolPlane::new();

        let error = plane
            .invoke(
                &prepared_call(
                    "missing",
                    ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    },
                ),
                &ExecContext::default(),
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .expect_err("missing tool should fail");

        assert_eq!(error, ToolError::NotFound("missing".to_string()));
    }

    #[tokio::test]
    async fn invoke_times_out_when_handler_takes_too_long() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(SlowHandler));
        let context = ExecContext {
            working_dir: std::env::current_dir().expect("current dir should exist"),
            timeout: Duration::from_millis(10),
            ..ExecContext::default()
        };

        let error = plane
            .invoke(
                &prepared_call(
                    "shell",
                    ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    },
                ),
                &context,
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .expect_err("slow tool should time out");

        assert_eq!(error, ToolError::Timeout(Duration::from_millis(10)));
    }

    #[tokio::test]
    async fn invoke_cancels_when_flag_is_set() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(SlowHandler));
        let cancelled = Arc::new(AtomicBool::new(true));

        let error = plane
            .invoke(
                &prepared_call(
                    "shell",
                    ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    },
                ),
                &ExecContext::default(),
                cancelled,
            )
            .await
            .expect_err("cancelled tool should fail");

        assert_eq!(error, ToolError::Cancelled);
    }

    #[test]
    fn authorize_returns_prepared_call_with_policy() {
        let mut plane = ToolPlane::with_policy_evaluator(Arc::new(RuleBasedPolicyEvaluator {
            rules: vec![PolicyRule {
                scope: PolicyScope::Conversation("conv-1".to_string()),
                tool_name: "shell".to_string(),
                effect: gemenr_core::config::PolicyEffect::Allow,
                sandbox: SandboxKind::Seatbelt,
            }],
        }));
        plane.register(spec("shell"), Box::new(StaticHandler { content: "ok" }));

        let decision = plane.authorize(
            &ToolCallSpec {
                call_id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: json!({"command": "pwd"}),
            },
            &PolicyContext {
                organization_id: None,
                workspace_id: None,
                conversation_id: Some("conv-1".to_string()),
            },
        );

        match decision {
            AuthorizationDecision::Prepared(prepared) => {
                assert_eq!(
                    prepared.request,
                    ToolCallSpec {
                        call_id: "call-1".to_string(),
                        name: "shell".to_string(),
                        arguments: json!({"command": "pwd"}),
                    }
                );
                assert_eq!(
                    prepared.execution_context.downcast_ref::<SandboxKind>(),
                    Some(&SandboxKind::Seatbelt)
                );
            }
            other => panic!("expected prepared decision, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn authorize_returns_need_confirmation_without_recomputing_invoke_policy() {
        let evaluator = Arc::new(CountingPolicyEvaluator::new());
        let mut plane = ToolPlane::with_policy_evaluator(evaluator.clone());
        plane.register(spec("shell"), Box::new(StaticHandler { content: "ok" }));

        let decision = plane.authorize(&call("shell"), &PolicyContext::default());
        let prepared = match decision {
            AuthorizationDecision::NeedConfirmation { prepared, message } => {
                assert_eq!(message, "confirm shell");
                prepared
            }
            other => panic!("expected confirmation, got {other:?}"),
        };

        let output = plane
            .invoke(
                &prepared,
                &ExecContext::default(),
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .expect("prepared call should execute");

        assert_eq!(output.content, "ok");
        assert_eq!(evaluator.count(), 1);
    }

    #[test]
    fn authorize_returns_need_confirmation_from_evaluator() {
        let mut plane = ToolPlane::with_policy_evaluator(Arc::new(RuleBasedPolicyEvaluator {
            rules: vec![PolicyRule {
                scope: PolicyScope::Conversation("conv-1".to_string()),
                tool_name: "shell".to_string(),
                effect: gemenr_core::config::PolicyEffect::NeedConfirmation,
                sandbox: SandboxKind::Seatbelt,
            }],
        }));
        plane.register(spec("shell"), Box::new(StaticHandler { content: "ok" }));

        let decision = plane.authorize(
            &ToolCallSpec {
                call_id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: json!({"command": "pwd"}),
            },
            &PolicyContext {
                organization_id: None,
                workspace_id: None,
                conversation_id: Some("conv-1".to_string()),
            },
        );

        match decision {
            AuthorizationDecision::NeedConfirmation { prepared, message } => {
                assert_eq!(message, "Tool 'shell' requires confirmation");
                assert_eq!(
                    prepared.request,
                    ToolCallSpec {
                        call_id: "call-1".to_string(),
                        name: "shell".to_string(),
                        arguments: json!({"command": "pwd"}),
                    }
                );
                assert_eq!(
                    prepared.execution_context.downcast_ref::<SandboxKind>(),
                    Some(&SandboxKind::Seatbelt)
                );
            }
            other => panic!("expected confirmation decision, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod allowlist_tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use async_trait::async_trait;
    use gemenr_core::{
        AuthorizationDecision, ExecutionContext, PreparedToolCall, RiskLevel, ToolAuthorizer,
        ToolCatalog, ToolExecutor, ToolInvokeError, ToolInvokeResult,
    };
    use serde_json::json;

    use super::{SandboxKind, allowlist_tool_invoker};

    struct StaticInvoker {
        specs: Vec<gemenr_core::ToolSpec>,
    }

    impl StaticInvoker {
        fn new() -> Self {
            Self {
                specs: ["shell", "fs.read"]
                    .into_iter()
                    .map(|name| gemenr_core::ToolSpec {
                        name: name.to_string(),
                        description: name.to_string(),
                        input_schema: json!({}),
                        risk_level: RiskLevel::Low,
                    })
                    .collect(),
            }
        }
    }

    impl ToolCatalog for StaticInvoker {
        fn lookup(&self, name: &str) -> Option<&gemenr_core::ToolSpec> {
            self.specs.iter().find(|spec| spec.name == name)
        }

        fn list_specs(&self) -> &[gemenr_core::ToolSpec] {
            &self.specs
        }
    }

    impl ToolAuthorizer for StaticInvoker {
        fn authorize(
            &self,
            request: &gemenr_core::ToolCallRequest,
            _context: &gemenr_core::PolicyContext,
        ) -> AuthorizationDecision {
            AuthorizationDecision::Prepared(PreparedToolCall {
                request: request.clone(),
                execution_context: ExecutionContext::new(SandboxKind::None),
            })
        }
    }

    #[async_trait]
    impl ToolExecutor for StaticInvoker {
        async fn invoke(
            &self,
            prepared: PreparedToolCall,
            _cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Ok(ToolInvokeResult {
                content: prepared.request.name,
                is_error: false,
            })
        }
    }

    #[test]
    fn allowlist_wrapper_denies_before_execution() {
        let view = allowlist_tool_invoker(Arc::new(StaticInvoker::new()), &["shell".to_string()]);
        let specs = view.list_specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "shell");

        let decision = view.authorize(
            &gemenr_core::ToolCallRequest {
                call_id: "1".to_string(),
                name: "fs.read".to_string(),
                arguments: json!({}),
            },
            &gemenr_core::PolicyContext::default(),
        );

        assert!(matches!(
            decision,
            AuthorizationDecision::Denied { reason }
                if reason == "tool `fs.read` is not available for this job"
        ));
    }
}
