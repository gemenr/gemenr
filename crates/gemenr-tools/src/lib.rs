//! Tool registration, policy evaluation, and execution primitives.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::{ToolSpec, tool_invoker};
use tracing::{debug, warn};

pub mod builtin;
pub mod handler;
pub mod policy;

pub use handler::{ExecContext, ToolCallSpec, ToolError, ToolHandler, ToolOutput};
pub use policy::{PolicyEvaluator, PolicyRule, PolicyScope, RuleBasedPolicyEvaluator};

/// Catalog of all registered tools.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCatalog {
    /// All registered tool specifications.
    pub tools: Vec<ToolSpec>,
}

/// Central registry and execution engine for tools.
pub struct ToolPlane {
    tools: HashMap<String, (ToolSpec, Box<dyn ToolHandler>)>,
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
            policy_evaluator,
        }
    }

    /// Replace the policy evaluator used for subsequent checks.
    pub fn set_policy_evaluator(&mut self, policy_evaluator: Arc<dyn PolicyEvaluator>) {
        self.policy_evaluator = policy_evaluator;
    }

    /// Register a tool with its specification and handler.
    pub fn register(&mut self, spec: ToolSpec, handler: Box<dyn ToolHandler>) {
        debug!(tool = %spec.name, "registering tool");
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
        let mut tools = self
            .tools
            .values()
            .map(|(spec, _)| spec.clone())
            .collect::<Vec<_>>();
        tools.sort_by(|left, right| left.name.cmp(&right.name));

        ToolCatalog { tools }
    }

    /// Invoke a tool by call spec and execution context.
    pub async fn invoke(
        &self,
        call: &ToolCallSpec,
        ctx: &ExecContext,
        cancelled: Arc<AtomicBool>,
    ) -> Result<ToolOutput, ToolError> {
        let Some((spec, handler)) = self.tools.get(&call.name) else {
            return Err(ToolError::NotFound(call.name.clone()));
        };

        debug!(call_id = %call.call_id, tool = %call.name, "invoking tool");

        let mut execution_context = ctx.clone();
        if execution_context.execution_policy.is_none() {
            execution_context.execution_policy = Some(self.policy_evaluator.evaluate(
                &execution_context.policy_context,
                spec,
                call,
            ));
        }

        let execution = async {
            let tool_future = handler.execute(&execution_context, call.arguments.clone());
            tokio::pin!(tool_future);

            loop {
                tokio::select! {
                    result = &mut tool_future => return result,
                    _ = tokio::time::sleep(Duration::from_millis(25)) => {
                        if cancelled.load(Ordering::Relaxed) {
                            warn!(call_id = %call.call_id, tool = %call.name, "tool invocation cancelled");
                            return Err(ToolError::Cancelled);
                        }
                    }
                }
            }
        };

        match tokio::time::timeout(ctx.timeout, execution).await {
            Ok(result) => result,
            Err(_) => {
                warn!(call_id = %call.call_id, tool = %call.name, timeout = ?ctx.timeout, "tool invocation timed out");
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

#[async_trait]
impl tool_invoker::ToolInvoker for ToolPlane {
    fn lookup(&self, name: &str) -> Option<&ToolSpec> {
        ToolPlane::lookup(self, name)
    }

    fn list_specs(&self) -> Vec<ToolSpec> {
        self.list().tools
    }

    fn check_policy(
        &self,
        name: &str,
        arguments: &serde_json::Value,
        context: &tool_invoker::PolicyContext,
    ) -> tool_invoker::ExecutionPolicy {
        let Some(spec) = self.lookup(name) else {
            return tool_invoker::ExecutionPolicy::Deny {
                reason: format!("Tool '{}' not found", name),
            };
        };

        let call = ToolCallSpec {
            call_id: String::new(),
            name: name.to_string(),
            arguments: arguments.clone(),
        };

        self.policy_evaluator.evaluate(context, spec, &call)
    }

    async fn invoke(
        &self,
        call_id: &str,
        name: &str,
        arguments: serde_json::Value,
        cancelled: Arc<AtomicBool>,
    ) -> Result<tool_invoker::ToolInvokeResult, tool_invoker::ToolInvokeError> {
        let call = ToolCallSpec {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
        };
        let ctx = ExecContext {
            execution_policy: Some(self.check_policy(
                name,
                &call.arguments,
                &tool_invoker::PolicyContext::default(),
            )),
            ..ExecContext::default()
        };

        match ToolPlane::invoke(self, &call, &ctx, cancelled).await {
            Ok(output) => Ok(tool_invoker::ToolInvokeResult {
                content: output.content,
                is_error: false,
            }),
            Err(ToolError::NotFound(name)) => Err(tool_invoker::ToolInvokeError::NotFound(name)),
            Err(ToolError::Timeout(_)) => Err(tool_invoker::ToolInvokeError::Timeout),
            Err(ToolError::Cancelled) => Err(tool_invoker::ToolInvokeError::Cancelled),
            Err(error) => Err(tool_invoker::ToolInvokeError::Execution(error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use async_trait::async_trait;
    use gemenr_core::{ExecutionPolicy, PolicyContext, RiskLevel, SandboxKind};
    use serde_json::json;

    use super::{
        ExecContext, PolicyRule, PolicyScope, RuleBasedPolicyEvaluator, ToolCallSpec, ToolError,
        ToolHandler, ToolOutput, ToolPlane,
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

    #[tokio::test]
    async fn invoke_executes_registered_tool() {
        let mut plane = ToolPlane::new();
        plane.register(spec("shell"), Box::new(StaticHandler { content: "hello" }));

        let output = plane
            .invoke(
                &call("shell"),
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
            .invoke(&call("shell"), &context, Arc::new(AtomicBool::new(false)))
            .await
            .expect("tool should receive context");

        assert_eq!(output.content, context.working_dir.display().to_string());
    }

    #[tokio::test]
    async fn invoke_returns_not_found_for_missing_tool() {
        let plane = ToolPlane::new();

        let error = plane
            .invoke(
                &call("missing"),
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
            .invoke(&call("shell"), &context, Arc::new(AtomicBool::new(false)))
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
            .invoke(&call("shell"), &ExecContext::default(), cancelled)
            .await
            .expect_err("cancelled tool should fail");

        assert_eq!(error, ToolError::Cancelled);
    }

    #[test]
    fn check_policy_returns_execution_plan_from_evaluator() {
        let mut plane = ToolPlane::with_policy_evaluator(Arc::new(RuleBasedPolicyEvaluator {
            rules: vec![PolicyRule {
                scope: PolicyScope::Conversation("conv-1".to_string()),
                tool_name: "shell".to_string(),
                effect: gemenr_core::PolicyEffect::NeedConfirmation,
                sandbox: SandboxKind::Seatbelt,
            }],
        }));
        plane.register(spec("shell"), Box::new(StaticHandler { content: "ok" }));

        let plan = <ToolPlane as gemenr_core::ToolInvoker>::check_policy(
            &plane,
            "shell",
            &json!({"command": "pwd"}),
            &PolicyContext {
                organization_id: None,
                workspace_id: None,
                conversation_id: Some("conv-1".to_string()),
            },
        );

        assert_eq!(
            plan,
            ExecutionPolicy::NeedConfirmation {
                message: "Tool 'shell' requires confirmation".to_string(),
                sandbox: SandboxKind::Seatbelt,
            }
        );
    }
}
