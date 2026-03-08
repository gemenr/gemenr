//! Tool registration, policy evaluation, and execution primitives.

use std::collections::HashMap;

use async_trait::async_trait;
use gemenr_core::{ToolSpec, tool_invoker};
use tracing::{debug, warn};

pub mod handler;
pub mod policy;

pub use handler::{ExecContext, ToolCallSpec, ToolError, ToolHandler, ToolOutput};
pub use policy::{PolicyDecision, evaluate_policy};

/// Catalog of all registered tools.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCatalog {
    /// All registered tool specifications.
    pub tools: Vec<ToolSpec>,
}

/// Central registry and execution engine for tools.
pub struct ToolPlane {
    tools: HashMap<String, (ToolSpec, Box<dyn ToolHandler>)>,
}

impl ToolPlane {
    /// Create an empty tool plane.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
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
    ) -> Result<ToolOutput, ToolError> {
        let Some((_, handler)) = self.tools.get(&call.name) else {
            return Err(ToolError::NotFound(call.name.clone()));
        };

        debug!(call_id = %call.call_id, tool = %call.name, "invoking tool");

        match tokio::time::timeout(ctx.timeout, handler.execute(call.arguments.clone())).await {
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
    ) -> tool_invoker::PolicyDecision {
        let Some(spec) = self.lookup(name) else {
            return tool_invoker::PolicyDecision::Deny(format!("Tool '{}' not found", name));
        };

        let call = ToolCallSpec {
            call_id: String::new(),
            name: name.to_string(),
            arguments: arguments.clone(),
        };

        match evaluate_policy(spec, &call) {
            crate::policy::PolicyDecision::Allow => tool_invoker::PolicyDecision::Allow,
            crate::policy::PolicyDecision::NeedConfirmation(message) => {
                tool_invoker::PolicyDecision::NeedConfirmation(message)
            }
            crate::policy::PolicyDecision::Deny(reason) => {
                tool_invoker::PolicyDecision::Deny(reason)
            }
        }
    }

    async fn invoke(
        &self,
        call_id: &str,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<tool_invoker::ToolInvokeResult, tool_invoker::ToolInvokeError> {
        let call = ToolCallSpec {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
        };
        let ctx = ExecContext::default();

        match ToolPlane::invoke(self, &call, &ctx).await {
            Ok(output) => Ok(tool_invoker::ToolInvokeResult {
                content: output.content,
                is_error: false,
            }),
            Err(ToolError::NotFound(name)) => Err(tool_invoker::ToolInvokeError::NotFound(name)),
            Err(ToolError::Timeout(_)) => Err(tool_invoker::ToolInvokeError::Timeout),
            Err(error) => Err(tool_invoker::ToolInvokeError::Execution(error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use gemenr_core::RiskLevel;
    use serde_json::json;

    use super::{ExecContext, ToolCallSpec, ToolError, ToolHandler, ToolOutput, ToolPlane};

    struct StaticHandler {
        content: &'static str,
    }

    #[async_trait]
    impl ToolHandler for StaticHandler {
        async fn execute(&self, _args: serde_json::Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                content: self.content.to_string(),
            })
        }
    }

    struct SlowHandler;

    #[async_trait]
    impl ToolHandler for SlowHandler {
        async fn execute(&self, _args: serde_json::Value) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(ToolOutput {
                content: "done".to_string(),
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
            .invoke(&call("shell"), &ExecContext::default())
            .await
            .expect("tool should execute");

        assert_eq!(output.content, "hello");
    }

    #[tokio::test]
    async fn invoke_returns_not_found_for_missing_tool() {
        let plane = ToolPlane::new();

        let error = plane
            .invoke(&call("missing"), &ExecContext::default())
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
        };

        let error = plane
            .invoke(&call("shell"), &context)
            .await
            .expect_err("slow tool should time out");

        assert_eq!(error, ToolError::Timeout(Duration::from_millis(10)));
    }
}
