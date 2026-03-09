use std::sync::Arc;

use gemenr_core::{
    ExecutionPolicy, PolicyContext, RiskLevel, SandboxKind, ToolCallRequest, ToolSpec,
    config::{PolicyConfig, PolicyEffect},
};

/// Evaluates the effective execution policy for one tool invocation.
pub trait PolicyEvaluator: Send + Sync {
    /// Resolve the final execution policy for the given context and tool call.
    fn evaluate(
        &self,
        ctx: &PolicyContext,
        spec: &ToolSpec,
        call: &ToolCallRequest,
    ) -> ExecutionPolicy;
}

/// One configured policy rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRule {
    /// Scope matched by this rule.
    pub scope: PolicyScope,
    /// Tool name matched by this rule.
    pub tool_name: String,
    /// Result effect applied when the rule matches.
    pub effect: PolicyEffect,
    /// Sandbox backend selected by the rule.
    pub sandbox: SandboxKind,
}

/// Scope selector for one policy rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyScope {
    /// Organization-level scope.
    Organization(String),
    /// Workspace-level scope.
    Workspace(String),
    /// Conversation-level scope.
    Conversation(String),
}

/// Rule-based policy evaluator with deterministic precedence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuleBasedPolicyEvaluator {
    /// Flattened ordered rule set.
    pub rules: Vec<PolicyRule>,
}

impl RuleBasedPolicyEvaluator {
    /// Create an evaluator from already-flattened rules.
    #[must_use]
    pub fn new(rules: Vec<PolicyRule>) -> Self {
        Self { rules }
    }

    /// Build an evaluator from configuration.
    #[must_use]
    pub fn from_config(config: &PolicyConfig) -> Self {
        let organizations = config.organizations.iter().flat_map(|scope| {
            scope.rules.iter().map(|rule| PolicyRule {
                scope: PolicyScope::Organization(scope.id.clone()),
                tool_name: rule.tool.clone(),
                effect: rule.effect,
                sandbox: rule.sandbox,
            })
        });
        let workspaces = config.workspaces.iter().flat_map(|scope| {
            scope.rules.iter().map(|rule| PolicyRule {
                scope: PolicyScope::Workspace(scope.id.clone()),
                tool_name: rule.tool.clone(),
                effect: rule.effect,
                sandbox: rule.sandbox,
            })
        });
        let conversations = config.conversations.iter().flat_map(|scope| {
            scope.rules.iter().map(|rule| PolicyRule {
                scope: PolicyScope::Conversation(scope.id.clone()),
                tool_name: rule.tool.clone(),
                effect: rule.effect,
                sandbox: rule.sandbox,
            })
        });

        Self::new(
            organizations
                .chain(workspaces)
                .chain(conversations)
                .collect(),
        )
    }
}

impl PolicyEvaluator for RuleBasedPolicyEvaluator {
    fn evaluate(
        &self,
        ctx: &PolicyContext,
        spec: &ToolSpec,
        call: &ToolCallRequest,
    ) -> ExecutionPolicy {
        if let Some(policy) = parameter_sensitive_override(spec, call) {
            return policy;
        }

        for level in [
            ScopeLevel::Conversation,
            ScopeLevel::Workspace,
            ScopeLevel::Organization,
        ] {
            let winning_rule = self
                .rules
                .iter()
                .filter(|rule| rule.tool_name == spec.name)
                .filter(|rule| scope_level(rule) == level)
                .filter(|rule| scope_matches(ctx, &rule.scope))
                .max_by_key(|rule| effect_rank(rule.effect));

            if let Some(rule) = winning_rule {
                return execution_policy_from_rule(rule);
            }
        }

        phase_one_default(spec)
    }
}

fn parameter_sensitive_override(
    spec: &ToolSpec,
    call: &ToolCallRequest,
) -> Option<ExecutionPolicy> {
    if spec.name == "shell" {
        let command = call
            .arguments
            .get("command")
            .and_then(serde_json::Value::as_str)?;
        let normalized = command.to_ascii_lowercase();
        if ["rm -rf", "sudo ", "mkfs", "shutdown", "reboot"]
            .iter()
            .any(|needle| normalized.contains(needle))
        {
            return Some(ExecutionPolicy::Deny {
                reason: format!(
                    "Tool '{}' is denied by policy because command arguments are high risk",
                    spec.name
                ),
            });
        }
    }

    if spec.name == "fs.write" {
        let path = call
            .arguments
            .get("path")
            .and_then(serde_json::Value::as_str)?;
        if [".ssh/", "/etc/", ".git/"]
            .iter()
            .any(|needle| path.contains(needle))
        {
            return Some(ExecutionPolicy::NeedConfirmation {
                message: format!(
                    "Tool '{}' targets a sensitive path and requires confirmation",
                    spec.name
                ),
                sandbox: SandboxKind::None,
            });
        }
    }

    None
}

/// Shared pointer to a policy evaluator implementation.
pub type SharedPolicyEvaluator = Arc<dyn PolicyEvaluator>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ScopeLevel {
    Organization,
    Workspace,
    Conversation,
}

fn scope_level(rule: &PolicyRule) -> ScopeLevel {
    match rule.scope {
        PolicyScope::Organization(_) => ScopeLevel::Organization,
        PolicyScope::Workspace(_) => ScopeLevel::Workspace,
        PolicyScope::Conversation(_) => ScopeLevel::Conversation,
    }
}

fn scope_matches(ctx: &PolicyContext, scope: &PolicyScope) -> bool {
    match scope {
        PolicyScope::Organization(id) => ctx.organization_id.as_deref() == Some(id.as_str()),
        PolicyScope::Workspace(id) => ctx.workspace_id.as_deref() == Some(id.as_str()),
        PolicyScope::Conversation(id) => ctx.conversation_id.as_deref() == Some(id.as_str()),
    }
}

fn effect_rank(effect: PolicyEffect) -> u8 {
    match effect {
        PolicyEffect::Allow => 1,
        PolicyEffect::NeedConfirmation => 2,
        PolicyEffect::Deny => 3,
    }
}

fn execution_policy_from_rule(rule: &PolicyRule) -> ExecutionPolicy {
    match rule.effect {
        PolicyEffect::Allow => ExecutionPolicy::Allow {
            sandbox: rule.sandbox,
        },
        PolicyEffect::NeedConfirmation => ExecutionPolicy::NeedConfirmation {
            message: format!("Tool '{}' requires confirmation", rule.tool_name),
            sandbox: rule.sandbox,
        },
        PolicyEffect::Deny => ExecutionPolicy::Deny {
            reason: format!("Tool '{}' is denied by policy", rule.tool_name),
        },
    }
}

fn phase_one_default(spec: &ToolSpec) -> ExecutionPolicy {
    match spec.risk_level {
        RiskLevel::Low => ExecutionPolicy::Allow {
            sandbox: SandboxKind::None,
        },
        RiskLevel::Medium => ExecutionPolicy::NeedConfirmation {
            message: format!(
                "Tool '{}' has medium risk level. Allow execution?",
                spec.name
            ),
            sandbox: SandboxKind::None,
        },
        RiskLevel::High => ExecutionPolicy::NeedConfirmation {
            message: format!("Tool '{}' has HIGH risk level. Allow execution?", spec.name),
            sandbox: SandboxKind::None,
        },
    }
}

#[cfg(test)]
mod tests {
    use gemenr_core::{
        ExecutionPolicy, PolicyContext, RiskLevel, SandboxKind, ToolCallRequest, ToolSpec,
        config::PolicyEffect,
    };
    use serde_json::json;

    use super::{PolicyEvaluator, PolicyRule, PolicyScope, RuleBasedPolicyEvaluator};

    fn call() -> ToolCallRequest {
        ToolCallRequest {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            arguments: json!({"command": "pwd"}),
        }
    }

    fn spec(risk_level: RiskLevel) -> ToolSpec {
        ToolSpec {
            name: "shell".to_string(),
            description: "Execute a shell command".to_string(),
            input_schema: json!({"type": "object"}),
            risk_level,
        }
    }

    #[test]
    fn no_rules_keep_phase_one_defaults() {
        let evaluator = RuleBasedPolicyEvaluator::default();
        let ctx = PolicyContext::default();

        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::Low), &call()),
            ExecutionPolicy::Allow {
                sandbox: SandboxKind::None,
            }
        );
        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::Medium), &call()),
            ExecutionPolicy::NeedConfirmation {
                message: "Tool 'shell' has medium risk level. Allow execution?".to_string(),
                sandbox: SandboxKind::None,
            }
        );
        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::High), &call()),
            ExecutionPolicy::NeedConfirmation {
                message: "Tool 'shell' has HIGH risk level. Allow execution?".to_string(),
                sandbox: SandboxKind::None,
            }
        );
    }

    #[test]
    fn conversation_rule_overrides_workspace_rule() {
        let evaluator = RuleBasedPolicyEvaluator::new(vec![
            PolicyRule {
                scope: PolicyScope::Workspace("ws-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::Deny,
                sandbox: SandboxKind::None,
            },
            PolicyRule {
                scope: PolicyScope::Conversation("conv-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::Allow,
                sandbox: SandboxKind::Seatbelt,
            },
        ]);
        let ctx = PolicyContext {
            organization_id: None,
            workspace_id: Some("ws-1".to_string()),
            conversation_id: Some("conv-1".to_string()),
        };

        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::High), &call()),
            ExecutionPolicy::Allow {
                sandbox: SandboxKind::Seatbelt,
            }
        );
    }

    #[test]
    fn workspace_rule_overrides_organization_rule() {
        let evaluator = RuleBasedPolicyEvaluator::new(vec![
            PolicyRule {
                scope: PolicyScope::Organization("org-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::Deny,
                sandbox: SandboxKind::None,
            },
            PolicyRule {
                scope: PolicyScope::Workspace("ws-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::NeedConfirmation,
                sandbox: SandboxKind::Landlock,
            },
        ]);
        let ctx = PolicyContext {
            organization_id: Some("org-1".to_string()),
            workspace_id: Some("ws-1".to_string()),
            conversation_id: None,
        };

        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::Low), &call()),
            ExecutionPolicy::NeedConfirmation {
                message: "Tool 'shell' requires confirmation".to_string(),
                sandbox: SandboxKind::Landlock,
            }
        );
    }

    #[test]
    fn same_scope_conflicts_use_deny_confirm_allow_priority() {
        let evaluator = RuleBasedPolicyEvaluator::new(vec![
            PolicyRule {
                scope: PolicyScope::Workspace("ws-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::Allow,
                sandbox: SandboxKind::Seatbelt,
            },
            PolicyRule {
                scope: PolicyScope::Workspace("ws-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::NeedConfirmation,
                sandbox: SandboxKind::Landlock,
            },
            PolicyRule {
                scope: PolicyScope::Workspace("ws-1".to_string()),
                tool_name: "shell".to_string(),
                effect: PolicyEffect::Deny,
                sandbox: SandboxKind::None,
            },
        ]);
        let ctx = PolicyContext {
            organization_id: None,
            workspace_id: Some("ws-1".to_string()),
            conversation_id: None,
        };

        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::Low), &call()),
            ExecutionPolicy::Deny {
                reason: "Tool 'shell' is denied by policy".to_string(),
            }
        );
    }

    #[test]
    fn high_risk_shell_arguments_are_denied() {
        let evaluator = RuleBasedPolicyEvaluator::default();
        let call = ToolCallRequest {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            arguments: json!({"command": "rm -rf /tmp/demo"}),
        };

        assert_eq!(
            evaluator.evaluate(&PolicyContext::default(), &spec(RiskLevel::Medium), &call),
            ExecutionPolicy::Deny {
                reason: "Tool 'shell' is denied by policy because command arguments are high risk"
                    .to_string(),
            }
        );
    }

    #[test]
    fn sensitive_fs_write_paths_require_confirmation() {
        let evaluator = RuleBasedPolicyEvaluator::default();
        let call = ToolCallRequest {
            call_id: "call-1".to_string(),
            name: "fs.write".to_string(),
            arguments: json!({"path": "/etc/hosts", "content": "127.0.0.1 example"}),
        };
        let spec = ToolSpec {
            name: "fs.write".to_string(),
            description: "Write a file".to_string(),
            input_schema: json!({"type": "object"}),
            risk_level: RiskLevel::Low,
        };

        assert_eq!(
            evaluator.evaluate(&PolicyContext::default(), &spec, &call),
            ExecutionPolicy::NeedConfirmation {
                message: "Tool 'fs.write' targets a sensitive path and requires confirmation"
                    .to_string(),
                sandbox: SandboxKind::None,
            }
        );
    }

    #[test]
    fn sandbox_plan_is_preserved_in_execution_policy() {
        let evaluator = RuleBasedPolicyEvaluator::new(vec![PolicyRule {
            scope: PolicyScope::Organization("org-1".to_string()),
            tool_name: "shell".to_string(),
            effect: PolicyEffect::Allow,
            sandbox: SandboxKind::Seatbelt,
        }]);
        let ctx = PolicyContext {
            organization_id: Some("org-1".to_string()),
            workspace_id: None,
            conversation_id: None,
        };

        assert_eq!(
            evaluator.evaluate(&ctx, &spec(RiskLevel::Low), &call()),
            ExecutionPolicy::Allow {
                sandbox: SandboxKind::Seatbelt,
            }
        );
    }
}
