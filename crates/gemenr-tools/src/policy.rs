use gemenr_core::{RiskLevel, ToolSpec};

use crate::handler::ToolCallSpec;

/// Result of policy evaluation for a tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Tool execution is allowed without confirmation.
    Allow,
    /// Tool execution requires user confirmation with a reason.
    NeedConfirmation(String),
    /// Tool execution is denied with a reason.
    Deny(String),
}

/// Evaluate the execution policy for a tool invocation.
///
/// Phase 1 keeps policy simple: low-risk tools are auto-approved, while
/// medium- and high-risk tools require explicit confirmation.
#[must_use]
pub fn evaluate_policy(spec: &ToolSpec, _call: &ToolCallSpec) -> PolicyDecision {
    match spec.risk_level {
        RiskLevel::Low => PolicyDecision::Allow,
        RiskLevel::Medium => PolicyDecision::NeedConfirmation(format!(
            "Tool '{}' has medium risk level. Allow execution?",
            spec.name
        )),
        RiskLevel::High => PolicyDecision::NeedConfirmation(format!(
            "Tool '{}' has HIGH risk level. Allow execution?",
            spec.name
        )),
    }
}

#[cfg(test)]
mod tests {
    use gemenr_core::{RiskLevel, ToolSpec};
    use serde_json::json;

    use super::{PolicyDecision, evaluate_policy};
    use crate::handler::ToolCallSpec;

    fn call() -> ToolCallSpec {
        ToolCallSpec {
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
    fn low_risk_tools_are_allowed() {
        let decision = evaluate_policy(&spec(RiskLevel::Low), &call());

        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[test]
    fn medium_risk_tools_need_confirmation() {
        let decision = evaluate_policy(&spec(RiskLevel::Medium), &call());

        assert_eq!(
            decision,
            PolicyDecision::NeedConfirmation(
                "Tool 'shell' has medium risk level. Allow execution?".to_string()
            )
        );
    }

    #[test]
    fn high_risk_tools_need_confirmation() {
        let decision = evaluate_policy(&spec(RiskLevel::High), &call());

        assert_eq!(
            decision,
            PolicyDecision::NeedConfirmation(
                "Tool 'shell' has HIGH risk level. Allow execution?".to_string()
            )
        );
    }
}
