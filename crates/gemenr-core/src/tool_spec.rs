use serde::{Deserialize, Serialize};

/// Specification of a tool that can be registered and invoked by the agent.
///
/// This type lives in `gemenr-core` because model providers need to reference
/// it when converting tool definitions, while `gemenr-core` must remain
/// independent from the future `gemenr-tools` crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Unique name of the tool (for example, `shell` or `fs.read`).
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool input parameters.
    pub input_schema: serde_json::Value,
    /// Risk level used for policy evaluation.
    pub risk_level: RiskLevel,
}

/// Risk level of a tool, used for policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    /// Safe to execute without user confirmation.
    Low,
    /// Requires user confirmation before execution.
    Medium,
    /// Requires user confirmation and may be destructive.
    High,
}

#[cfg(test)]
mod tests {
    use super::{RiskLevel, ToolSpec};

    #[test]
    fn tool_spec_round_trips_through_json() {
        let spec = ToolSpec {
            name: "shell".to_string(),
            description: "Execute a shell command".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
            risk_level: RiskLevel::High,
        };

        let json = serde_json::to_string(&spec).expect("tool spec should serialize");
        let decoded: ToolSpec = serde_json::from_str(&json).expect("tool spec should deserialize");

        assert_eq!(decoded, spec);
    }

    #[test]
    fn risk_levels_serialize_to_lowercase_strings() {
        let low = serde_json::to_string(&RiskLevel::Low).expect("low should serialize");
        let medium = serde_json::to_string(&RiskLevel::Medium).expect("medium should serialize");
        let high = serde_json::to_string(&RiskLevel::High).expect("high should serialize");

        assert_eq!(low, r#""low""#);
        assert_eq!(medium, r#""medium""#);
        assert_eq!(high, r#""high""#);
    }
}
