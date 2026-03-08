use crate::agent::dispatcher::ParsedToolCall;

/// Decision made after analyzing a model response for the current turn.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionDecision {
    /// Invoke one or more tools sequentially.
    InvokeTools(Vec<ParsedToolCall>),
    /// Send the model's final text response back to the caller.
    Respond(String),
    /// End the turn because there is no actionable output.
    CompleteTurn,
}

/// High-level phase executed inside one turn-loop iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    /// Rebuild provider-visible context from tape and in-memory history.
    BuildContext,
    /// Call the model and classify the response.
    CallModel,
    /// Execute an emitted batch of tool calls.
    ExecuteTools,
    /// Finish the turn successfully.
    Complete,
    /// Finish the turn with an error.
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ModelStepOutcome {
    Complete(String),
    InvokeTools(Vec<ParsedToolCall>),
}

/// Stateless controller that chooses the next runtime action.
#[derive(Debug, Default, Clone, Copy)]
pub struct TurnController;

impl TurnController {
    /// Determine the next action based on parsed text and tool calls.
    #[must_use]
    pub fn next_action(
        &self,
        text: Option<String>,
        tool_calls: Vec<ParsedToolCall>,
    ) -> ActionDecision {
        if !tool_calls.is_empty() {
            ActionDecision::InvokeTools(tool_calls)
        } else if let Some(text) = text {
            if text.trim().is_empty() {
                ActionDecision::CompleteTurn
            } else {
                ActionDecision::Respond(text)
            }
        } else {
            ActionDecision::CompleteTurn
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionDecision, TurnController};
    use crate::agent::ParsedToolCall;

    fn sample_tool_call() -> ParsedToolCall {
        ParsedToolCall {
            id: "call-1".to_string(),
            name: "shell".to_string(),
            arguments: serde_json::json!({"command": "pwd"}),
        }
    }

    #[test]
    fn next_action_returns_invoke_tools_when_calls_exist() {
        let controller = TurnController;
        let tool_calls = vec![sample_tool_call()];

        let decision = controller.next_action(None, tool_calls.clone());

        assert_eq!(decision, ActionDecision::InvokeTools(tool_calls));
    }

    #[test]
    fn next_action_returns_respond_for_plain_text() {
        let controller = TurnController;

        let decision = controller.next_action(Some("done".to_string()), Vec::new());

        assert_eq!(decision, ActionDecision::Respond("done".to_string()));
    }

    #[test]
    fn next_action_returns_complete_turn_for_empty_text() {
        let controller = TurnController;

        let decision = controller.next_action(Some(String::new()), Vec::new());

        assert_eq!(decision, ActionDecision::CompleteTurn);
    }

    #[test]
    fn next_action_returns_complete_turn_for_blank_text() {
        let controller = TurnController;

        let decision = controller.next_action(Some("   \n\t".to_string()), Vec::new());

        assert_eq!(decision, ActionDecision::CompleteTurn);
    }

    #[test]
    fn next_action_prioritizes_tool_calls_over_text() {
        let controller = TurnController;
        let tool_calls = vec![sample_tool_call()];

        let decision = controller.next_action(Some("ignore me".to_string()), tool_calls.clone());

        assert_eq!(decision, ActionDecision::InvokeTools(tool_calls));
    }
}
