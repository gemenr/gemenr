use std::fmt::Write;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::message::ChatMessage;
use crate::model::{ChatResponse, ToolCall};
use crate::tool_spec::ToolSpec;

/// A parsed tool call extracted from a model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    /// Tool call identifier from the provider, or a generated ID for XML mode.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Parsed JSON arguments for the tool.
    pub arguments: serde_json::Value,
}

/// Result of a tool execution, ready for formatting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    /// Identifier of the original tool call.
    pub call_id: String,
    /// Name of the executed tool.
    pub name: String,
    /// Output returned by the tool.
    pub content: String,
    /// Whether the tool execution failed.
    pub is_error: bool,
}

/// A conversation message kept in agent history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationMessage {
    /// A regular chat message.
    Chat(ChatMessage),
    /// An assistant response that contains structured tool calls.
    AssistantToolCalls {
        /// Optional assistant text emitted alongside the tool calls.
        text: Option<String>,
        /// Structured tool calls emitted by the assistant.
        tool_calls: Vec<ToolCall>,
    },
    /// Tool execution results waiting to be sent back to the provider.
    ToolResults(Vec<ToolExecutionResult>),
}

/// Closed-world tool protocol selection for the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedToolDispatcher {
    /// Provider-native structured tool calling.
    Native,
    /// XML-tagged tool calling embedded in assistant text.
    Xml,
}

/// Native structured tool-calling mode.
#[allow(non_upper_case_globals)]
pub const NativeToolDispatcher: SelectedToolDispatcher = SelectedToolDispatcher::Native;

/// XML-tagged tool-calling mode.
#[allow(non_upper_case_globals)]
pub const XmlToolDispatcher: SelectedToolDispatcher = SelectedToolDispatcher::Xml;

impl SelectedToolDispatcher {
    /// Extract assistant text and parsed tool calls from a provider response.
    #[must_use]
    pub fn parse_response(&self, response: &ChatResponse) -> (Option<String>, Vec<ParsedToolCall>) {
        match self {
            Self::Native => native_parse_response(response),
            Self::Xml => xml_parse_response(response),
        }
    }

    /// Format tool execution results into an internal history entry.
    #[must_use]
    pub fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage {
        match self {
            Self::Native => ConversationMessage::ToolResults(results.to_vec()),
            Self::Xml => {
                ConversationMessage::Chat(ChatMessage::user(render_xml_tool_results(results)))
            }
        }
    }

    /// Build system-prompt instructions for the active tool-calling protocol.
    #[must_use]
    pub fn prompt_instructions(&self, tools: &[ToolSpec]) -> String {
        match self {
            Self::Native => String::new(),
            Self::Xml => xml_prompt_instructions(tools),
        }
    }

    /// Project internal history into provider-consumable chat messages.
    #[must_use]
    pub fn to_provider_messages(&self, history: &[ConversationMessage]) -> Vec<ChatMessage> {
        match self {
            Self::Native => native_to_provider_messages(history),
            Self::Xml => xml_to_provider_messages(history),
        }
    }

    /// Return whether provider-native tool definitions should be sent via API.
    #[must_use]
    pub fn should_send_tool_specs(&self) -> bool {
        matches!(self, Self::Native)
    }
}

fn native_parse_response(response: &ChatResponse) -> (Option<String>, Vec<ParsedToolCall>) {
    let calls = response
        .tool_calls
        .iter()
        .map(|tool_call| ParsedToolCall {
            id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            arguments: serde_json::from_str(&tool_call.arguments)
                .unwrap_or(serde_json::Value::Null),
        })
        .collect();

    (response.text.clone(), calls)
}

fn native_to_provider_messages(history: &[ConversationMessage]) -> Vec<ChatMessage> {
    history
        .iter()
        .flat_map(|message| match message {
            ConversationMessage::Chat(chat) => vec![chat.clone()],
            ConversationMessage::AssistantToolCalls { text, tool_calls } => vec![
                ChatMessage::assistant(text.clone().unwrap_or_default()).with_metadata(
                    "tool_calls",
                    serde_json::to_string(tool_calls)
                        .expect("tool calls should serialize to metadata"),
                ),
            ],
            ConversationMessage::ToolResults(results) => results
                .iter()
                .map(|result| {
                    ChatMessage::user(result.content.clone())
                        .with_metadata("tool_result_for", result.call_id.clone())
                        .with_metadata("tool_name", result.name.clone())
                        .with_metadata("is_error", result.is_error.to_string())
                })
                .collect(),
        })
        .collect()
}

fn xml_parse_response(response: &ChatResponse) -> (Option<String>, Vec<ParsedToolCall>) {
    let text = response.text.as_deref().unwrap_or_default();
    let regex = tool_call_regex();
    let calls = regex
        .captures_iter(text)
        .filter_map(|captures| captures.get(1))
        .filter_map(|payload| {
            serde_json::from_str::<serde_json::Value>(payload.as_str().trim()).ok()
        })
        .filter_map(|parsed| {
            let name = parsed.get("name").and_then(serde_json::Value::as_str)?;
            if name.is_empty() {
                return None;
            }

            Some(ParsedToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name: name.to_string(),
                arguments: parsed
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            })
        })
        .collect();

    let cleaned = regex.replace_all(text, "");
    let cleaned = cleaned.trim();
    let cleaned = if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    };

    (cleaned, calls)
}

fn xml_prompt_instructions(tools: &[ToolSpec]) -> String {
    let mut instructions = String::from(
        "You have access to the following tools. To call a tool, output a <tool_call> tag containing a JSON object with \"name\" and \"arguments\".\n\n",
    );
    instructions.push_str(
        "Example:\n<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value\"}}</tool_call>\n\n",
    );
    instructions.push_str("Available tools:\n\n");

    for tool in tools {
        let _ = writeln!(instructions, "- **{}**: {}", tool.name, tool.description);
        let _ = writeln!(instructions, "  Parameters: {}", tool.input_schema);
        instructions.push('\n');
    }

    instructions
}

fn xml_to_provider_messages(history: &[ConversationMessage]) -> Vec<ChatMessage> {
    history
        .iter()
        .flat_map(|message| match message {
            ConversationMessage::Chat(chat) => vec![chat.clone()],
            ConversationMessage::AssistantToolCalls { text, .. } => {
                vec![ChatMessage::assistant(text.clone().unwrap_or_default())]
            }
            ConversationMessage::ToolResults(results) => {
                vec![ChatMessage::user(render_xml_tool_results(results))]
            }
        })
        .collect()
}

fn tool_call_regex() -> &'static Regex {
    static TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();
    TOOL_CALL_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").expect("tool call regex should compile")
    })
}

fn render_xml_tool_results(results: &[ToolExecutionResult]) -> String {
    let mut content = String::from("[Tool results]");

    for result in results {
        let status = if result.is_error { "error" } else { "ok" };
        let _ = write!(
            content,
            "\n<tool_result name=\"{}\" status=\"{}\">{}</tool_result>",
            result.name, status, result.content
        );
    }

    content
}

#[cfg(test)]
mod tests {
    use super::{
        ConversationMessage, NativeToolDispatcher, SelectedToolDispatcher, ToolExecutionResult,
        XmlToolDispatcher,
    };
    use crate::message::{ChatMessage, ChatRole};
    use crate::model::{ChatResponse, ToolCall};
    use crate::tool_spec::{RiskLevel, ToolSpec};

    fn sample_tool_spec() -> ToolSpec {
        ToolSpec {
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
        }
    }

    fn sample_tool_result() -> ToolExecutionResult {
        ToolExecutionResult {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            content: "ok".to_string(),
            is_error: false,
        }
    }

    #[test]
    fn native_parse_response_extracts_tool_calls() {
        let dispatcher = NativeToolDispatcher;
        let response = ChatResponse {
            text: None,
            tool_calls: vec![ToolCall {
                id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: r#"{"command":"ls"}"#.to_string(),
            }],
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(text, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments, serde_json::json!({"command": "ls"}));
    }

    #[test]
    fn native_parse_response_without_tool_calls() {
        let dispatcher = NativeToolDispatcher;
        let response = ChatResponse {
            text: Some("hello".to_string()),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(text.as_deref(), Some("hello"));
        assert!(calls.is_empty());
    }

    #[test]
    fn native_parse_response_with_text_and_tool_calls() {
        let dispatcher = NativeToolDispatcher;
        let response = ChatResponse {
            text: Some("working".to_string()),
            tool_calls: vec![ToolCall {
                id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: "not-json".to_string(),
            }],
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(text.as_deref(), Some("working"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, serde_json::Value::Null);
    }

    #[test]
    fn native_format_results_returns_structured_results() {
        let dispatcher = NativeToolDispatcher;
        let message = dispatcher.format_results(&[sample_tool_result()]);

        assert_eq!(
            message,
            ConversationMessage::ToolResults(vec![sample_tool_result()])
        );
    }

    #[test]
    fn native_prompt_instructions_is_empty() {
        let dispatcher = NativeToolDispatcher;
        assert!(
            dispatcher
                .prompt_instructions(&[sample_tool_spec()])
                .is_empty()
        );
    }

    #[test]
    fn native_should_send_tool_specs_is_true() {
        let dispatcher = NativeToolDispatcher;
        assert!(dispatcher.should_send_tool_specs());
    }

    #[test]
    fn native_to_provider_messages_passes_chat_messages_through() {
        let dispatcher = NativeToolDispatcher;
        let history = vec![ConversationMessage::Chat(ChatMessage::assistant("hello"))];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages, vec![ChatMessage::assistant("hello")]);
    }

    #[test]
    fn native_to_provider_messages_preserves_tool_metadata() {
        let dispatcher = NativeToolDispatcher;
        let history = vec![
            ConversationMessage::AssistantToolCalls {
                text: Some("checking".to_string()),
                tool_calls: vec![ToolCall {
                    id: "call-1".to_string(),
                    name: "shell".to_string(),
                    arguments: r#"{"command":"pwd"}"#.to_string(),
                }],
            },
            ConversationMessage::ToolResults(vec![sample_tool_result()]),
        ];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::Assistant);
        assert_eq!(messages[0].content, "checking");
        assert_eq!(
            messages[0].metadata.get("tool_calls").map(String::as_str),
            Some(r#"[{"id":"call-1","name":"shell","arguments":"{\"command\":\"pwd\"}"}]"#)
        );
        assert_eq!(messages[1].role, ChatRole::User);
        assert_eq!(messages[1].content, "ok");
        assert_eq!(
            messages[1]
                .metadata
                .get("tool_result_for")
                .map(String::as_str),
            Some("call-1")
        );
        assert_eq!(
            messages[1].metadata.get("tool_name").map(String::as_str),
            Some("shell")
        );
        assert_eq!(
            messages[1].metadata.get("is_error").map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn xml_parse_response_extracts_tool_call() {
        let dispatcher = XmlToolDispatcher;
        let response = ChatResponse {
            text: Some(
                "Checking\n<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>"
                    .to_string(),
            ),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(text.as_deref(), Some("Checking"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments, serde_json::json!({"command": "ls"}));
        assert!(!calls[0].id.is_empty());
    }

    #[test]
    fn xml_parse_response_extracts_multiple_tool_calls() {
        let dispatcher = XmlToolDispatcher;
        let response = ChatResponse {
            text: Some(
                concat!(
                    "Before\n",
                    "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>\n",
                    "Middle\n",
                    "<tool_call>{\"name\":\"fs.read\",\"arguments\":{\"path\":\"a.txt\"}}</tool_call>"
                )
                .to_string(),
            ),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[1].name, "fs.read");
        assert_eq!(text.as_deref(), Some("Before\n\nMiddle"));
    }

    #[test]
    fn xml_parse_response_ignores_malformed_payloads_without_panicking() {
        let dispatcher = XmlToolDispatcher;
        let response = ChatResponse {
            text: Some("<tool_call>{bad json}</tool_call>".to_string()),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(text, None);
        assert!(calls.is_empty());
    }

    #[test]
    fn xml_parse_response_removes_tool_tags_from_text() {
        let dispatcher = XmlToolDispatcher;
        let response = ChatResponse {
            text: Some(
                "Intro <tool_call>{\"name\":\"shell\",\"arguments\":{}}</tool_call> Outro"
                    .to_string(),
            ),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(calls.len(), 1);
        let cleaned = text.expect("cleaned text should exist");
        assert!(!cleaned.contains("<tool_call>"));
        assert!(cleaned.contains("Intro"));
        assert!(cleaned.contains("Outro"));
    }

    #[test]
    fn xml_format_results_renders_tool_result_tags() {
        let dispatcher = XmlToolDispatcher;
        let message = dispatcher.format_results(&[sample_tool_result()]);

        match message {
            ConversationMessage::Chat(chat) => {
                assert_eq!(chat.role, ChatRole::User);
                assert!(
                    chat.content
                        .contains("<tool_result name=\"shell\" status=\"ok\">ok</tool_result>")
                );
            }
            other => panic!("expected chat message, got {other:?}"),
        }
    }

    #[test]
    fn xml_prompt_instructions_include_tool_info() {
        let dispatcher = XmlToolDispatcher;
        let instructions = dispatcher.prompt_instructions(&[sample_tool_spec()]);

        assert!(instructions.contains("<tool_call>"));
        assert!(instructions.contains("shell"));
        assert!(instructions.contains("Execute a shell command"));
    }

    #[test]
    fn xml_should_send_tool_specs_is_false() {
        let dispatcher = XmlToolDispatcher;
        assert!(!dispatcher.should_send_tool_specs());
    }

    #[test]
    fn xml_to_provider_messages_keeps_assistant_text_only() {
        let dispatcher = XmlToolDispatcher;
        let history = vec![ConversationMessage::AssistantToolCalls {
            text: Some("plain text".to_string()),
            tool_calls: vec![ToolCall {
                id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: "{}".to_string(),
            }],
        }];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages, vec![ChatMessage::assistant("plain text")]);
    }

    #[test]
    fn native_round_trip_survives_tape_restore() {
        let restored_messages = [
            ChatMessage::assistant("working").with_metadata(
                "tool_calls",
                r#"[{"id":"call-1","name":"shell","arguments":"{\"command\":\"ls\"}"}]"#,
            ),
            ChatMessage::user(
                r#"[Tool results]\n<tool_result name="shell" status="ok">done</tool_result>"#,
            )
            .with_metadata("tool_result_for", "call-1")
            .with_metadata("tool_result_content", "done")
            .with_metadata("is_error", "false"),
        ];

        assert_eq!(
            restored_messages[0]
                .metadata
                .get("tool_calls")
                .map(String::as_str),
            Some(r#"[{"id":"call-1","name":"shell","arguments":"{\"command\":\"ls\"}"}]"#)
        );
        assert_eq!(
            restored_messages[1]
                .metadata
                .get("tool_result_for")
                .map(String::as_str),
            Some("call-1")
        );
        assert_eq!(
            restored_messages[1]
                .metadata
                .get("tool_result_content")
                .map(String::as_str),
            Some("done")
        );
    }

    #[test]
    fn xml_round_trip_survives_tape_restore() {
        let restored_messages = [
            ChatMessage::assistant("Need output"),
            ChatMessage::user(
                r#"[Tool results]\n<tool_result name="shell" status="ok">done</tool_result>"#,
            ),
        ];

        assert_eq!(restored_messages[0], ChatMessage::assistant("Need output"));
        assert!(
            restored_messages[1]
                .content
                .contains(r#"<tool_result name="shell" status="ok">done</tool_result>"#)
        );
    }

    #[test]
    fn native_round_trip_preserves_tool_chain() {
        let dispatcher = NativeToolDispatcher;
        let response = ChatResponse {
            text: Some("running".to_string()),
            tool_calls: vec![ToolCall {
                id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: r#"{"command":"ls"}"#.to_string(),
            }],
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        let history = vec![
            ConversationMessage::AssistantToolCalls {
                text,
                tool_calls: response.tool_calls.clone(),
            },
            dispatcher.format_results(&[ToolExecutionResult {
                call_id: calls[0].id.clone(),
                name: calls[0].name.clone(),
                content: "done".to_string(),
                is_error: false,
            }]),
        ];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].metadata.get("tool_calls").map(String::as_str),
            Some(r#"[{"id":"call-1","name":"shell","arguments":"{\"command\":\"ls\"}"}]"#)
        );
        assert_eq!(
            messages[1]
                .metadata
                .get("tool_result_for")
                .map(String::as_str),
            Some("call-1")
        );
    }

    #[test]
    fn xml_round_trip_preserves_text_protocol() {
        let dispatcher = XmlToolDispatcher;
        let response = ChatResponse {
            text: Some(
                "Need output\n<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>"
                    .to_string(),
            ),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        let history = vec![
            ConversationMessage::AssistantToolCalls {
                text,
                tool_calls: vec![ToolCall {
                    id: calls[0].id.clone(),
                    name: calls[0].name.clone(),
                    arguments: serde_json::to_string(&calls[0].arguments)
                        .expect("arguments should serialize"),
                }],
            },
            dispatcher.format_results(&[ToolExecutionResult {
                call_id: calls[0].id.clone(),
                name: calls[0].name.clone(),
                content: "done".to_string(),
                is_error: false,
            }]),
        ];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], ChatMessage::assistant("Need output"));
        assert!(
            messages[1]
                .content
                .contains("<tool_result name=\"shell\" status=\"ok\">done</tool_result>")
        );
    }

    #[test]
    fn selected_dispatcher_native_behavior_matches_previous_trait_impl() {
        let dispatcher = SelectedToolDispatcher::Native;
        let response = ChatResponse {
            text: Some("working".to_string()),
            tool_calls: vec![ToolCall {
                id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: r#"{"command":"pwd"}"#.to_string(),
            }],
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        let history = vec![
            ConversationMessage::AssistantToolCalls {
                text,
                tool_calls: response.tool_calls.clone(),
            },
            dispatcher.format_results(&[ToolExecutionResult {
                call_id: calls[0].id.clone(),
                name: calls[0].name.clone(),
                content: "ok".to_string(),
                is_error: false,
            }]),
        ];

        assert!(
            dispatcher
                .prompt_instructions(&[sample_tool_spec()])
                .is_empty()
        );
        assert!(dispatcher.should_send_tool_specs());
        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::Assistant);
        assert_eq!(messages[0].content, "working");
        assert_eq!(messages[1].role, ChatRole::User);
        assert_eq!(messages[1].content, "ok");
        assert_eq!(
            messages[0].metadata.get("tool_calls").map(String::as_str),
            Some(r#"[{"id":"call-1","name":"shell","arguments":"{\"command\":\"pwd\"}"}]"#)
        );
    }

    #[test]
    fn selected_dispatcher_xml_behavior_matches_previous_trait_impl() {
        let dispatcher = SelectedToolDispatcher::Xml;
        let response = ChatResponse {
            text: Some(
                "Need output\n<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>"
                    .to_string(),
            ),
            tool_calls: Vec::new(),
            usage: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);
        let history = vec![
            ConversationMessage::AssistantToolCalls {
                text,
                tool_calls: vec![ToolCall {
                    id: calls[0].id.clone(),
                    name: calls[0].name.clone(),
                    arguments: serde_json::to_string(&calls[0].arguments)
                        .expect("arguments should serialize"),
                }],
            },
            dispatcher.format_results(&[ToolExecutionResult {
                call_id: calls[0].id.clone(),
                name: calls[0].name.clone(),
                content: "done".to_string(),
                is_error: false,
            }]),
        ];

        let instructions = dispatcher.prompt_instructions(&[sample_tool_spec()]);
        assert!(instructions.contains("<tool_call>"));
        assert!(!dispatcher.should_send_tool_specs());
        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages[0], ChatMessage::assistant("Need output"));
        assert!(
            messages[1]
                .content
                .contains("<tool_result name=\"shell\" status=\"ok\">done</tool_result>")
        );
    }
}
