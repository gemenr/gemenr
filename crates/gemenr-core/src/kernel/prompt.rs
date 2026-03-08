use crate::agent::dispatcher::ToolDispatcher;
use crate::message::ChatMessage;
use crate::model::ChatRequest;
use crate::tool_spec::ToolSpec;

/// Assembles model input from SOUL content, system prompt, tools, and context.
///
/// This type is intentionally stateless and performs pure data transformation.
#[derive(Debug, Default, Clone, Copy)]
pub struct PromptComposer;

impl PromptComposer {
    /// Build a complete chat request for the configured model.
    ///
    /// The resulting request always begins with a synthesized system message,
    /// followed by the provided context messages in their original order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn build_prompt(
        &self,
        soul_content: &str,
        system_prompt: &str,
        context_messages: Vec<ChatMessage>,
        tools: &[ToolSpec],
        dispatcher: &dyn ToolDispatcher,
        model: &str,
        max_tokens: Option<u32>,
    ) -> ChatRequest {
        let mut system = String::new();

        if !soul_content.is_empty() {
            system.push_str(soul_content);
            system.push_str("\n\n");
        }

        system.push_str(system_prompt);

        let tool_instructions = dispatcher.prompt_instructions(tools);
        if !tool_instructions.is_empty() {
            system.push_str("\n\n");
            system.push_str(&tool_instructions);
        }

        let mut messages = vec![ChatMessage::system(system)];
        messages.extend(context_messages);

        let request_tools = if dispatcher.should_send_tool_specs() && !tools.is_empty() {
            Some(tools.to_vec())
        } else {
            None
        };

        ChatRequest {
            messages,
            model: model.to_string(),
            max_tokens,
            tools: request_tools,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PromptComposer;
    use crate::agent::{NativeToolDispatcher, XmlToolDispatcher};
    use crate::message::{ChatMessage, ChatRole};
    use crate::tool_spec::{RiskLevel, ToolSpec};

    fn sample_tool() -> ToolSpec {
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

    #[test]
    fn build_prompt_basic_includes_messages_and_model() {
        let composer = PromptComposer;
        let context_messages = vec![ChatMessage::user("hello")];

        let request = composer.build_prompt(
            "",
            "Base prompt",
            context_messages,
            &[sample_tool()],
            &NativeToolDispatcher,
            "claude-haiku-4-5-20251001",
            Some(256),
        );

        assert_eq!(request.model, "claude-haiku-4-5-20251001");
        assert_eq!(request.max_tokens, Some(256));
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0], ChatMessage::system("Base prompt"));
        assert_eq!(request.messages[1], ChatMessage::user("hello"));
    }

    #[test]
    fn build_prompt_injects_soul_content_when_present() {
        let composer = PromptComposer;

        let request = composer.build_prompt(
            "# SOUL\nRemember constraints.",
            "Base prompt",
            Vec::new(),
            &[],
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, ChatRole::System);
        assert!(
            request.messages[0]
                .content
                .starts_with("# SOUL\nRemember constraints.\n\nBase prompt")
        );
    }

    #[test]
    fn build_prompt_skips_empty_soul_header() {
        let composer = PromptComposer;

        let request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &[],
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert_eq!(request.messages[0].content, "Base prompt");
    }

    #[test]
    fn build_prompt_injects_xml_tool_instructions() {
        let composer = PromptComposer;

        let request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &[sample_tool()],
            &XmlToolDispatcher,
            "model",
            None,
        );

        let system = &request.messages[0].content;
        assert!(system.contains("Base prompt"));
        assert!(system.contains("<tool_call>"));
        assert!(system.contains("shell"));
    }

    #[test]
    fn build_prompt_does_not_inject_native_tool_instructions() {
        let composer = PromptComposer;

        let request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &[sample_tool()],
            &NativeToolDispatcher,
            "model",
            None,
        );

        let system = &request.messages[0].content;
        assert!(system.contains("Base prompt"));
        assert!(!system.contains("<tool_call>"));
        assert!(!system.contains("You have access to the following tools"));
    }

    #[test]
    fn build_prompt_sets_tools_for_native_dispatcher() {
        let composer = PromptComposer;
        let tools = vec![sample_tool()];

        let request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &tools,
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert_eq!(request.tools, Some(tools));
    }

    #[test]
    fn build_prompt_omits_tools_for_xml_dispatcher() {
        let composer = PromptComposer;

        let request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &[sample_tool()],
            &XmlToolDispatcher,
            "model",
            None,
        );

        assert_eq!(request.tools, None);
    }

    #[test]
    fn build_prompt_appends_context_after_system_message() {
        let composer = PromptComposer;
        let context_messages = vec![
            ChatMessage::user("question"),
            ChatMessage::assistant("answer"),
        ];

        let request = composer.build_prompt(
            "",
            "Base prompt",
            context_messages.clone(),
            &[],
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert_eq!(request.messages[0].role, ChatRole::System);
        assert_eq!(request.messages[1..], context_messages);
    }

    #[test]
    fn build_prompt_omits_native_tools_when_list_is_empty() {
        let composer = PromptComposer;

        let request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &[],
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert_eq!(request.tools, None);
    }
}
