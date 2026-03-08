use crate::agent::dispatcher::ToolDispatcher;
use crate::message::ChatMessage;
use crate::model::ChatRequest;
use crate::tool_spec::ToolSpec;

const SOUL_BUDGET_WARNING_THRESHOLD_TOKENS: usize = 1_024;
const SOUL_BUDGET_WARNING: &str = "SOUL.md is near its prompt budget. Preserve durable identity and preferences, but compact repetitive or stale SOUL entries before adding more.";

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

        if !soul_content.trim().is_empty() {
            system.push_str(soul_content);
            system.push_str("\n\n");

            if estimated_soul_tokens(soul_content) > SOUL_BUDGET_WARNING_THRESHOLD_TOKENS {
                system.push_str(SOUL_BUDGET_WARNING);
                system.push_str("\n\n");
            }
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

fn estimated_soul_tokens(soul_content: &str) -> usize {
    soul_content.chars().count().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{PromptComposer, SOUL_BUDGET_WARNING, SOUL_BUDGET_WARNING_THRESHOLD_TOKENS};
    use crate::agent::{NativeToolDispatcher, XmlToolDispatcher};
    use crate::context::{ContextManager, InMemoryTapeStore, SoulManager, TapeStore};
    use crate::message::{ChatMessage, ChatRole};
    use crate::protocol::SessionId;
    use crate::tool_spec::{RiskLevel, ToolSpec};
    use tokio::sync::RwLock;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = env::temp_dir().join(format!("gemenr-prompt-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

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

    #[tokio::test]
    async fn build_prompt_uses_latest_soul_content() {
        let composer = PromptComposer;
        let workspace = temp_dir("latest-soul");
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&workspace).expect("SOUL.md should load"),
        ));
        let manager = ContextManager::new(SessionId::new(), tape_store, soul);
        let updated_content = "# Identity\nLatest identity rules.\n\n# Preferences\nStay precise.\n\n# Experiences\nTrack regressions.\n\n# Notes\nKeep reports brief.\n";

        thread::sleep(Duration::from_millis(20));
        fs::write(workspace.join("SOUL.md"), updated_content).expect("SOUL.md should be updated");

        let latest_soul = manager
            .latest_soul_content()
            .await
            .expect("latest soul content should be readable");
        let request = composer.build_prompt(
            &latest_soul,
            "Base prompt",
            Vec::new(),
            &[],
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert!(request.messages[0].content.starts_with(updated_content));
        fs::remove_dir_all(workspace).expect("temp directory should be removed");
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
    fn build_prompt_warns_when_soul_exceeds_budget_threshold() {
        let composer = PromptComposer;
        let large_soul = format!(
            "# Identity\n{}",
            "a".repeat((SOUL_BUDGET_WARNING_THRESHOLD_TOKENS * 4) + 1)
        );

        let request = composer.build_prompt(
            &large_soul,
            "Base prompt",
            Vec::new(),
            &[],
            &NativeToolDispatcher,
            "model",
            None,
        );

        assert!(request.messages[0].content.contains(SOUL_BUDGET_WARNING));
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
    fn build_prompt_keeps_existing_tool_instruction_behavior() {
        let composer = PromptComposer;
        let tools = vec![sample_tool()];

        let native_request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &tools,
            &NativeToolDispatcher,
            "model",
            None,
        );
        let native_system = &native_request.messages[0].content;
        assert!(native_system.contains("Base prompt"));
        assert!(!native_system.contains("<tool_call>"));
        assert!(!native_system.contains("You have access to the following tools"));
        assert_eq!(native_request.tools, Some(tools.clone()));

        let xml_request = composer.build_prompt(
            "",
            "Base prompt",
            Vec::new(),
            &tools,
            &XmlToolDispatcher,
            "model",
            None,
        );
        let xml_system = &xml_request.messages[0].content;
        assert!(xml_system.contains("Base prompt"));
        assert!(xml_system.contains("<tool_call>"));
        assert!(xml_system.contains("shell"));
        assert_eq!(xml_request.tools, None);
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
