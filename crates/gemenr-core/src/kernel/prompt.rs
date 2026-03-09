use crate::agent::dispatcher::SelectedToolDispatcher;
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

/// Runtime context for prompt composition.
///
/// Bundles all data required by [`PromptComposer::build_prompt`] into a single
/// structured input so the kernel can evolve prompt assembly without growing
/// the method signature further.
#[derive(Debug)]
pub struct PromptContext<'a> {
    /// SOUL.md content to inject into the synthesized system prompt.
    pub soul_content: &'a str,
    /// Base system prompt text configured for the runtime.
    pub system_prompt: &'a str,
    /// Conversation history messages to append after the system message.
    pub context_messages: Vec<ChatMessage>,
    /// Tool specifications available for the current turn.
    pub tools: &'a [ToolSpec],
    /// Dispatcher that decides prompt instructions and tool payload mode.
    pub dispatcher: &'a SelectedToolDispatcher,
    /// Target model identifier for the outgoing request.
    pub model: &'a str,
    /// Maximum tokens to request from the provider, when configured.
    pub max_tokens: Option<u32>,
}

impl PromptComposer {
    /// Build a complete chat request for the configured model.
    ///
    /// The resulting request always begins with a synthesized system message,
    /// followed by the provided context messages in their original order.
    #[must_use]
    pub fn build_prompt(&self, ctx: PromptContext<'_>) -> ChatRequest {
        let mut system = String::new();

        if !ctx.soul_content.trim().is_empty() {
            system.push_str(ctx.soul_content);
            system.push_str("\n\n");

            if estimated_soul_tokens(ctx.soul_content) > SOUL_BUDGET_WARNING_THRESHOLD_TOKENS {
                system.push_str(SOUL_BUDGET_WARNING);
                system.push_str("\n\n");
            }
        }

        system.push_str(ctx.system_prompt);

        let tool_instructions = ctx.dispatcher.prompt_instructions(ctx.tools);
        if !tool_instructions.is_empty() {
            system.push_str("\n\n");
            system.push_str(&tool_instructions);
        }

        let mut messages = vec![ChatMessage::system(system)];
        messages.extend(ctx.context_messages);

        let request_tools = if ctx.dispatcher.should_send_tool_specs() && !ctx.tools.is_empty() {
            Some(ctx.tools.to_vec())
        } else {
            None
        };

        ChatRequest {
            messages,
            model: ctx.model.to_string(),
            max_tokens: ctx.max_tokens,
            tools: request_tools,
        }
    }
}

fn estimated_soul_tokens(soul_content: &str) -> usize {
    soul_content.chars().count().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::{
        PromptComposer, PromptContext, SOUL_BUDGET_WARNING, SOUL_BUDGET_WARNING_THRESHOLD_TOKENS,
    };
    use crate::agent::{NativeToolDispatcher, XmlToolDispatcher};
    use crate::context::{ContextManager, InMemoryTapeStore, SoulManager, TapeStore};
    use crate::message::{ChatMessage, ChatRole};
    use crate::model::ChatRequest;
    use crate::protocol::SessionId;
    use crate::test_support::temp_dir;
    use crate::tool_spec::{RiskLevel, ToolSpec};
    use tokio::sync::RwLock;

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

    fn build_prompt(composer: &PromptComposer, ctx: PromptContext<'_>) -> ChatRequest {
        composer.build_prompt(ctx)
    }

    #[test]
    fn test_build_prompt_with_context() {
        let composer = PromptComposer;
        let context_messages = vec![ChatMessage::user("hello")];

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages,
                tools: &[sample_tool()],
                dispatcher: &NativeToolDispatcher,
                model: "claude-haiku-4-5-20251001",
                max_tokens: Some(256),
            },
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

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "# SOUL\nRemember constraints.",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
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
        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: &latest_soul,
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        assert!(request.messages[0].content.starts_with(updated_content));
        fs::remove_dir_all(workspace).expect("temp directory should be removed");
    }

    #[test]
    fn test_build_prompt_empty_soul() {
        let composer = PromptComposer;

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
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

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: &large_soul,
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        assert!(request.messages[0].content.contains(SOUL_BUDGET_WARNING));
    }

    #[test]
    fn test_build_prompt_xml_dispatcher() {
        let composer = PromptComposer;

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[sample_tool()],
                dispatcher: &XmlToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        let system = &request.messages[0].content;
        assert!(system.contains("Base prompt"));
        assert!(system.contains("<tool_call>"));
        assert!(system.contains("shell"));
    }

    #[test]
    fn build_prompt_does_not_inject_native_tool_instructions() {
        let composer = PromptComposer;

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[sample_tool()],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
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

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &tools,
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        assert_eq!(request.tools, Some(tools));
    }

    #[test]
    fn build_prompt_omits_tools_for_xml_dispatcher() {
        let composer = PromptComposer;

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[sample_tool()],
                dispatcher: &XmlToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        assert_eq!(request.tools, None);
    }

    #[test]
    fn build_prompt_keeps_existing_tool_instruction_behavior() {
        let composer = PromptComposer;
        let tools = vec![sample_tool()];

        let native_request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &tools,
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );
        let native_system = &native_request.messages[0].content;
        assert!(native_system.contains("Base prompt"));
        assert!(!native_system.contains("<tool_call>"));
        assert!(!native_system.contains("You have access to the following tools"));
        assert_eq!(native_request.tools, Some(tools.clone()));

        let xml_request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &tools,
                dispatcher: &XmlToolDispatcher,
                model: "model",
                max_tokens: None,
            },
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

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: context_messages.clone(),
                tools: &[],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        assert_eq!(request.messages[0].role, ChatRole::System);
        assert_eq!(request.messages[1..], context_messages);
    }

    #[test]
    fn build_prompt_omits_native_tools_when_list_is_empty() {
        let composer = PromptComposer;

        let request = build_prompt(
            &composer,
            PromptContext {
                soul_content: "",
                system_prompt: "Base prompt",
                context_messages: Vec::new(),
                tools: &[],
                dispatcher: &NativeToolDispatcher,
                model: "model",
                max_tokens: None,
            },
        );

        assert_eq!(request.tools, None);
    }
}
