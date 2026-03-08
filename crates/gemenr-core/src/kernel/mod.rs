//! Runtime kernel — agent loop, prompt composition, and turn control.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::json;

use crate::agent::dispatcher::{ConversationMessage, ToolDispatcher, ToolExecutionResult};
use crate::context::{ContextBuildResult, ContextManager, TokenBudget};
use crate::message::ChatMessage;
use crate::model::{ModelProvider, ModelRequest, ToolCall};
use crate::protocol::{EventEnvelope, EventKind, SessionId, TurnId};
use crate::tool_invoker::{PolicyDecision, ToolInvokeError, ToolInvoker};

pub mod prompt;
pub mod turn;

pub use prompt::PromptComposer;
pub use turn::{ActionDecision, TurnController};

const SUMMARY_PROMPT: &str = "Summarize the following conversation.";
const MAX_TURN_STEPS: usize = 50;

/// Errors from the agent runtime.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// Model call failed.
    #[error("model error: {0}")]
    Model(String),

    /// Context management error.
    #[error("context error: {0}")]
    Context(String),

    /// Tool execution error.
    #[error("tool error: {0}")]
    Tool(String),

    /// Turn was cancelled.
    #[error("turn cancelled")]
    Cancelled,
}

/// Independent runtime for a single task or conversation.
///
/// Each task gets its own [`AgentRuntime`] with an independent context tape.
/// Shared resources are injected via [`Arc`].
pub struct AgentRuntime {
    /// Context manager that owns this session's tape.
    context: ContextManager,
    /// Shared model provider.
    model: Arc<dyn ModelProvider>,
    /// Shared tool invoker.
    tools: Arc<dyn ToolInvoker>,
    /// Tool-calling strategy for this runtime.
    tool_dispatcher: Box<dyn ToolDispatcher>,
    /// Prompt composer.
    composer: PromptComposer,
    /// Turn controller.
    controller: TurnController,
    /// Token budget for context building.
    budget: TokenBudget,
    /// System prompt for this runtime.
    system_prompt: String,
    /// Model identifier for requests.
    model_name: String,
    /// Optional max tokens for requests.
    max_tokens: Option<u32>,
    /// Shared cancellation flag for the active turn.
    cancelled: Arc<AtomicBool>,
}

impl AgentRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: ContextManager,
        model: Arc<dyn ModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        tool_dispatcher: Box<dyn ToolDispatcher>,
        composer: PromptComposer,
        controller: TurnController,
        budget: TokenBudget,
        system_prompt: String,
        model_name: String,
        max_tokens: Option<u32>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            context,
            model,
            tools,
            tool_dispatcher,
            composer,
            controller,
            budget,
            system_prompt,
            model_name,
            max_tokens,
            cancelled,
        }
    }

    /// Execute a complete agent turn.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<String, AgentError> {
        self.check_cancelled()?;

        let turn_id = TurnId::new();
        self.append_user_input(&turn_id, user_input).await?;

        let mut history = Vec::new();

        for _ in 0..MAX_TURN_STEPS {
            self.check_cancelled()?;

            let mut provider_messages = match self.context.build_context(&self.budget) {
                ContextBuildResult::Ready(messages) => messages,
                ContextBuildResult::NeedsSummary { messages } => {
                    let summary = self.summarize(&messages).await?;
                    self.context
                        .apply_summary(summary.clone())
                        .await
                        .map_err(|error| AgentError::Context(error.to_string()))?;
                    self.context
                        .append(EventEnvelope::new(
                            self.context.session_id().clone(),
                            Some(turn_id.clone()),
                            EventKind::ContextSummarized,
                            json!({"summary": summary}),
                        ))
                        .await
                        .map_err(|error| AgentError::Context(error.to_string()))?;

                    match self.context.build_context(&self.budget) {
                        ContextBuildResult::Ready(messages) => messages,
                        ContextBuildResult::NeedsSummary { .. } => {
                            return Err(AgentError::Context(
                                "context still exceeds token budget after summary".to_string(),
                            ));
                        }
                    }
                }
            };

            provider_messages.extend(self.tool_dispatcher.to_provider_messages(&history));

            let request = self.composer.build_prompt(
                &self.context.soul_content().await,
                &self.system_prompt,
                provider_messages,
                &self.tools.list_specs(),
                self.tool_dispatcher.as_ref(),
                &self.model_name,
                self.max_tokens,
            );
            let response = self
                .model
                .chat(request)
                .await
                .map_err(|error| AgentError::Model(error.to_string()))?;
            let (text, tool_calls) = self.tool_dispatcher.parse_response(&response);

            match self.controller.next_action(text.clone(), tool_calls) {
                ActionDecision::Respond(response_text) => {
                    self.append_model_response(&turn_id, &response_text).await?;
                    return Ok(response_text);
                }
                ActionDecision::CompleteTurn => {
                    let final_text = text.unwrap_or_default();
                    if !final_text.is_empty() {
                        self.append_model_response(&turn_id, &final_text).await?;
                    }
                    return Ok(final_text);
                }
                ActionDecision::InvokeTools(tool_calls) => {
                    history.push(ConversationMessage::AssistantToolCalls {
                        text,
                        tool_calls: tool_calls
                            .iter()
                            .map(|call| ToolCall {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: serde_json::to_string(&call.arguments)
                                    .expect("tool arguments should serialize"),
                            })
                            .collect(),
                    });

                    let mut results = Vec::with_capacity(tool_calls.len());
                    for call in tool_calls {
                        self.check_cancelled()?;

                        match self.tools.check_policy(&call.name, &call.arguments) {
                            PolicyDecision::Allow => {}
                            PolicyDecision::NeedConfirmation(message) => {
                                if !self.confirm_via_stdin(&message) {
                                    let content = "User denied execution".to_string();
                                    self.emit_tool_event(
                                        &turn_id,
                                        EventKind::ToolFailed,
                                        &call.name,
                                        &content,
                                    )
                                    .await?;
                                    results.push(ToolExecutionResult {
                                        call_id: call.id,
                                        name: call.name,
                                        content,
                                        is_error: true,
                                    });
                                    continue;
                                }
                            }
                            PolicyDecision::Deny(reason) => {
                                let content = format!("Denied: {reason}");
                                self.emit_tool_event(
                                    &turn_id,
                                    EventKind::ToolFailed,
                                    &call.name,
                                    &content,
                                )
                                .await?;
                                results.push(ToolExecutionResult {
                                    call_id: call.id,
                                    name: call.name,
                                    content,
                                    is_error: true,
                                });
                                continue;
                            }
                        }

                        self.emit_tool_event(&turn_id, EventKind::ToolStarted, &call.name, "")
                            .await?;

                        match self
                            .tools
                            .invoke(&call.id, &call.name, call.arguments.clone())
                            .await
                        {
                            Ok(result) => {
                                let kind = if result.is_error {
                                    EventKind::ToolFailed
                                } else {
                                    EventKind::ToolCompleted
                                };
                                self.emit_tool_event(&turn_id, kind, &call.name, &result.content)
                                    .await?;
                                results.push(ToolExecutionResult {
                                    call_id: call.id,
                                    name: call.name,
                                    content: result.content,
                                    is_error: result.is_error,
                                });
                            }
                            Err(error) => {
                                let content = tool_error_message(&error, &call.name);
                                self.emit_tool_event(
                                    &turn_id,
                                    EventKind::ToolFailed,
                                    &call.name,
                                    &content,
                                )
                                .await?;
                                results.push(ToolExecutionResult {
                                    call_id: call.id,
                                    name: call.name,
                                    content,
                                    is_error: true,
                                });
                            }
                        }
                    }

                    history.push(self.tool_dispatcher.format_results(&results));
                }
            }
        }

        Err(AgentError::Tool(
            "turn exceeded maximum tool-calling steps".to_string(),
        ))
    }

    /// Abort the currently executing turn.
    pub fn abort_turn(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Return this runtime's session identifier.
    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        self.context.session_id()
    }

    async fn append_user_input(&mut self, turn_id: &TurnId, text: &str) -> Result<(), AgentError> {
        self.context
            .append(EventEnvelope::new(
                self.context.session_id().clone(),
                Some(turn_id.clone()),
                EventKind::UserInput,
                json!({"text": text}),
            ))
            .await
            .map_err(|error| AgentError::Context(error.to_string()))
    }

    async fn append_model_response(
        &mut self,
        turn_id: &TurnId,
        text: &str,
    ) -> Result<(), AgentError> {
        self.context
            .append(EventEnvelope::new(
                self.context.session_id().clone(),
                Some(turn_id.clone()),
                EventKind::ModelResponse,
                json!({"text": text}),
            ))
            .await
            .map_err(|error| AgentError::Context(error.to_string()))
    }

    async fn emit_tool_event(
        &mut self,
        turn_id: &TurnId,
        kind: EventKind,
        tool_name: &str,
        content: &str,
    ) -> Result<(), AgentError> {
        self.context
            .append(EventEnvelope::new(
                self.context.session_id().clone(),
                Some(turn_id.clone()),
                kind,
                json!({
                    "name": tool_name,
                    "result": content,
                }),
            ))
            .await
            .map_err(|error| AgentError::Context(error.to_string()))
    }

    async fn summarize(&self, messages: &[ChatMessage]) -> Result<String, AgentError> {
        let mut summary_messages = Vec::with_capacity(messages.len() + 1);
        summary_messages.push(ChatMessage::system(SUMMARY_PROMPT));
        summary_messages.extend(messages.iter().cloned());

        let response = self
            .model
            .complete(ModelRequest {
                messages: summary_messages,
                model: self.model_name.clone(),
                max_tokens: self.max_tokens,
            })
            .await
            .map_err(|error| AgentError::Model(error.to_string()))?;

        Ok(response.content)
    }

    fn confirm_via_stdin(&self, message: &str) -> bool {
        eprintln!("{message} [y/N]");
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        input.trim().eq_ignore_ascii_case("y")
    }

    fn check_cancelled(&self) -> Result<(), AgentError> {
        if self.cancelled.swap(false, Ordering::Relaxed) {
            Err(AgentError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn tool_error_message(error: &ToolInvokeError, tool_name: &str) -> String {
    match error {
        ToolInvokeError::NotFound(_) => format!("Tool '{tool_name}' not found"),
        ToolInvokeError::Execution(message) => message.clone(),
        ToolInvokeError::Timeout => "Tool execution timed out".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::Duration;

    use tokio::sync::RwLock;

    use super::*;
    use crate::agent::{NativeToolDispatcher, XmlToolDispatcher};
    use crate::context::SoulManager;
    use crate::context::{InMemoryTapeStore, TapeStore};
    use crate::error::ModelError;
    use crate::message::ChatRole;
    use crate::model::{ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelResponse};
    use crate::tool_invoker::ToolInvokeResult;
    use crate::tool_spec::{RiskLevel, ToolSpec};
    use async_trait::async_trait;

    struct ScriptedModelProvider {
        responses: Mutex<VecDeque<ChatResponse>>,
        requests: Mutex<Vec<ChatRequest>>,
        capabilities: ModelCapabilities,
    }

    impl ScriptedModelProvider {
        fn new(responses: Vec<ChatResponse>, capabilities: ModelCapabilities) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
                capabilities,
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl ModelProvider for ScriptedModelProvider {
        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
            let chat_request = ChatRequest {
                messages: request.messages,
                model: request.model,
                max_tokens: request.max_tokens,
                tools: None,
            };
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .push(chat_request);

            let response = self
                .responses
                .lock()
                .expect("responses lock should not be poisoned")
                .pop_front()
                .expect("scripted response should be available");

            Ok(ModelResponse {
                content: response.text.unwrap_or_default(),
                finish_reason: FinishReason::Stop,
            })
        }

        fn capabilities(&self) -> ModelCapabilities {
            self.capabilities
        }

        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .push(request);

            self.responses
                .lock()
                .expect("responses lock should not be poisoned")
                .pop_front()
                .ok_or_else(|| ModelError::Api {
                    status: 500,
                    message: "no scripted response available".to_string(),
                })
        }
    }

    struct ScriptedToolInvoker {
        specs: HashMap<String, ToolSpec>,
        policies: HashMap<String, PolicyDecision>,
        outputs: Mutex<HashMap<String, VecDeque<Result<ToolInvokeResult, ToolInvokeError>>>>,
        delay: Duration,
    }

    impl ScriptedToolInvoker {
        fn new(specs: Vec<ToolSpec>) -> Self {
            let specs = specs
                .into_iter()
                .map(|spec| (spec.name.clone(), spec))
                .collect();

            Self {
                specs,
                policies: HashMap::new(),
                outputs: Mutex::new(HashMap::new()),
                delay: Duration::from_millis(0),
            }
        }

        fn with_output(
            self,
            name: &str,
            output: Result<ToolInvokeResult, ToolInvokeError>,
        ) -> Self {
            self.outputs
                .lock()
                .expect("outputs lock should not be poisoned")
                .entry(name.to_string())
                .or_default()
                .push_back(output);
            self
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }
    }

    #[async_trait]
    impl ToolInvoker for ScriptedToolInvoker {
        fn lookup(&self, name: &str) -> Option<&ToolSpec> {
            self.specs.get(name)
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            self.specs.values().cloned().collect()
        }

        fn check_policy(&self, name: &str, _arguments: &serde_json::Value) -> PolicyDecision {
            self.policies
                .get(name)
                .cloned()
                .unwrap_or(PolicyDecision::Allow)
        }

        async fn invoke(
            &self,
            _call_id: &str,
            name: &str,
            _arguments: serde_json::Value,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            tokio::time::sleep(self.delay).await;

            let mut outputs_guard = self
                .outputs
                .lock()
                .expect("outputs lock should not be poisoned");
            let Some(outputs) = outputs_guard.get_mut(name) else {
                return Err(ToolInvokeError::NotFound(name.to_string()));
            };

            outputs
                .pop_front()
                .unwrap_or_else(|| Err(ToolInvokeError::NotFound(name.to_string())))
        }
    }

    fn sample_tool_spec() -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: "Echo content".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
            risk_level: RiskLevel::Low,
        }
    }

    fn runtime(
        model: Arc<dyn ModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        tape_store: Arc<dyn TapeStore>,
        tool_dispatcher: Box<dyn ToolDispatcher>,
    ) -> AgentRuntime {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-runtime");
        std::fs::create_dir_all(&workspace).expect("test workspace should be created");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&workspace).expect("SOUL.md should load"),
        ));

        AgentRuntime::new(
            ContextManager::new(SessionId::new(), tape_store, soul),
            model,
            tools,
            tool_dispatcher,
            PromptComposer,
            TurnController,
            TokenBudget::default(),
            "You are a helpful assistant.".to_string(),
            "test-model".to_string(),
            Some(256),
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[tokio::test]
    async fn run_turn_returns_plain_text_response() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![ChatResponse {
                text: Some("done".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            }],
            ModelCapabilities::default(),
        ));
        let tools = Arc::new(ScriptedToolInvoker::new(Vec::new()));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(
            model,
            tools,
            tape_store.clone(),
            Box::new(NativeToolDispatcher),
        );

        let response = runtime
            .run_turn("hello")
            .await
            .expect("turn should succeed");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert_eq!(response, "done");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::UserInput))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ModelResponse))
        );
    }

    #[tokio::test]
    async fn run_turn_executes_single_native_tool_call() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![
                ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                        arguments: json!({"value": "hello"}).to_string(),
                    }],
                    usage: None,
                },
                ChatResponse {
                    text: Some("all set".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                },
            ],
            ModelCapabilities {
                native_tool_calling: true,
                vision: false,
            },
        ));
        let tools = Arc::new(
            ScriptedToolInvoker::new(vec![sample_tool_spec()]).with_output(
                "echo",
                Ok(ToolInvokeResult {
                    content: "hello".to_string(),
                    is_error: false,
                }),
            ),
        );
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(
            model,
            tools,
            tape_store.clone(),
            Box::new(NativeToolDispatcher),
        );

        let response = runtime
            .run_turn("say hello")
            .await
            .expect("turn should succeed");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert_eq!(response, "all set");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ToolStarted))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ToolCompleted))
        );
    }

    #[tokio::test]
    async fn run_turn_executes_multiple_xml_tool_calls() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![
                ChatResponse {
                    text: Some(
                        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"value\":\"one\"}}</tool_call>"
                            .to_string(),
                    ),
                    tool_calls: Vec::new(),
                    usage: None,
                },
                ChatResponse {
                    text: Some(
                        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"value\":\"two\"}}</tool_call>"
                            .to_string(),
                    ),
                    tool_calls: Vec::new(),
                    usage: None,
                },
                ChatResponse {
                    text: Some("finished".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                },
            ],
            ModelCapabilities::default(),
        ));
        let tools = Arc::new(
            ScriptedToolInvoker::new(vec![sample_tool_spec()])
                .with_output(
                    "echo",
                    Ok(ToolInvokeResult {
                        content: "one".to_string(),
                        is_error: false,
                    }),
                )
                .with_output(
                    "echo",
                    Ok(ToolInvokeResult {
                        content: "two".to_string(),
                        is_error: false,
                    }),
                ),
        );
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(model, tools, tape_store, Box::new(XmlToolDispatcher));

        let response = runtime.run_turn("go").await.expect("turn should succeed");

        assert_eq!(response, "finished");
    }

    #[tokio::test]
    async fn run_turn_returns_tool_not_found_result_to_model() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![
                ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call-1".to_string(),
                        name: "missing".to_string(),
                        arguments: json!({}).to_string(),
                    }],
                    usage: None,
                },
                ChatResponse {
                    text: Some("handled missing tool".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                },
            ],
            ModelCapabilities {
                native_tool_calling: true,
                vision: false,
            },
        ));
        let tools = Arc::new(ScriptedToolInvoker::new(vec![sample_tool_spec()]));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let model_for_assertions = model.clone();
        let mut runtime = runtime(
            model,
            tools,
            tape_store.clone(),
            Box::new(NativeToolDispatcher),
        );

        let response = runtime
            .run_turn("call missing tool")
            .await
            .expect("turn should succeed");
        let requests = model_for_assertions.requests();
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert_eq!(response, "handled missing tool");
        assert!(requests[1].messages.iter().any(|message| {
            message.role == ChatRole::User && message.content.contains("Tool 'missing' not found")
        }));
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ToolFailed))
        );
    }

    #[tokio::test]
    async fn abort_turn_cancels_the_loop() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![ChatResponse {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".to_string(),
                    name: "echo".to_string(),
                    arguments: json!({"value": "hello"}).to_string(),
                }],
                usage: None,
            }],
            ModelCapabilities {
                native_tool_calling: true,
                vision: false,
            },
        ));
        let tools = Arc::new(
            ScriptedToolInvoker::new(vec![sample_tool_spec()])
                .with_output(
                    "echo",
                    Ok(ToolInvokeResult {
                        content: "hello".to_string(),
                        is_error: false,
                    }),
                )
                .with_delay(Duration::from_millis(50)),
        );
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(model, tools, tape_store, Box::new(NativeToolDispatcher));
        let cancelled = runtime.cancelled.clone();

        let handle = tokio::spawn(async move { runtime.run_turn("cancel me").await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancelled.store(true, Ordering::Relaxed);

        let result = handle.await.expect("task should join");
        assert!(matches!(result, Err(AgentError::Cancelled)));
    }
}
