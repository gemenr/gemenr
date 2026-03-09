//! Runtime kernel — agent loop, prompt composition, and turn control.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::agent::dispatcher::{ConversationMessage, SelectedToolDispatcher, ToolExecutionResult};
use crate::context::{ContextBuildResult, ContextManager, TokenBudget};
use crate::message::ChatMessage;
use crate::model::{
    ChatRequest, ChatResponse, ModelProvider, ModelRequest, ModelResponse, RequestContext, ToolCall,
};
use crate::protocol::{
    AssistantToolCallsPayload, EventEnvelope, EventKind, SessionId, ToolCallRecord,
    ToolResultPayload, TurnId,
};
use crate::tool_invoker::{
    AuthorizationDecision, PolicyContext, PreparedToolCall, ToolCallRequest, ToolInvokeError,
    ToolInvokeResult, ToolInvoker,
};

/// Prompt assembly helpers.
pub mod prompt;
/// Turn-state transitions and decisions.
pub mod turn;

pub use prompt::PromptComposer;
pub use turn::{ActionDecision, TurnController, TurnState};

use self::turn::ModelStepOutcome;

const SUMMARY_PROMPT: &str = "Summarize the following conversation.";
const MAX_TURN_STEPS: usize = 50;

/// Errors from the agent runtime.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// Model call failed.
    #[error(transparent)]
    Model(#[from] crate::error::ModelError),

    /// Tape persistence or reconstruction error.
    #[error(transparent)]
    Tape(#[from] crate::context::TapeError),

    /// SOUL.md loading error.
    #[error(transparent)]
    Soul(#[from] crate::context::SoulError),

    /// Tool execution error.
    #[error(transparent)]
    Tool(#[from] crate::tool_invoker::ToolInvokeError),

    /// Turn was cancelled.
    #[error("turn cancelled")]
    Cancelled,

    /// Turn exceeded the maximum number of tool-calling steps.
    #[error("turn exceeded maximum tool-calling steps")]
    TurnLimitExceeded,
}

/// Typed approval request for a pending tool execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    /// Tool name that triggered the approval gate.
    pub tool_name: String,
    /// User-facing approval prompt.
    pub message: String,
}

/// Runtime input for one complete turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnInput {
    /// User-visible text to append to the conversation tape.
    pub text: String,
    /// Policy context forwarded to tool authorization.
    pub policy_context: PolicyContext,
}

/// Decision returned by an approval handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// The pending action may proceed.
    Approved,
    /// The pending action must be rejected.
    Rejected,
}

/// Handles user confirmations required before tool execution.
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// Ask whether a pending tool call should proceed.
    async fn confirm(&self, request: ApprovalRequest) -> ApprovalDecision;
}

/// Default approval handler used when no external confirmation mechanism exists.
#[derive(Debug, Default)]
pub struct DenyAllApprovals;

#[async_trait]
impl ApprovalHandler for DenyAllApprovals {
    async fn confirm(&self, _request: ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Rejected
    }
}

/// Receives runtime events for presentation or observability.
#[async_trait]
pub trait EventSink: Send + Sync {
    /// Publish a runtime event after it has been persisted to tape.
    async fn publish(&self, event: &EventEnvelope);
}

/// Default event sink that ignores all runtime events.
#[derive(Debug, Default)]
pub struct NoopEventSink;

#[async_trait]
impl EventSink for NoopEventSink {
    async fn publish(&self, _event: &EventEnvelope) {}
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
    tool_dispatcher: SelectedToolDispatcher,
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
    /// Shared request context for the active turn.
    request_context: RequestContext,
    /// Confirmation handler for medium/high-risk tool execution.
    approval_handler: Arc<dyn ApprovalHandler>,
    /// Event sink used for real-time event delivery.
    event_sink: Arc<dyn EventSink>,
}

impl AgentRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: ContextManager,
        model: Arc<dyn ModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        tool_dispatcher: SelectedToolDispatcher,
        composer: PromptComposer,
        controller: TurnController,
        budget: TokenBudget,
        system_prompt: String,
        model_name: String,
        max_tokens: Option<u32>,
        request_context: RequestContext,
        approval_handler: Arc<dyn ApprovalHandler>,
        event_sink: Arc<dyn EventSink>,
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
            request_context,
            approval_handler,
            event_sink,
        }
    }

    /// Restore the latest persisted anchor and post-anchor events for this session.
    pub async fn restore_from_tape(&mut self) -> Result<(), AgentError> {
        self.context.restore_from_tape().await?;
        Ok(())
    }

    /// Execute a complete agent turn with explicit runtime input.
    pub async fn run_turn_with_input(&mut self, input: TurnInput) -> Result<String, AgentError> {
        let turn_id = TurnId::new();
        let result = self.run_turn_inner(&turn_id, &input).await;
        let finalized = match result {
            Ok(response) => self.finalize_turn_success(&turn_id, response).await,
            Err(error) => {
                self.finalize_turn_failure(&turn_id, &error).await?;
                Err(error)
            }
        };

        self.request_context
            .cancelled
            .store(false, Ordering::Relaxed);

        finalized
    }

    /// Execute a complete agent turn with the default empty policy context.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<String, AgentError> {
        self.run_turn_with_input(TurnInput {
            text: user_input.to_string(),
            policy_context: PolicyContext::default(),
        })
        .await
    }

    async fn run_turn_inner(
        &mut self,
        turn_id: &TurnId,
        input: &TurnInput,
    ) -> Result<String, AgentError> {
        self.check_cancelled()?;
        self.append_user_input(turn_id, &input.text).await?;

        let mut history = Vec::new();

        for _ in 0..MAX_TURN_STEPS {
            self.check_cancelled()?;

            match self.run_model_step(turn_id, &mut history).await? {
                ModelStepOutcome::Complete(response_text) => return Ok(response_text),
                ModelStepOutcome::InvokeTools(tool_calls) => {
                    let formatted_results = self
                        .execute_tool_batch(turn_id, tool_calls, &input.policy_context)
                        .await?;
                    history.push(formatted_results);
                }
            }
        }

        Err(AgentError::TurnLimitExceeded)
    }

    async fn build_provider_messages_for_step(
        &mut self,
        turn_id: &TurnId,
        history: &[ConversationMessage],
    ) -> Result<Vec<ChatMessage>, AgentError> {
        let mut provider_messages = self.maybe_summarize_context(turn_id).await?;
        provider_messages.extend(self.tool_dispatcher.to_provider_messages(history));
        Ok(provider_messages)
    }

    async fn maybe_summarize_context(
        &mut self,
        turn_id: &TurnId,
    ) -> Result<Vec<ChatMessage>, AgentError> {
        match self.context.build_context(&self.budget) {
            ContextBuildResult::Ready(messages) => Ok(messages),
            ContextBuildResult::NeedsSummary { messages } => {
                let summary = self.summarize(&messages).await?;
                self.context.apply_summary(summary.clone()).await?;
                self.append_event(EventEnvelope::new(
                    self.context.session_id().clone(),
                    Some(turn_id.clone()),
                    EventKind::ContextSummarized,
                    json!({"summary": summary}),
                ))
                .await?;

                match self.context.build_context(&self.budget) {
                    ContextBuildResult::Ready(messages) => Ok(messages),
                    ContextBuildResult::NeedsSummary { .. } => Err(context_invariant_error(
                        "context still exceeds token budget after summary",
                    )),
                }
            }
        }
    }

    async fn run_model_step(
        &mut self,
        turn_id: &TurnId,
        history: &mut Vec<ConversationMessage>,
    ) -> Result<ModelStepOutcome, AgentError> {
        let provider_messages = self
            .build_provider_messages_for_step(turn_id, history)
            .await?;
        let soul_content = self.context.latest_soul_content().await?;
        let request = self.composer.build_prompt(
            &soul_content,
            &self.system_prompt,
            provider_messages,
            &self.tools.list_specs(),
            &self.tool_dispatcher,
            &self.model_name,
            self.max_tokens,
        );
        let response = self.invoke_chat_request(request).await?;
        let (text, tool_calls) = self.tool_dispatcher.parse_response(&response);

        match self.controller.next_action(text.clone(), tool_calls) {
            ActionDecision::Respond(response_text) => {
                self.append_model_response(turn_id, &response_text).await?;
                Ok(ModelStepOutcome::Complete(response_text))
            }
            ActionDecision::CompleteTurn => {
                let final_text = text.unwrap_or_default();
                if !final_text.is_empty() {
                    self.append_model_response(turn_id, &final_text).await?;
                }
                Ok(ModelStepOutcome::Complete(final_text))
            }
            ActionDecision::InvokeTools(tool_calls) => {
                let assistant_tool_calls = AssistantToolCallsPayload {
                    text: text.clone(),
                    tool_calls: tool_calls
                        .iter()
                        .map(|call| ToolCallRecord {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        })
                        .collect(),
                };
                self.append_assistant_tool_calls(turn_id, &assistant_tool_calls)
                    .await?;

                history.push(ConversationMessage::AssistantToolCalls {
                    text: assistant_tool_calls.text.clone(),
                    tool_calls: assistant_tool_calls
                        .tool_calls
                        .iter()
                        .map(|call| ToolCall {
                            id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: serde_json::to_string(&call.arguments)
                                .expect("tool arguments should serialize"),
                        })
                        .collect(),
                });

                Ok(ModelStepOutcome::InvokeTools(tool_calls))
            }
        }
    }

    async fn execute_tool_batch(
        &mut self,
        turn_id: &TurnId,
        tool_calls: Vec<crate::agent::dispatcher::ParsedToolCall>,
        policy_context: &PolicyContext,
    ) -> Result<ConversationMessage, AgentError> {
        let mut results = Vec::with_capacity(tool_calls.len());

        for call in tool_calls {
            self.check_cancelled()?;

            let request = ToolCallRequest {
                call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            };
            let prepared = match self.tools.authorize(&request, policy_context) {
                AuthorizationDecision::Prepared(prepared) => prepared,
                AuthorizationDecision::NeedConfirmation { prepared, message } => {
                    let approval = self
                        .approval_handler
                        .confirm(ApprovalRequest {
                            tool_name: call.name.clone(),
                            message: message.clone(),
                        })
                        .await;
                    if approval != ApprovalDecision::Approved {
                        let content = format!("Execution not approved: {message}");
                        self.emit_tool_event(
                            turn_id,
                            EventKind::ToolDenied,
                            &call.id,
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
                    prepared
                }
                AuthorizationDecision::Denied { reason } => {
                    let content = format!("Denied: {reason}");
                    self.emit_tool_event(
                        turn_id,
                        EventKind::ToolDenied,
                        &call.id,
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
            };

            self.emit_tool_event(turn_id, EventKind::ToolStarted, &call.id, &call.name, "")
                .await?;

            match self.invoke_tool(prepared).await {
                Ok(result) => {
                    let kind = if result.is_error {
                        EventKind::ToolFailed
                    } else {
                        EventKind::ToolCompleted
                    };
                    self.emit_tool_event(turn_id, kind, &call.id, &call.name, &result.content)
                        .await?;
                    results.push(ToolExecutionResult {
                        call_id: call.id,
                        name: call.name,
                        content: result.content,
                        is_error: result.is_error,
                    });
                }
                Err(error) => {
                    if matches!(error, ToolInvokeError::Cancelled) {
                        self.emit_tool_event(
                            turn_id,
                            EventKind::ToolFailed,
                            &call.id,
                            &call.name,
                            "Tool execution cancelled",
                        )
                        .await?;
                        return Err(AgentError::Cancelled);
                    }

                    let content = tool_error_message(&error, &call.name);
                    let kind = tool_error_event_kind(&error);
                    self.emit_tool_event(turn_id, kind, &call.id, &call.name, &content)
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

        Ok(self.tool_dispatcher.format_results(&results))
    }

    async fn finalize_turn_success(
        &mut self,
        turn_id: &TurnId,
        response: String,
    ) -> Result<String, AgentError> {
        self.append_turn_completed(turn_id, &response).await?;
        Ok(response)
    }

    async fn finalize_turn_failure(
        &mut self,
        turn_id: &TurnId,
        error: &AgentError,
    ) -> Result<(), AgentError> {
        self.append_turn_failed(turn_id, error).await
    }

    /// Abort the currently executing turn.
    pub fn abort_turn(&self) {
        self.request_context
            .cancelled
            .store(true, Ordering::Relaxed);
    }

    /// Return a clone of the runtime cancellation flag for external interrupt wiring.
    #[must_use]
    pub fn cancellation_handle(&self) -> Arc<AtomicBool> {
        self.request_context.cancellation_handle()
    }

    /// Return this runtime's session identifier.
    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        self.context.session_id()
    }

    #[cfg(test)]
    pub(crate) fn selected_tool_dispatcher(&self) -> SelectedToolDispatcher {
        self.tool_dispatcher
    }

    async fn append_user_input(&mut self, turn_id: &TurnId, text: &str) -> Result<(), AgentError> {
        self.append_event(EventEnvelope::new(
            self.context.session_id().clone(),
            Some(turn_id.clone()),
            EventKind::UserInput,
            json!({"text": text}),
        ))
        .await
    }

    async fn append_model_response(
        &mut self,
        turn_id: &TurnId,
        text: &str,
    ) -> Result<(), AgentError> {
        self.append_event(EventEnvelope::new(
            self.context.session_id().clone(),
            Some(turn_id.clone()),
            EventKind::ModelResponse,
            json!({"text": text}),
        ))
        .await
    }

    async fn append_assistant_tool_calls(
        &mut self,
        turn_id: &TurnId,
        payload: &AssistantToolCallsPayload,
    ) -> Result<(), AgentError> {
        self.append_event(EventEnvelope::new(
            self.context.session_id().clone(),
            Some(turn_id.clone()),
            EventKind::AssistantToolCalls,
            serde_json::to_value(payload).expect("assistant tool calls payload should serialize"),
        ))
        .await
    }

    async fn emit_tool_event(
        &mut self,
        turn_id: &TurnId,
        kind: EventKind,
        call_id: &str,
        tool_name: &str,
        content: &str,
    ) -> Result<(), AgentError> {
        let payload = match kind {
            EventKind::ToolStarted => json!({"call_id": call_id, "name": tool_name}),
            EventKind::ToolFailed | EventKind::ToolDenied | EventKind::ToolTimedOut => {
                let mut payload = serde_json::to_value(ToolResultPayload {
                    call_id: call_id.to_string(),
                    name: tool_name.to_string(),
                    content: content.to_string(),
                    is_error: true,
                })
                .expect("tool result payload should serialize");
                payload
                    .as_object_mut()
                    .expect("tool result payload should be an object")
                    .insert("error".to_string(), json!(content));
                payload
            }
            EventKind::ToolCompleted => {
                let mut payload = serde_json::to_value(ToolResultPayload {
                    call_id: call_id.to_string(),
                    name: tool_name.to_string(),
                    content: content.to_string(),
                    is_error: false,
                })
                .expect("tool result payload should serialize");
                payload
                    .as_object_mut()
                    .expect("tool result payload should be an object")
                    .insert("result".to_string(), json!(content));
                payload
            }
            _ => json!({"call_id": call_id, "name": tool_name, "content": content}),
        };

        self.append_event(EventEnvelope::new(
            self.context.session_id().clone(),
            Some(turn_id.clone()),
            kind,
            payload,
        ))
        .await
    }

    async fn append_turn_completed(
        &mut self,
        turn_id: &TurnId,
        response: &str,
    ) -> Result<(), AgentError> {
        self.append_event(EventEnvelope::new(
            self.context.session_id().clone(),
            Some(turn_id.clone()),
            EventKind::TurnCompleted,
            json!({"response": response}),
        ))
        .await
    }

    async fn append_turn_failed(
        &mut self,
        turn_id: &TurnId,
        error: &AgentError,
    ) -> Result<(), AgentError> {
        self.append_event(EventEnvelope::new(
            self.context.session_id().clone(),
            Some(turn_id.clone()),
            EventKind::TurnFailed,
            json!({
                "error": error.to_string(),
                "category": turn_failure_category(error),
            }),
        ))
        .await
    }

    async fn append_event(&mut self, event: EventEnvelope) -> Result<(), AgentError> {
        self.context.append(event.clone()).await?;
        self.event_sink.publish(&event).await;
        Ok(())
    }

    async fn summarize(&self, messages: &[ChatMessage]) -> Result<String, AgentError> {
        let mut summary_messages = Vec::with_capacity(messages.len() + 1);
        summary_messages.push(ChatMessage::system(SUMMARY_PROMPT));
        summary_messages.extend(messages.iter().cloned());

        let response = self
            .invoke_completion_request(ModelRequest {
                messages: summary_messages,
                model: self.model_name.clone(),
                max_tokens: self.max_tokens,
            })
            .await?;

        Ok(response.content)
    }

    fn check_cancelled(&self) -> Result<(), AgentError> {
        if self.request_context.cancelled.load(Ordering::Relaxed) {
            Err(AgentError::Cancelled)
        } else {
            Ok(())
        }
    }

    async fn invoke_chat_request(&self, request: ChatRequest) -> Result<ChatResponse, AgentError> {
        let context = self.request_context.clone();
        let request_future = self.model.chat(request, context.clone());

        match self.wait_for_model_request(request_future, context).await? {
            Some(response) => Ok(response),
            None => Err(AgentError::Cancelled),
        }
    }

    async fn invoke_completion_request(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse, AgentError> {
        let context = self.request_context.clone();
        let request_future = self.model.complete(request, context.clone());

        match self.wait_for_model_request(request_future, context).await? {
            Some(response) => Ok(response),
            None => Err(AgentError::Cancelled),
        }
    }

    async fn invoke_tool(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolInvokeResult, ToolInvokeError> {
        let context = self.request_context.clone();
        let request_future = self.tools.invoke(prepared, context.cancellation_handle());
        tokio::pin!(request_future);

        let cancellation_future = wait_for_cancellation(context.cancellation_handle());
        tokio::pin!(cancellation_future);

        if let Some(timeout) = context.timeout {
            let timeout_future = tokio::time::sleep(timeout);
            tokio::pin!(timeout_future);

            tokio::select! {
                result = &mut request_future => result,
                _ = &mut cancellation_future => Err(ToolInvokeError::Cancelled),
                _ = &mut timeout_future => Err(ToolInvokeError::Timeout),
            }
        } else {
            tokio::select! {
                result = &mut request_future => result,
                _ = &mut cancellation_future => Err(ToolInvokeError::Cancelled),
            }
        }
    }

    async fn wait_for_model_request<T, Fut>(
        &self,
        request_future: Fut,
        context: RequestContext,
    ) -> Result<Option<T>, AgentError>
    where
        Fut: std::future::Future<Output = Result<T, crate::error::ModelError>>,
    {
        tokio::pin!(request_future);

        let cancellation_future = wait_for_cancellation(context.cancellation_handle());
        tokio::pin!(cancellation_future);

        let model_result = if let Some(timeout) = context.timeout {
            let timeout_future = tokio::time::sleep(timeout);
            tokio::pin!(timeout_future);

            tokio::select! {
                result = &mut request_future => Some(result),
                _ = &mut cancellation_future => None,
                _ = &mut timeout_future => Some(Err(crate::error::ModelError::Timeout)),
            }
        } else {
            tokio::select! {
                result = &mut request_future => Some(result),
                _ = &mut cancellation_future => None,
            }
        };

        match model_result {
            Some(Ok(response)) => Ok(Some(response)),
            Some(Err(crate::error::ModelError::Cancelled)) | None => Ok(None),
            Some(Err(error)) => Err(AgentError::Model(error)),
        }
    }
}

async fn wait_for_cancellation(cancelled: Arc<AtomicBool>) {
    while !cancelled.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn context_invariant_error(message: &str) -> AgentError {
    crate::context::TapeError::Io(std::io::Error::other(message.to_string())).into()
}

fn tool_error_message(error: &ToolInvokeError, tool_name: &str) -> String {
    match error {
        ToolInvokeError::NotFound(_) => format!("Tool '{tool_name}' not found"),
        ToolInvokeError::Denied { reason } => format!("Denied: {reason}"),
        ToolInvokeError::ApprovalDenied { message } => {
            format!("Execution not approved: {message}")
        }
        ToolInvokeError::Execution { message } => message.clone(),
        ToolInvokeError::Timeout => "Tool execution timed out".to_string(),
        ToolInvokeError::Cancelled => format!("Tool '{tool_name}' execution cancelled"),
    }
}

fn tool_error_event_kind(error: &ToolInvokeError) -> EventKind {
    match error {
        ToolInvokeError::Denied { .. } | ToolInvokeError::ApprovalDenied { .. } => {
            EventKind::ToolDenied
        }
        ToolInvokeError::Timeout => EventKind::ToolTimedOut,
        ToolInvokeError::NotFound(_)
        | ToolInvokeError::Cancelled
        | ToolInvokeError::Execution { .. } => EventKind::ToolFailed,
    }
}

fn turn_failure_category(error: &AgentError) -> &'static str {
    match error {
        AgentError::Model(_) => "model",
        AgentError::Tape(_) => "tape",
        AgentError::Soul(_) => "soul",
        AgentError::Tool(_) => "tool",
        AgentError::Cancelled => "cancelled",
        AgentError::TurnLimitExceeded => "turn_limit_exceeded",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use tokio::sync::RwLock;

    use super::*;
    use crate::agent::{NativeToolDispatcher, SelectedToolDispatcher, XmlToolDispatcher};
    use crate::context::SoulManager;
    use crate::context::{InMemoryTapeStore, TapeStore};
    use crate::error::ModelError;
    use crate::message::{ChatMessage, ChatRole};
    use crate::model::{ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelResponse};
    use crate::tool_invoker::{
        AuthorizationDecision, ExecutionPolicy, PolicyContext, PreparedToolCall, SandboxKind,
        ToolAuthorizer, ToolCallRequest, ToolCatalog, ToolExecutor, ToolInvokeResult,
    };
    use crate::tool_spec::{RiskLevel, ToolSpec};
    use async_trait::async_trait;

    struct ScriptedModelProvider {
        responses: Mutex<VecDeque<ChatResponse>>,
        requests: Mutex<Vec<ChatRequest>>,
        contexts: Mutex<Vec<RequestContext>>,
        capabilities: ModelCapabilities,
    }

    impl ScriptedModelProvider {
        fn new(responses: Vec<ChatResponse>, capabilities: ModelCapabilities) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
                contexts: Mutex::new(Vec::new()),
                capabilities,
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .clone()
        }

        fn contexts(&self) -> Vec<RequestContext> {
            self.contexts
                .lock()
                .expect("contexts lock should not be poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl ModelProvider for ScriptedModelProvider {
        async fn complete(
            &self,
            request: ModelRequest,
            context: RequestContext,
        ) -> Result<ModelResponse, ModelError> {
            self.contexts
                .lock()
                .expect("contexts lock should not be poisoned")
                .push(context);
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

        async fn chat(
            &self,
            request: ChatRequest,
            context: RequestContext,
        ) -> Result<ChatResponse, ModelError> {
            self.contexts
                .lock()
                .expect("contexts lock should not be poisoned")
                .push(context);
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
        policies: HashMap<String, ExecutionPolicy>,
        authorization_contexts: Mutex<Vec<PolicyContext>>,
        outputs: Mutex<HashMap<String, VecDeque<Result<ToolInvokeResult, ToolInvokeError>>>>,
        cancellation_handles: Mutex<Vec<Arc<AtomicBool>>>,
        delay: Duration,
        invocations: AtomicUsize,
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
                authorization_contexts: Mutex::new(Vec::new()),
                outputs: Mutex::new(HashMap::new()),
                cancellation_handles: Mutex::new(Vec::new()),
                delay: Duration::from_millis(0),
                invocations: AtomicUsize::new(0),
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

        fn with_policy(mut self, name: &str, policy: ExecutionPolicy) -> Self {
            self.policies.insert(name.to_string(), policy);
            self
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }

        fn invocation_count(&self) -> usize {
            self.invocations.load(Ordering::Relaxed)
        }

        fn last_cancellation_handle(&self) -> Option<Arc<AtomicBool>> {
            self.cancellation_handles
                .lock()
                .expect("cancellation handles lock should not be poisoned")
                .last()
                .cloned()
        }

        fn authorization_contexts(&self) -> Vec<PolicyContext> {
            self.authorization_contexts
                .lock()
                .expect("authorization contexts lock should not be poisoned")
                .clone()
        }
    }

    impl ToolCatalog for ScriptedToolInvoker {
        fn lookup(&self, name: &str) -> Option<&ToolSpec> {
            self.specs.get(name)
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            self.specs.values().cloned().collect()
        }
    }

    impl ToolAuthorizer for ScriptedToolInvoker {
        fn authorize(
            &self,
            request: &ToolCallRequest,
            context: &PolicyContext,
        ) -> AuthorizationDecision {
            self.authorization_contexts
                .lock()
                .expect("authorization contexts lock should not be poisoned")
                .push(context.clone());
            let policy =
                self.policies
                    .get(&request.name)
                    .cloned()
                    .unwrap_or(ExecutionPolicy::Allow {
                        sandbox: SandboxKind::None,
                    });
            let prepared = PreparedToolCall {
                request: request.clone(),
                policy: policy.clone(),
            };

            match policy {
                ExecutionPolicy::Allow { .. } => AuthorizationDecision::Prepared(prepared),
                ExecutionPolicy::NeedConfirmation { message, .. } => {
                    AuthorizationDecision::NeedConfirmation { prepared, message }
                }
                ExecutionPolicy::Deny { reason } => AuthorizationDecision::Denied { reason },
            }
        }
    }

    #[async_trait]
    impl ToolExecutor for ScriptedToolInvoker {
        async fn invoke(
            &self,
            prepared: PreparedToolCall,
            cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            self.cancellation_handles
                .lock()
                .expect("cancellation handles lock should not be poisoned")
                .push(Arc::clone(&cancelled));

            if cancelled.load(Ordering::Relaxed) {
                return Err(ToolInvokeError::Cancelled);
            }

            self.invocations.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(self.delay).await;

            if cancelled.load(Ordering::Relaxed) {
                return Err(ToolInvokeError::Cancelled);
            }

            let mut outputs_guard = self
                .outputs
                .lock()
                .expect("outputs lock should not be poisoned");
            let tool_name = prepared.request.name.clone();
            let Some(outputs) = outputs_guard.get_mut(&tool_name) else {
                return Err(ToolInvokeError::NotFound(tool_name));
            };

            outputs
                .pop_front()
                .unwrap_or(Err(ToolInvokeError::NotFound(tool_name)))
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
        tool_dispatcher: SelectedToolDispatcher,
    ) -> AgentRuntime {
        runtime_with_approval(
            model,
            tools,
            tape_store,
            tool_dispatcher,
            Arc::new(DenyAllApprovals),
        )
    }

    fn runtime_with_approval(
        model: Arc<dyn ModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        tape_store: Arc<dyn TapeStore>,
        tool_dispatcher: SelectedToolDispatcher,
        approval_handler: Arc<dyn ApprovalHandler>,
    ) -> AgentRuntime {
        runtime_with_session(
            SessionId::new(),
            model,
            tools,
            tape_store,
            tool_dispatcher,
            approval_handler,
        )
    }

    fn runtime_with_session(
        session_id: SessionId,
        model: Arc<dyn ModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        tape_store: Arc<dyn TapeStore>,
        tool_dispatcher: SelectedToolDispatcher,
        approval_handler: Arc<dyn ApprovalHandler>,
    ) -> AgentRuntime {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/test-runtime-{}-{timestamp}",
            session_id.0
        ));
        std::fs::create_dir_all(&workspace).expect("test workspace should be created");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&workspace).expect("SOUL.md should load"),
        ));

        AgentRuntime::new(
            ContextManager::new(session_id, tape_store, soul),
            model,
            tools,
            tool_dispatcher,
            PromptComposer,
            TurnController,
            TokenBudget::default(),
            "You are a helpful assistant.".to_string(),
            "test-model".to_string(),
            Some(256),
            RequestContext::new(Arc::new(AtomicBool::new(false))),
            approval_handler,
            Arc::new(NoopEventSink),
        )
    }

    struct ApproveAllApprovals;

    #[async_trait]
    impl ApprovalHandler for ApproveAllApprovals {
        async fn confirm(&self, _request: ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Approved
        }
    }

    struct DelayedApprovalHandler {
        delay: Duration,
        confirmations: AtomicUsize,
    }

    impl DelayedApprovalHandler {
        fn new(delay: Duration) -> Self {
            Self {
                delay,
                confirmations: AtomicUsize::new(0),
            }
        }

        fn confirmation_count(&self) -> usize {
            self.confirmations.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl ApprovalHandler for DelayedApprovalHandler {
        async fn confirm(&self, _request: ApprovalRequest) -> ApprovalDecision {
            tokio::time::sleep(self.delay).await;
            self.confirmations.fetch_add(1, Ordering::Relaxed);
            ApprovalDecision::Approved
        }
    }

    struct BlockingModelProvider {
        started: Arc<tokio::sync::Notify>,
    }

    impl BlockingModelProvider {
        fn new(started: Arc<tokio::sync::Notify>) -> Self {
            Self { started }
        }
    }

    #[async_trait]
    impl ModelProvider for BlockingModelProvider {
        async fn complete(
            &self,
            _request: ModelRequest,
            _context: RequestContext,
        ) -> Result<ModelResponse, ModelError> {
            unreachable!("blocking test provider should use chat()")
        }

        async fn chat(
            &self,
            _request: ChatRequest,
            context: RequestContext,
        ) -> Result<ChatResponse, ModelError> {
            self.started.notify_waiters();
            super::wait_for_cancellation(context.cancellation_handle()).await;
            Err(ModelError::Cancelled)
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities::default()
        }
    }

    #[tokio::test]
    async fn model_step_returns_complete_when_no_tool_calls() {
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
        let mut runtime = runtime(model, tools, tape_store, NativeToolDispatcher);
        let turn_id = TurnId::new();
        let mut history = Vec::new();

        runtime
            .append_user_input(&turn_id, "hello")
            .await
            .expect("user input should append");

        let outcome = runtime
            .run_model_step(&turn_id, &mut history)
            .await
            .expect("model step should succeed");

        assert_eq!(outcome, ModelStepOutcome::Complete("done".to_string()));
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn tool_batch_step_formats_results_via_selected_dispatcher() {
        let model = Arc::new(ScriptedModelProvider::new(
            Vec::new(),
            ModelCapabilities::default(),
        ));
        let tools = Arc::new(
            ScriptedToolInvoker::new(vec![sample_tool_spec()]).with_output(
                "echo",
                Ok(ToolInvokeResult {
                    content: "done".to_string(),
                    is_error: false,
                }),
            ),
        );
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(model, tools, tape_store, XmlToolDispatcher);
        let turn_id = TurnId::new();

        let message = runtime
            .execute_tool_batch(
                &turn_id,
                vec![crate::agent::ParsedToolCall {
                    id: "call-1".to_string(),
                    name: "echo".to_string(),
                    arguments: json!({"value": "hello"}),
                }],
                &PolicyContext::default(),
            )
            .await
            .expect("tool batch should succeed");

        assert_eq!(
            message,
            ConversationMessage::Chat(ChatMessage::user(
                "[Tool results]\n<tool_result name=\"echo\" status=\"ok\">done</tool_result>",
            ))
        );
    }

    #[tokio::test]
    async fn summarize_step_rebuilds_context_once_budget_is_exceeded() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![ChatResponse {
                text: Some("compressed".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            }],
            ModelCapabilities::default(),
        ));
        let model_for_assertions = model.clone();
        let tools = Arc::new(ScriptedToolInvoker::new(Vec::new()));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);
        let turn_id = TurnId::new();
        runtime.budget = TokenBudget {
            max_tokens: 16,
            threshold: 0.8,
        };

        runtime
            .append_user_input(&turn_id, &"a".repeat(64))
            .await
            .expect("user input should append");
        runtime
            .append_model_response(&turn_id, &"b".repeat(64))
            .await
            .expect("model response should append");

        let messages = runtime
            .maybe_summarize_context(&turn_id)
            .await
            .expect("summarize step should succeed");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert_eq!(
            messages,
            vec![ChatMessage::system(
                "Summary of earlier context:\ncompressed"
            )]
        );
        assert_eq!(model_for_assertions.requests().len(), 1);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event.kind, EventKind::ContextSummarized))
                .count(),
            1
        );
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
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

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
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::TurnCompleted))
        );
    }

    #[tokio::test]
    async fn run_turn_persists_turn_completed_event_on_success() {
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
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

        let response = runtime
            .run_turn("hello")
            .await
            .expect("turn should succeed");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert_eq!(response, "done");
        assert!(events.iter().any(|event| {
            matches!(event.kind, EventKind::TurnCompleted)
                && event
                    .payload
                    .get("response")
                    .and_then(|value| value.as_str())
                    == Some("done")
        }));
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
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

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
        let mut runtime = runtime(model, tools, tape_store, XmlToolDispatcher);

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
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

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
        let tape_store_for_assertions = tape_store.clone();
        let mut runtime = runtime(model, tools, tape_store, NativeToolDispatcher);
        let session_id = runtime.session_id().clone();
        let cancelled = runtime.cancellation_handle();

        let handle = tokio::spawn(async move { runtime.run_turn("cancel me").await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancelled.store(true, Ordering::Relaxed);

        let result = handle.await.expect("task should join");
        assert!(matches!(result, Err(AgentError::Cancelled)));
        let events = tape_store_for_assertions
            .load_all(&session_id)
            .await
            .expect("events should load");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::TurnFailed))
        );
    }

    #[tokio::test]
    async fn abort_turn_cancels_inflight_model_request() {
        let started = Arc::new(tokio::sync::Notify::new());
        let started_wait = started.notified();
        let model = Arc::new(BlockingModelProvider::new(Arc::clone(&started)));
        let tools = Arc::new(ScriptedToolInvoker::new(Vec::new()));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let tape_store_for_assertions = Arc::clone(&tape_store);
        let mut runtime = runtime(model, tools, Arc::clone(&tape_store), NativeToolDispatcher);
        let session_id = runtime.session_id().clone();
        let cancelled = runtime.cancellation_handle();

        let handle = tokio::spawn(async move { runtime.run_turn("block on model").await });
        started_wait.await;
        cancelled.store(true, Ordering::Relaxed);

        let result = handle.await.expect("task should join");
        assert!(matches!(result, Err(AgentError::Cancelled)));

        let events = tape_store_for_assertions
            .load_all(&session_id)
            .await
            .expect("events should load");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::TurnFailed))
        );
    }

    #[tokio::test]
    async fn approval_handler_is_async_compatible() {
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
                    text: Some("handled confirmation".to_string()),
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
            ScriptedToolInvoker::new(vec![sample_tool_spec()])
                .with_policy(
                    "echo",
                    ExecutionPolicy::NeedConfirmation {
                        message: "approve echo".to_string(),
                        sandbox: SandboxKind::Seatbelt,
                    },
                )
                .with_output(
                    "echo",
                    Ok(ToolInvokeResult {
                        content: "hello".to_string(),
                        is_error: false,
                    }),
                ),
        );
        let approval_handler = Arc::new(DelayedApprovalHandler::new(Duration::from_millis(10)));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime_with_approval(
            model,
            tools.clone(),
            tape_store,
            NativeToolDispatcher,
            approval_handler.clone(),
        );

        let response = runtime
            .run_turn("try confirmed tool")
            .await
            .expect("turn should succeed after async approval");

        assert_eq!(response, "handled confirmation");
        assert_eq!(tools.invocation_count(), 1);
        assert_eq!(approval_handler.confirmation_count(), 1);
    }

    #[tokio::test]
    async fn run_turn_with_input_forwards_policy_context() {
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
                    text: Some("done".to_string()),
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
        let mut runtime = runtime(model, tools.clone(), tape_store, NativeToolDispatcher);
        let policy_context = PolicyContext {
            organization_id: Some("org-1".to_string()),
            workspace_id: Some("ws-1".to_string()),
            conversation_id: Some("conv-1".to_string()),
        };

        let response = runtime
            .run_turn_with_input(TurnInput {
                text: "hello".to_string(),
                policy_context: policy_context.clone(),
            })
            .await
            .expect("turn should succeed");

        assert_eq!(response, "done");
        assert_eq!(tools.authorization_contexts(), vec![policy_context]);
    }

    #[tokio::test]
    async fn tool_and_model_share_same_cancellation_source() {
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
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                },
            ],
            ModelCapabilities {
                native_tool_calling: true,
                vision: false,
            },
        ));
        let model_for_assertions = Arc::clone(&model);
        let tools = Arc::new(
            ScriptedToolInvoker::new(vec![sample_tool_spec()]).with_output(
                "echo",
                Ok(ToolInvokeResult {
                    content: "hello".to_string(),
                    is_error: false,
                }),
            ),
        );
        let tools_for_assertions = Arc::clone(&tools);
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(model, tools, tape_store, NativeToolDispatcher);

        let response = runtime
            .run_turn("share cancellation")
            .await
            .expect("turn should succeed");

        assert_eq!(response, "done");
        let contexts = model_for_assertions.contexts();
        assert!(!contexts.is_empty());
        let tool_cancelled = tools_for_assertions
            .last_cancellation_handle()
            .expect("tool invocation should record cancellation handle");
        assert!(
            contexts
                .iter()
                .all(|context| Arc::ptr_eq(&context.cancelled, &tool_cancelled))
        );
    }

    #[tokio::test]
    async fn run_turn_persists_turn_failed_event_on_step_limit() {
        let responses = (0..MAX_TURN_STEPS)
            .map(|index| ChatResponse {
                text: None,
                tool_calls: vec![ToolCall {
                    id: format!("call-{index}"),
                    name: "echo".to_string(),
                    arguments: json!({"value": index}).to_string(),
                }],
                usage: None,
            })
            .collect();
        let model = Arc::new(ScriptedModelProvider::new(
            responses,
            ModelCapabilities {
                native_tool_calling: true,
                vision: false,
            },
        ));
        let mut scripted_tools = ScriptedToolInvoker::new(vec![sample_tool_spec()]);
        for index in 0..MAX_TURN_STEPS {
            scripted_tools = scripted_tools.with_output(
                "echo",
                Ok(ToolInvokeResult {
                    content: index.to_string(),
                    is_error: false,
                }),
            );
        }
        let tools = Arc::new(scripted_tools);
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

        let error = runtime
            .run_turn("loop forever")
            .await
            .expect_err("turn should hit step limit");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert!(matches!(error, AgentError::TurnLimitExceeded));
        assert!(events.iter().any(|event| {
            matches!(event.kind, EventKind::TurnFailed)
                && event
                    .payload
                    .get("category")
                    .and_then(|value| value.as_str())
                    == Some("turn_limit_exceeded")
        }));
    }

    #[tokio::test]
    async fn runtime_executes_prepared_tool_call_after_confirmation() {
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
                    text: Some("handled confirmation".to_string()),
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
            ScriptedToolInvoker::new(vec![sample_tool_spec()])
                .with_policy(
                    "echo",
                    ExecutionPolicy::NeedConfirmation {
                        message: "approve echo".to_string(),
                        sandbox: SandboxKind::Seatbelt,
                    },
                )
                .with_output(
                    "echo",
                    Ok(ToolInvokeResult {
                        content: "hello".to_string(),
                        is_error: false,
                    }),
                ),
        );
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime_with_approval(
            model,
            tools.clone(),
            tape_store,
            NativeToolDispatcher,
            Arc::new(ApproveAllApprovals),
        );

        let response = runtime
            .run_turn("try confirmed tool")
            .await
            .expect("turn should succeed after approval");

        assert_eq!(response, "handled confirmation");
        assert_eq!(tools.invocation_count(), 1);
    }

    #[tokio::test]
    async fn runtime_does_not_invoke_executor_when_authorization_denied() {
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
                    text: Some("handled denial".to_string()),
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
            ScriptedToolInvoker::new(vec![sample_tool_spec()]).with_policy(
                "echo",
                ExecutionPolicy::Deny {
                    reason: "policy blocked tool".to_string(),
                },
            ),
        );
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let mut runtime = runtime(
            model,
            tools.clone(),
            tape_store.clone(),
            NativeToolDispatcher,
        );

        let response = runtime
            .run_turn("try denied tool")
            .await
            .expect("turn should succeed after denial");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");

        assert_eq!(response, "handled denial");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ToolDenied))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ToolFailed))
        );
        assert_eq!(tools.invocation_count(), 0);
    }

    #[tokio::test]
    async fn run_turn_persists_assistant_tool_calls_to_tape() {
        let model = Arc::new(ScriptedModelProvider::new(
            vec![
                ChatResponse {
                    text: Some("working".to_string()),
                    tool_calls: vec![ToolCall {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                        arguments: json!({"value": "hello"}).to_string(),
                    }],
                    usage: None,
                },
                ChatResponse {
                    text: Some("done".to_string()),
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
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

        runtime
            .run_turn("say hello")
            .await
            .expect("turn should succeed");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");
        let assistant_event = events
            .iter()
            .find(|event| matches!(event.kind, EventKind::AssistantToolCalls))
            .expect("assistant tool calls event should exist");
        let payload = serde_json::from_value::<crate::protocol::AssistantToolCallsPayload>(
            assistant_event.payload.clone(),
        )
        .expect("assistant tool calls payload should deserialize");

        assert_eq!(payload.text.as_deref(), Some("working"));
        assert_eq!(payload.tool_calls.len(), 1);
        assert_eq!(payload.tool_calls[0].call_id, "call-1");
        assert_eq!(payload.tool_calls[0].name, "echo");
        assert_eq!(payload.tool_calls[0].arguments, json!({"value": "hello"}));
    }

    #[tokio::test]
    async fn tool_result_events_include_call_id_for_round_trip() {
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
                    text: Some("done".to_string()),
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
        let mut runtime = runtime(model, tools, tape_store.clone(), NativeToolDispatcher);

        runtime
            .run_turn("say hello")
            .await
            .expect("turn should succeed");
        let events = tape_store
            .load_all(runtime.session_id())
            .await
            .expect("events should load");
        let tool_event = events
            .iter()
            .find(|event| matches!(event.kind, EventKind::ToolCompleted))
            .expect("tool completed event should exist");
        let payload = serde_json::from_value::<crate::protocol::ToolResultPayload>(
            tool_event.payload.clone(),
        )
        .expect("tool result payload should deserialize");

        assert_eq!(payload.call_id, "call-1");
        assert_eq!(payload.name, "echo");
        assert_eq!(payload.content, "hello");
        assert!(!payload.is_error);
    }

    #[tokio::test]
    async fn restore_from_tape_recovers_existing_history_for_new_runtime() {
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let session_id = SessionId::new();

        let initial_model = Arc::new(ScriptedModelProvider::new(
            vec![ChatResponse {
                text: Some("first answer".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            }],
            ModelCapabilities::default(),
        ));
        let tools = Arc::new(ScriptedToolInvoker::new(Vec::new()));
        let mut initial_runtime = runtime_with_session(
            session_id.clone(),
            initial_model,
            tools.clone(),
            tape_store.clone(),
            NativeToolDispatcher,
            Arc::new(DenyAllApprovals),
        );

        initial_runtime
            .run_turn("first question")
            .await
            .expect("initial turn should succeed");

        let restored_model = Arc::new(ScriptedModelProvider::new(
            vec![ChatResponse {
                text: Some("second answer".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            }],
            ModelCapabilities::default(),
        ));
        let restored_model_for_assertions = restored_model.clone();
        let mut restored_runtime = runtime_with_session(
            session_id,
            restored_model,
            tools,
            tape_store,
            NativeToolDispatcher,
            Arc::new(DenyAllApprovals),
        );

        restored_runtime
            .restore_from_tape()
            .await
            .expect("runtime should restore tape");
        restored_runtime
            .run_turn("second question")
            .await
            .expect("restored turn should succeed");

        let request = &restored_model_for_assertions.requests()[0];
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::User && message.content == "first question"
        }));
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::Assistant && message.content == "first answer"
        }));
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::User && message.content == "second question"
        }));
    }

    #[tokio::test]
    async fn restore_from_tape_recovers_anchor_and_post_anchor_context() {
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let session_id = SessionId::new();

        let initial_model = Arc::new(ScriptedModelProvider::new(
            vec![
                ChatResponse {
                    text: Some("before answer".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                },
                ChatResponse {
                    text: Some("after answer".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                },
            ],
            ModelCapabilities::default(),
        ));
        let tools = Arc::new(ScriptedToolInvoker::new(Vec::new()));
        let mut initial_runtime = runtime_with_session(
            session_id.clone(),
            initial_model,
            tools.clone(),
            tape_store.clone(),
            NativeToolDispatcher,
            Arc::new(DenyAllApprovals),
        );

        initial_runtime
            .run_turn("before anchor")
            .await
            .expect("pre-anchor turn should succeed");
        initial_runtime
            .context
            .create_anchor("earlier summary".to_string())
            .await
            .expect("anchor should be created");
        initial_runtime
            .run_turn("after anchor")
            .await
            .expect("post-anchor turn should succeed");

        let restored_model = Arc::new(ScriptedModelProvider::new(
            vec![ChatResponse {
                text: Some("current answer".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            }],
            ModelCapabilities::default(),
        ));
        let restored_model_for_assertions = restored_model.clone();
        let mut restored_runtime = runtime_with_session(
            session_id,
            restored_model,
            tools,
            tape_store,
            NativeToolDispatcher,
            Arc::new(DenyAllApprovals),
        );

        restored_runtime
            .restore_from_tape()
            .await
            .expect("runtime should restore tape");
        restored_runtime
            .run_turn("current question")
            .await
            .expect("restored turn should succeed");

        let request = &restored_model_for_assertions.requests()[0];
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::System
                && message.content == "Summary of earlier context:\nearlier summary"
        }));
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::User && message.content == "after anchor"
        }));
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::Assistant && message.content == "after answer"
        }));
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::User && message.content == "current question"
        }));
    }
}
