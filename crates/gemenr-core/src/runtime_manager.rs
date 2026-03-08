use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde_json::Value;
use thiserror::Error;

use crate::access::{AccessInbound, AccessOutbound, ConversationId};
use crate::builder::RuntimeBuilder;
use crate::kernel::{AgentError, AgentRuntime};
use crate::protocol::SessionId;

/// Manages long-lived conversations on top of [`AgentRuntime`].
pub struct RuntimeManager {
    builder: RuntimeBuilder,
    system_prompt: String,
    conversations: HashMap<ConversationId, ManagedConversation>,
}

/// Errors that can occur while managing long-lived runtimes.
#[derive(Debug, Error)]
pub enum RuntimeManagerError {
    /// The requested conversation is not currently managed.
    #[error("conversation not found: {0}")]
    ConversationNotFound(String),
    /// The runtime failed while restoring or running a turn.
    #[error(transparent)]
    Runtime(#[from] AgentError),
    /// No outbound message was produced for a queued dispatch.
    #[error("conversation queue produced no outbound message")]
    MissingOutbound,
}

/// In-memory state for one managed conversation.
struct ManagedConversation {
    runtime: Option<AgentRuntime>,
    session_id: SessionId,
    queued_turns: VecDeque<AccessInbound>,
    last_activity: Instant,
    needs_restore: bool,
}

impl RuntimeManager {
    /// Create a runtime manager with a shared builder and stable system prompt.
    #[must_use]
    pub fn new(builder: RuntimeBuilder, system_prompt: String) -> Self {
        Self {
            builder,
            system_prompt,
            conversations: HashMap::new(),
        }
    }

    /// Dispatch one inbound message through the managed runtime for its conversation.
    pub async fn dispatch(
        &mut self,
        inbound: AccessInbound,
    ) -> Result<AccessOutbound, RuntimeManagerError> {
        let conversation_id = inbound.conversation_id.clone();
        let mut conversation = self
            .conversations
            .remove(&conversation_id)
            .unwrap_or_else(|| self.new_conversation());
        conversation.queued_turns.push_back(inbound);

        let result = self.process_queue(&mut conversation).await;
        self.conversations.insert(conversation_id, conversation);
        result
    }

    /// Hibernate one conversation by dropping its in-memory runtime.
    pub fn hibernate(&mut self, id: &ConversationId) -> Result<(), RuntimeManagerError> {
        let conversation = self
            .conversations
            .get_mut(id)
            .ok_or_else(|| RuntimeManagerError::ConversationNotFound(id.0.clone()))?;
        conversation.runtime = None;
        conversation.needs_restore = true;
        conversation.last_activity = Instant::now();
        Ok(())
    }

    /// Recreate an in-memory runtime for one managed conversation.
    pub fn resume(&mut self, id: &ConversationId) -> Result<(), RuntimeManagerError> {
        let conversation = self
            .conversations
            .get_mut(id)
            .ok_or_else(|| RuntimeManagerError::ConversationNotFound(id.0.clone()))?;
        if conversation.runtime.is_none() {
            conversation.runtime =
                Some(self.builder.build_with_session(
                    self.system_prompt.clone(),
                    conversation.session_id.clone(),
                ));
        }
        conversation.needs_restore = true;
        conversation.last_activity = Instant::now();
        Ok(())
    }

    /// Hibernate all conversations that have been idle for at least `max_idle`.
    pub fn hibernate_idle(
        &mut self,
        max_idle: Duration,
    ) -> Result<Vec<ConversationId>, RuntimeManagerError> {
        let mut hibernated = Vec::new();
        for (conversation_id, conversation) in &mut self.conversations {
            if conversation.runtime.is_some() && conversation.last_activity.elapsed() >= max_idle {
                conversation.runtime = None;
                conversation.needs_restore = true;
                hibernated.push(conversation_id.clone());
            }
        }
        Ok(hibernated)
    }

    fn new_conversation(&self) -> ManagedConversation {
        let session_id = SessionId::new();
        ManagedConversation {
            runtime: Some(
                self.builder
                    .build_with_session(self.system_prompt.clone(), session_id.clone()),
            ),
            session_id,
            queued_turns: VecDeque::new(),
            last_activity: Instant::now(),
            needs_restore: false,
        }
    }

    async fn process_queue(
        &self,
        conversation: &mut ManagedConversation,
    ) -> Result<AccessOutbound, RuntimeManagerError> {
        let mut last_outbound = None;

        while let Some(inbound) = conversation.queued_turns.pop_front() {
            if conversation.runtime.is_none() {
                conversation.runtime = Some(self.builder.build_with_session(
                    self.system_prompt.clone(),
                    conversation.session_id.clone(),
                ));
            }

            let runtime = conversation
                .runtime
                .as_mut()
                .expect("runtime should be present while processing a queue");
            if conversation.needs_restore {
                runtime.restore_from_tape().await?;
                conversation.needs_restore = false;
            }

            let content = runtime.run_turn(&inbound.text).await?;
            conversation.last_activity = Instant::now();
            last_outbound = Some(AccessOutbound {
                conversation_id: inbound.conversation_id,
                route: inbound.route,
                content,
                metadata: Value::Null,
            });
        }

        last_outbound.ok_or(RuntimeManagerError::MissingOutbound)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use serde_json::json;
    use tokio::sync::RwLock;

    use super::RuntimeManager;
    use crate::access::{AccessInbound, ConversationId, ReplyRoute};
    use crate::builder::RuntimeBuilder;
    use crate::context::{InMemoryTapeStore, SoulManager, TapeStore};
    use crate::error::ModelError;
    use crate::message::ChatRole;
    use crate::model::{
        ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
        ModelResponse,
    };
    use crate::tool_invoker::{
        ExecutionPolicy, PolicyContext, SandboxKind, ToolInvokeError, ToolInvokeResult, ToolInvoker,
    };
    use crate::tool_spec::ToolSpec;

    struct RecordingModelProvider {
        requests: Mutex<Vec<ChatRequest>>,
    }

    impl RecordingModelProvider {
        fn new() -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
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
    impl ModelProvider for RecordingModelProvider {
        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
            Ok(ModelResponse {
                content: request
                    .messages
                    .iter()
                    .rev()
                    .find(|message| message.role == ChatRole::User)
                    .map(|message| format!("echo:{}", message.content))
                    .unwrap_or_else(|| "echo:".to_string()),
                finish_reason: FinishReason::Stop,
            })
        }

        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .push(request.clone());
            Ok(ChatResponse {
                text: Some(
                    request
                        .messages
                        .iter()
                        .rev()
                        .find(|message| message.role == ChatRole::User)
                        .map(|message| format!("echo:{}", message.content))
                        .unwrap_or_else(|| "echo:".to_string()),
                ),
                tool_calls: Vec::new(),
                usage: None,
            })
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities::default()
        }
    }

    struct NoopToolInvoker;

    #[async_trait]
    impl ToolInvoker for NoopToolInvoker {
        fn lookup(&self, _name: &str) -> Option<&ToolSpec> {
            None
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            Vec::new()
        }

        fn check_policy(
            &self,
            _name: &str,
            _arguments: &serde_json::Value,
            _context: &PolicyContext,
        ) -> ExecutionPolicy {
            ExecutionPolicy::Allow {
                sandbox: SandboxKind::None,
            }
        }

        async fn invoke(
            &self,
            _call_id: &str,
            name: &str,
            _arguments: serde_json::Value,
            _cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Err(ToolInvokeError::NotFound(name.to_string()))
        }
    }

    fn builder(
        model: Arc<RecordingModelProvider>,
        tape_store: Arc<dyn TapeStore>,
        workspace_tag: &str,
    ) -> RuntimeBuilder {
        let workspace = std::env::temp_dir().join(format!(
            "gemenr-runtime-manager-{workspace_tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&workspace).expect("workspace should exist");
        let soul = SoulManager::load(&workspace).expect("soul should load");

        RuntimeBuilder::new(
            model,
            Arc::new(NoopToolInvoker),
            Arc::new(RwLock::new(soul)),
            tape_store,
        )
        .model_name("test-model".to_string())
    }

    fn inbound(conversation_id: &str, text: &str) -> AccessInbound {
        AccessInbound {
            conversation_id: ConversationId(conversation_id.to_string()),
            user_id: "user-1".to_string(),
            text: text.to_string(),
            route: ReplyRoute::stdio(),
            metadata: json!({"source": "test"}),
        }
    }

    #[tokio::test]
    async fn same_conversation_dispatches_turns_in_order() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model.clone(), tape_store.clone(), "ordered");
        let mut manager = RuntimeManager::new(builder, "system".to_string());

        let first = manager
            .dispatch(inbound("conv-1", "hello"))
            .await
            .expect("first turn should succeed");
        let second = manager
            .dispatch(inbound("conv-1", "follow up"))
            .await
            .expect("second turn should succeed");

        assert_eq!(first.content, "echo:hello");
        assert_eq!(second.content, "echo:follow up");

        let requests = model.requests();
        assert_eq!(requests.len(), 2);
        let user_messages = requests[1]
            .messages
            .iter()
            .filter(|message| message.role == ChatRole::User)
            .map(|message| message.content.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            user_messages,
            vec!["hello".to_string(), "follow up".to_string()]
        );
    }

    #[tokio::test]
    async fn different_conversations_keep_independent_sessions() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model, tape_store, "isolated");
        let mut manager = RuntimeManager::new(builder, "system".to_string());

        manager
            .dispatch(inbound("conv-a", "one"))
            .await
            .expect("first conversation should succeed");
        manager
            .dispatch(inbound("conv-b", "two"))
            .await
            .expect("second conversation should succeed");

        let session_a = manager
            .conversations
            .get(&ConversationId("conv-a".to_string()))
            .expect("conversation a should exist")
            .session_id
            .clone();
        let session_b = manager
            .conversations
            .get(&ConversationId("conv-b".to_string()))
            .expect("conversation b should exist")
            .session_id
            .clone();

        assert_ne!(session_a, session_b);
    }

    #[tokio::test]
    async fn hibernate_drops_runtime_but_keeps_session_metadata() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model, tape_store, "hibernate");
        let mut manager = RuntimeManager::new(builder, "system".to_string());
        let conversation_id = ConversationId("conv-1".to_string());

        manager
            .dispatch(inbound("conv-1", "hello"))
            .await
            .expect("dispatch should succeed");
        manager
            .hibernate(&conversation_id)
            .expect("hibernate should succeed");

        let conversation = manager
            .conversations
            .get(&conversation_id)
            .expect("conversation should exist");
        assert!(conversation.runtime.is_none());
        assert!(conversation.needs_restore);
    }

    #[tokio::test]
    async fn resume_restores_context_from_tape() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model.clone(), tape_store, "resume");
        let mut manager = RuntimeManager::new(builder, "system".to_string());
        let conversation_id = ConversationId("conv-restore".to_string());

        manager
            .dispatch(inbound("conv-restore", "first turn"))
            .await
            .expect("first turn should succeed");
        manager
            .hibernate(&conversation_id)
            .expect("hibernate should succeed");
        manager
            .resume(&conversation_id)
            .expect("resume should succeed");
        manager
            .dispatch(inbound("conv-restore", "second turn"))
            .await
            .expect("second turn should succeed");

        let requests = model.requests();
        assert_eq!(requests.len(), 2);
        let restored_user_messages = requests[1]
            .messages
            .iter()
            .filter(|message| message.role == ChatRole::User)
            .map(|message| message.content.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            restored_user_messages,
            vec!["first turn".to_string(), "second turn".to_string()]
        );
    }

    #[tokio::test]
    async fn hibernate_idle_only_reclaims_expired_conversations() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model, tape_store, "idle");
        let mut manager = RuntimeManager::new(builder, "system".to_string());

        manager
            .dispatch(inbound("conv-hot", "hot"))
            .await
            .expect("hot conversation should succeed");
        manager
            .dispatch(inbound("conv-cold", "cold"))
            .await
            .expect("cold conversation should succeed");

        manager
            .conversations
            .get_mut(&ConversationId("conv-cold".to_string()))
            .expect("cold conversation should exist")
            .last_activity = Instant::now() - Duration::from_secs(300);

        let reclaimed = manager
            .hibernate_idle(Duration::from_secs(60))
            .expect("idle collection should succeed");

        assert_eq!(reclaimed, vec![ConversationId("conv-cold".to_string())]);
        assert!(
            manager
                .conversations
                .get(&ConversationId("conv-cold".to_string()))
                .expect("cold conversation should exist")
                .runtime
                .is_none()
        );
        assert!(
            manager
                .conversations
                .get(&ConversationId("conv-hot".to_string()))
                .expect("hot conversation should exist")
                .runtime
                .is_some()
        );
    }

    #[tokio::test]
    async fn builder_task_mode_path_still_works_without_runtime_manager() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model, tape_store, "task-mode");
        let mut runtime = builder.build("system".to_string());

        let response = runtime
            .run_turn("standalone")
            .await
            .expect("runtime builder path should still work");

        assert_eq!(response, "echo:standalone");
    }
}
