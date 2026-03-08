use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::access::{AccessInbound, AccessOutbound, ConversationId};
use crate::builder::RuntimeBuilder;
use crate::kernel::{AgentError, AgentRuntime, TurnInput};
use crate::protocol::SessionId;
use crate::tool_invoker::PolicyContext;

/// Manages long-lived conversations on top of [`AgentRuntime`].
pub struct RuntimeManager {
    builder: RuntimeBuilder,
    system_prompt: String,
    conversations: Arc<Mutex<HashMap<ConversationId, ConversationHandle>>>,
    clock: Arc<dyn RuntimeClock>,
}

/// Errors that can occur while managing long-lived runtimes.
#[derive(Debug, Error)]
pub enum RuntimeManagerError {
    /// The requested conversation is not currently managed.
    #[error("conversation not found: {0}")]
    ConversationNotFound(String),
    /// The conversation actor stopped before replying.
    #[error("conversation actor stopped: {0}")]
    ConversationClosed(String),
    /// The runtime failed while restoring or running a turn.
    #[error(transparent)]
    Runtime(#[from] AgentError),
}

#[derive(Clone)]
struct ConversationHandle {
    sender: mpsc::Sender<ConversationCommand>,
    session_id: SessionId,
    shared: Arc<StdMutex<ConversationSharedState>>,
}

struct ConversationSharedState {
    last_activity: Instant,
    runtime_loaded: bool,
}

struct ConversationActor {
    builder: RuntimeBuilder,
    system_prompt: String,
    session_id: SessionId,
    runtime: Option<AgentRuntime>,
    needs_restore: bool,
    shared: Arc<StdMutex<ConversationSharedState>>,
    clock: Arc<dyn RuntimeClock>,
}

enum ConversationCommand {
    Dispatch {
        inbound: AccessInbound,
        reply: oneshot::Sender<Result<AccessOutbound, RuntimeManagerError>>,
    },
    Hibernate {
        reply: oneshot::Sender<()>,
    },
    Resume {
        reply: oneshot::Sender<()>,
    },
}

trait RuntimeClock: Send + Sync {
    fn now(&self) -> Instant;
}

struct SystemClock;

impl RuntimeClock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

impl RuntimeManager {
    /// Create a runtime manager with a shared builder and stable system prompt.
    #[must_use]
    pub fn new(builder: RuntimeBuilder, system_prompt: String) -> Self {
        Self::new_with_clock(builder, system_prompt, Arc::new(SystemClock))
    }

    /// Dispatch one inbound message through the managed runtime for its conversation.
    pub async fn dispatch(
        &self,
        inbound: AccessInbound,
    ) -> Result<AccessOutbound, RuntimeManagerError> {
        let conversation_id = inbound.conversation_id.clone();
        let handle = self.conversation_handle(&conversation_id).await;
        let (reply_tx, reply_rx) = oneshot::channel();
        handle
            .sender
            .send(ConversationCommand::Dispatch {
                inbound,
                reply: reply_tx,
            })
            .await
            .map_err(|_| RuntimeManagerError::ConversationClosed(conversation_id.0.clone()))?;
        reply_rx
            .await
            .map_err(|_| RuntimeManagerError::ConversationClosed(conversation_id.0.clone()))?
    }

    /// Hibernate one conversation by dropping its in-memory runtime.
    pub async fn hibernate(&self, id: &ConversationId) -> Result<(), RuntimeManagerError> {
        let handle = self
            .existing_handle(id)
            .await
            .ok_or_else(|| RuntimeManagerError::ConversationNotFound(id.0.clone()))?;
        let (reply_tx, reply_rx) = oneshot::channel();
        handle
            .sender
            .send(ConversationCommand::Hibernate { reply: reply_tx })
            .await
            .map_err(|_| RuntimeManagerError::ConversationClosed(id.0.clone()))?;
        reply_rx
            .await
            .map_err(|_| RuntimeManagerError::ConversationClosed(id.0.clone()))?;
        Ok(())
    }

    /// Recreate an in-memory runtime for one managed conversation.
    pub async fn resume(&self, id: &ConversationId) -> Result<(), RuntimeManagerError> {
        let handle = self
            .existing_handle(id)
            .await
            .ok_or_else(|| RuntimeManagerError::ConversationNotFound(id.0.clone()))?;
        let (reply_tx, reply_rx) = oneshot::channel();
        handle
            .sender
            .send(ConversationCommand::Resume { reply: reply_tx })
            .await
            .map_err(|_| RuntimeManagerError::ConversationClosed(id.0.clone()))?;
        reply_rx
            .await
            .map_err(|_| RuntimeManagerError::ConversationClosed(id.0.clone()))?;
        Ok(())
    }

    /// Hibernate all conversations that have been idle for at least `max_idle`.
    pub async fn hibernate_idle(
        &self,
        max_idle: Duration,
    ) -> Result<Vec<ConversationId>, RuntimeManagerError> {
        let now = self.clock.now();
        let eligible = {
            let conversations = self.conversations.lock().await;
            conversations
                .iter()
                .filter_map(|(conversation_id, handle)| {
                    let state = handle
                        .shared
                        .lock()
                        .expect("conversation shared state should not be poisoned");
                    let idle_for = now.saturating_duration_since(state.last_activity);
                    (state.runtime_loaded && idle_for >= max_idle)
                        .then(|| (conversation_id.clone(), handle.clone()))
                })
                .collect::<Vec<_>>()
        };

        let mut reclaimed = Vec::new();
        for (conversation_id, handle) in eligible {
            let (reply_tx, reply_rx) = oneshot::channel();
            handle
                .sender
                .send(ConversationCommand::Hibernate { reply: reply_tx })
                .await
                .map_err(|_| RuntimeManagerError::ConversationClosed(conversation_id.0.clone()))?;
            reply_rx
                .await
                .map_err(|_| RuntimeManagerError::ConversationClosed(conversation_id.0.clone()))?;
            reclaimed.push(conversation_id);
        }
        Ok(reclaimed)
    }

    fn new_with_clock(
        builder: RuntimeBuilder,
        system_prompt: String,
        clock: Arc<dyn RuntimeClock>,
    ) -> Self {
        Self {
            builder,
            system_prompt,
            conversations: Arc::new(Mutex::new(HashMap::new())),
            clock,
        }
    }

    async fn existing_handle(&self, id: &ConversationId) -> Option<ConversationHandle> {
        let conversations = self.conversations.lock().await;
        conversations.get(id).cloned()
    }

    async fn conversation_handle(&self, id: &ConversationId) -> ConversationHandle {
        let mut conversations = self.conversations.lock().await;
        if let Some(handle) = conversations.get(id)
            && !handle.sender.is_closed()
        {
            return handle.clone();
        }

        let session_id = conversations
            .get(id)
            .map(|handle| handle.session_id.clone())
            .unwrap_or_else(SessionId::new);
        let handle = self.spawn_conversation(session_id, false);
        conversations.insert(id.clone(), handle.clone());
        handle
    }

    fn spawn_conversation(&self, session_id: SessionId, needs_restore: bool) -> ConversationHandle {
        let (sender, receiver) = mpsc::channel(32);
        let shared = Arc::new(StdMutex::new(ConversationSharedState {
            last_activity: self.clock.now(),
            runtime_loaded: false,
        }));
        let actor = ConversationActor {
            builder: self.builder.clone(),
            system_prompt: self.system_prompt.clone(),
            session_id: session_id.clone(),
            runtime: None,
            needs_restore,
            shared: Arc::clone(&shared),
            clock: Arc::clone(&self.clock),
        };
        tokio::spawn(async move {
            actor.run(receiver).await;
        });

        ConversationHandle {
            sender,
            session_id,
            shared,
        }
    }
}

impl ConversationActor {
    async fn run(mut self, mut receiver: mpsc::Receiver<ConversationCommand>) {
        while let Some(command) = receiver.recv().await {
            match command {
                ConversationCommand::Dispatch { inbound, reply } => {
                    let result = self.handle_dispatch(inbound).await;
                    let _ = reply.send(result);
                }
                ConversationCommand::Hibernate { reply } => {
                    self.handle_hibernate();
                    let _ = reply.send(());
                }
                ConversationCommand::Resume { reply } => {
                    self.handle_resume();
                    let _ = reply.send(());
                }
            }
        }
    }

    async fn handle_dispatch(
        &mut self,
        inbound: AccessInbound,
    ) -> Result<AccessOutbound, RuntimeManagerError> {
        self.ensure_runtime();
        if self.needs_restore {
            self.runtime_mut().restore_from_tape().await?;
            self.needs_restore = false;
        }

        let content = self
            .runtime_mut()
            .run_turn_with_input(turn_input_from_inbound(&inbound))
            .await?;
        self.update_state(self.clock.now(), true);

        Ok(AccessOutbound {
            conversation_id: inbound.conversation_id,
            route: inbound.route,
            content,
            metadata: Value::Null,
        })
    }

    fn handle_hibernate(&mut self) {
        self.runtime = None;
        self.needs_restore = true;
        self.update_state(self.clock.now(), false);
    }

    fn handle_resume(&mut self) {
        self.ensure_runtime();
        self.needs_restore = true;
        self.update_state(self.clock.now(), true);
    }

    fn ensure_runtime(&mut self) {
        if self.runtime.is_none() {
            self.runtime = Some(
                self.builder
                    .build_with_session(self.system_prompt.clone(), self.session_id.clone()),
            );
        }
    }

    fn runtime_mut(&mut self) -> &mut AgentRuntime {
        self.runtime
            .as_mut()
            .expect("runtime should be present while processing a dispatch")
    }

    fn update_state(&self, last_activity: Instant, runtime_loaded: bool) {
        let mut shared = self
            .shared
            .lock()
            .expect("conversation shared state should not be poisoned");
        shared.last_activity = last_activity;
        shared.runtime_loaded = runtime_loaded;
    }
}

fn turn_input_from_inbound(inbound: &AccessInbound) -> TurnInput {
    TurnInput {
        text: inbound.text.clone(),
        policy_context: PolicyContext {
            organization_id: metadata_string(&inbound.metadata, "organization_id"),
            workspace_id: metadata_string(&inbound.metadata, "workspace_id"),
            conversation_id: Some(inbound.conversation_id.0.clone()),
        },
    }
}

fn metadata_string(metadata: &Value, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use serde_json::json;
    use tokio::sync::{Notify, RwLock};
    use tokio::task::yield_now;
    use tokio::time::timeout;

    use super::{
        ConversationHandle, ConversationSharedState, RuntimeClock, RuntimeManager,
        turn_input_from_inbound,
    };
    use crate::access::{AccessInbound, ConversationId, ReplyRoute};
    use crate::builder::RuntimeBuilder;
    use crate::context::{InMemoryTapeStore, SoulManager, TapeStore};
    use crate::error::ModelError;
    use crate::kernel::TurnInput;
    use crate::message::ChatRole;
    use crate::model::{
        ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
        ModelResponse, RequestContext,
    };
    use crate::tool_invoker::{
        AuthorizationDecision, ExecutionPolicy, PolicyContext, PreparedToolCall, SandboxKind,
        ToolAuthorizer, ToolCallRequest, ToolCatalog, ToolExecutor, ToolInvokeError,
        ToolInvokeResult,
    };
    use crate::tool_spec::ToolSpec;

    struct RequestSignal {
        started: AtomicBool,
        notify: Notify,
    }

    impl RequestSignal {
        fn new() -> Self {
            Self {
                started: AtomicBool::new(false),
                notify: Notify::new(),
            }
        }

        fn mark_started(&self) {
            self.started.store(true, Ordering::SeqCst);
            self.notify.notify_waiters();
        }

        async fn wait_started(&self) {
            if self.started.load(Ordering::SeqCst) {
                return;
            }
            self.notify.notified().await;
        }
    }

    struct RecordingModelProvider {
        requests: Mutex<Vec<ChatRequest>>,
        blockers: Mutex<HashMap<String, Arc<Notify>>>,
        signals: Mutex<HashMap<String, Arc<RequestSignal>>>,
        inflight: AtomicUsize,
        max_inflight: AtomicUsize,
    }

    impl RecordingModelProvider {
        fn new() -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                blockers: Mutex::new(HashMap::new()),
                signals: Mutex::new(HashMap::new()),
                inflight: AtomicUsize::new(0),
                max_inflight: AtomicUsize::new(0),
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .clone()
        }

        fn block_text(&self, text: &str) -> Arc<Notify> {
            let notify = Arc::new(Notify::new());
            self.blockers
                .lock()
                .expect("blockers lock should not be poisoned")
                .insert(text.to_string(), Arc::clone(&notify));
            notify
        }

        async fn wait_started(&self, text: &str) {
            let signal = self
                .signals
                .lock()
                .expect("signals lock should not be poisoned")
                .entry(text.to_string())
                .or_insert_with(|| Arc::new(RequestSignal::new()))
                .clone();
            signal.wait_started().await;
        }

        fn max_inflight(&self) -> usize {
            self.max_inflight.load(Ordering::SeqCst)
        }

        fn update_max_inflight(&self, current: usize) {
            let mut observed = self.max_inflight.load(Ordering::SeqCst);
            while current > observed {
                match self.max_inflight.compare_exchange(
                    observed,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => observed = actual,
                }
            }
        }

        fn signal_for(&self, text: &str) -> Arc<RequestSignal> {
            self.signals
                .lock()
                .expect("signals lock should not be poisoned")
                .entry(text.to_string())
                .or_insert_with(|| Arc::new(RequestSignal::new()))
                .clone()
        }
    }

    #[async_trait]
    impl ModelProvider for RecordingModelProvider {
        async fn complete(
            &self,
            request: ModelRequest,
            _context: RequestContext,
        ) -> Result<ModelResponse, ModelError> {
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

        async fn chat(
            &self,
            request: ChatRequest,
            _context: RequestContext,
        ) -> Result<ChatResponse, ModelError> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .push(request.clone());

            let user_text = request
                .messages
                .iter()
                .rev()
                .find(|message| message.role == ChatRole::User)
                .map(|message| message.content.clone())
                .unwrap_or_default();
            self.signal_for(&user_text).mark_started();

            let current_inflight = self.inflight.fetch_add(1, Ordering::SeqCst) + 1;
            self.update_max_inflight(current_inflight);
            let blocker = self
                .blockers
                .lock()
                .expect("blockers lock should not be poisoned")
                .get(&user_text)
                .cloned();
            if let Some(blocker) = blocker {
                blocker.notified().await;
            }
            self.inflight.fetch_sub(1, Ordering::SeqCst);

            Ok(ChatResponse {
                text: Some(format!("echo:{user_text}")),
                tool_calls: Vec::new(),
                usage: None,
            })
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities::default()
        }
    }

    struct NoopToolInvoker;

    impl ToolCatalog for NoopToolInvoker {
        fn lookup(&self, _name: &str) -> Option<&ToolSpec> {
            None
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            Vec::new()
        }
    }

    impl ToolAuthorizer for NoopToolInvoker {
        fn authorize(
            &self,
            request: &ToolCallRequest,
            _context: &PolicyContext,
        ) -> AuthorizationDecision {
            AuthorizationDecision::Prepared(PreparedToolCall {
                request: request.clone(),
                policy: ExecutionPolicy::Allow {
                    sandbox: SandboxKind::None,
                },
            })
        }
    }

    #[async_trait]
    impl ToolExecutor for NoopToolInvoker {
        async fn invoke(
            &self,
            prepared: PreparedToolCall,
            _cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Err(ToolInvokeError::NotFound(prepared.request.name))
        }
    }

    #[derive(Clone)]
    struct ManualClock {
        now: Arc<Mutex<Instant>>,
    }

    impl ManualClock {
        fn new(start: Instant) -> Self {
            Self {
                now: Arc::new(Mutex::new(start)),
            }
        }

        fn advance(&self, duration: Duration) {
            let mut now = self.now.lock().expect("clock lock should not be poisoned");
            *now += duration;
        }
    }

    impl RuntimeClock for ManualClock {
        fn now(&self) -> Instant {
            *self.now.lock().expect("clock lock should not be poisoned")
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

    fn stdio_route() -> ReplyRoute {
        ReplyRoute::new("stdio", "", json!({}))
    }

    fn inbound(conversation_id: &str, text: &str) -> AccessInbound {
        AccessInbound {
            conversation_id: ConversationId(conversation_id.to_string()),
            user_id: "user-1".to_string(),
            text: text.to_string(),
            route: stdio_route(),
            metadata: json!({"source": "test"}),
        }
    }

    async fn conversation_handle(
        manager: &RuntimeManager,
        id: &ConversationId,
    ) -> ConversationHandle {
        manager
            .conversations
            .lock()
            .await
            .get(id)
            .expect("conversation should exist")
            .clone()
    }

    fn shared_state(handle: &ConversationHandle) -> ConversationSharedState {
        let state = handle
            .shared
            .lock()
            .expect("conversation shared state should not be poisoned");
        ConversationSharedState {
            last_activity: state.last_activity,
            runtime_loaded: state.runtime_loaded,
        }
    }

    #[test]
    fn dispatch_maps_access_inbound_to_turn_input() {
        let inbound = AccessInbound {
            conversation_id: ConversationId("conv-42".to_string()),
            user_id: "user-1".to_string(),
            text: "hello".to_string(),
            route: stdio_route(),
            metadata: json!({
                "organization_id": "org-9",
                "workspace_id": "ws-7",
                "source": "test"
            }),
        };

        assert_eq!(
            turn_input_from_inbound(&inbound),
            TurnInput {
                text: "hello".to_string(),
                policy_context: PolicyContext {
                    organization_id: Some("org-9".to_string()),
                    workspace_id: Some("ws-7".to_string()),
                    conversation_id: Some("conv-42".to_string()),
                },
            }
        );
    }

    #[tokio::test]
    async fn same_conversation_is_processed_in_order() {
        let model = Arc::new(RecordingModelProvider::new());
        let first_gate = model.block_text("hello");
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model.clone(), tape_store, "ordered");
        let manager = Arc::new(RuntimeManager::new(builder, "system".to_string()));

        let first_manager = Arc::clone(&manager);
        let first_task = tokio::spawn(async move {
            first_manager
                .dispatch(inbound("conv-1", "hello"))
                .await
                .expect("first turn should succeed")
        });
        model.wait_started("hello").await;

        let second_manager = Arc::clone(&manager);
        let second_task = tokio::spawn(async move {
            second_manager
                .dispatch(inbound("conv-1", "follow up"))
                .await
                .expect("second turn should succeed")
        });

        for _ in 0..5 {
            yield_now().await;
        }
        assert_eq!(model.requests().len(), 1);

        first_gate.notify_waiters();
        let first = first_task.await.expect("first task should join");
        let second = second_task.await.expect("second task should join");

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
    async fn different_conversations_can_progress_concurrently() {
        let model = Arc::new(RecordingModelProvider::new());
        let gate_a = model.block_text("one");
        let gate_b = model.block_text("two");
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model.clone(), tape_store, "concurrent");
        let manager = Arc::new(RuntimeManager::new(builder, "system".to_string()));

        let task_a = {
            let manager = Arc::clone(&manager);
            tokio::spawn(async move {
                manager
                    .dispatch(inbound("conv-a", "one"))
                    .await
                    .expect("conversation a should succeed")
            })
        };
        let task_b = {
            let manager = Arc::clone(&manager);
            tokio::spawn(async move {
                manager
                    .dispatch(inbound("conv-b", "two"))
                    .await
                    .expect("conversation b should succeed")
            })
        };

        timeout(Duration::from_secs(1), model.wait_started("one"))
            .await
            .expect("conversation a should reach the model");
        timeout(Duration::from_secs(1), model.wait_started("two"))
            .await
            .expect("conversation b should reach the model");
        assert!(model.max_inflight() >= 2);

        gate_a.notify_waiters();
        gate_b.notify_waiters();

        let outbound_a = task_a.await.expect("conversation a task should join");
        let outbound_b = task_b.await.expect("conversation b task should join");
        assert_eq!(outbound_a.content, "echo:one");
        assert_eq!(outbound_b.content, "echo:two");

        let session_a = conversation_handle(&manager, &ConversationId("conv-a".to_string()))
            .await
            .session_id;
        let session_b = conversation_handle(&manager, &ConversationId("conv-b".to_string()))
            .await
            .session_id;
        assert_ne!(session_a, session_b);
    }

    #[tokio::test]
    async fn hibernate_drops_runtime_but_keeps_session_metadata() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model, tape_store, "hibernate");
        let manager = RuntimeManager::new(builder, "system".to_string());
        let conversation_id = ConversationId("conv-1".to_string());

        manager
            .dispatch(inbound("conv-1", "hello"))
            .await
            .expect("dispatch should succeed");
        let session_before = conversation_handle(&manager, &conversation_id)
            .await
            .session_id;
        manager
            .hibernate(&conversation_id)
            .await
            .expect("hibernate should succeed");

        let handle = conversation_handle(&manager, &conversation_id).await;
        let state = shared_state(&handle);
        assert_eq!(handle.session_id, session_before);
        assert!(!state.runtime_loaded);
    }

    #[tokio::test]
    async fn resume_restores_context_from_tape() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model.clone(), tape_store, "resume");
        let manager = RuntimeManager::new(builder, "system".to_string());
        let conversation_id = ConversationId("conv-restore".to_string());

        manager
            .dispatch(inbound("conv-restore", "first turn"))
            .await
            .expect("first turn should succeed");
        manager
            .hibernate(&conversation_id)
            .await
            .expect("hibernate should succeed");
        manager
            .resume(&conversation_id)
            .await
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
    async fn hibernate_idle_reclaims_only_expired_handles() {
        let model = Arc::new(RecordingModelProvider::new());
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::default());
        let builder = builder(model, tape_store, "idle");
        let clock = Arc::new(ManualClock::new(Instant::now()));
        let manager = RuntimeManager::new_with_clock(builder, "system".to_string(), clock.clone());

        manager
            .dispatch(inbound("conv-cold", "cold"))
            .await
            .expect("cold conversation should succeed");
        clock.advance(Duration::from_secs(300));
        manager
            .dispatch(inbound("conv-hot", "hot"))
            .await
            .expect("hot conversation should succeed");

        let reclaimed = manager
            .hibernate_idle(Duration::from_secs(60))
            .await
            .expect("idle collection should succeed");

        assert_eq!(reclaimed, vec![ConversationId("conv-cold".to_string())]);
        let cold_state = shared_state(
            &conversation_handle(&manager, &ConversationId("conv-cold".to_string())).await,
        );
        let hot_state = shared_state(
            &conversation_handle(&manager, &ConversationId("conv-hot".to_string())).await,
        );
        assert!(!cold_state.runtime_loaded);
        assert!(hot_state.runtime_loaded);
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
