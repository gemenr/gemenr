//! Unified test infrastructure for `gemenr-core`.
//!
//! Production code must not depend on this module. It centralizes reusable
//! mocks and helpers so test behavior stays aligned across modules.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::Notify;

use crate::error::ModelError;
use crate::message::ChatRole;
use crate::model::{
    ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
    ModelResponse, RequestContext,
};
use crate::tool_invoker::{
    AuthorizationDecision, ExecutionContext, PolicyContext, PreparedToolCall, ToolAuthorizer,
    ToolCallRequest, ToolCatalog, ToolExecutor, ToolInvokeError, ToolInvokeResult,
};
use crate::tool_spec::ToolSpec;

/// A model provider that records requests and can replay scripted responses.
///
/// When no scripted response is available, it falls back to echoing the latest
/// user message. This keeps the mock useful for both builder and runtime
/// manager tests.
pub(crate) struct RecordingModelProvider {
    responses: Mutex<VecDeque<ChatResponse>>,
    requests: Mutex<Vec<ChatRequest>>,
    capabilities: ModelCapabilities,
    blockers: Mutex<std::collections::HashMap<String, Arc<Notify>>>,
    signals: Mutex<std::collections::HashMap<String, Arc<RequestSignal>>>,
    inflight: AtomicUsize,
    max_inflight: AtomicUsize,
}

impl RecordingModelProvider {
    /// Create a provider with the supplied capabilities and no scripted
    /// responses.
    #[must_use]
    pub(crate) fn new(capabilities: ModelCapabilities) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            capabilities,
            blockers: Mutex::new(std::collections::HashMap::new()),
            signals: Mutex::new(std::collections::HashMap::new()),
            inflight: AtomicUsize::new(0),
            max_inflight: AtomicUsize::new(0),
        }
    }

    /// Return the recorded chat requests.
    #[must_use]
    pub(crate) fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .expect("requests lock should not be poisoned")
            .clone()
    }

    /// Block matching chat requests until the returned notifier is triggered.
    #[must_use]
    pub(crate) fn block_text(&self, text: &str) -> Arc<Notify> {
        let notify = Arc::new(Notify::new());
        self.blockers
            .lock()
            .expect("blockers lock should not be poisoned")
            .insert(text.to_string(), Arc::clone(&notify));
        notify
    }

    /// Wait until a request containing the supplied latest user text starts.
    pub(crate) async fn wait_started(&self, text: &str) {
        let signal = self.signal_for(text);
        signal.wait_started().await;
    }

    /// Return the highest number of concurrent in-flight chat requests seen so
    /// far.
    #[must_use]
    pub(crate) fn max_inflight(&self) -> usize {
        self.max_inflight.load(Ordering::SeqCst)
    }

    fn signal_for(&self, text: &str) -> Arc<RequestSignal> {
        self.signals
            .lock()
            .expect("signals lock should not be poisoned")
            .entry(text.to_string())
            .or_insert_with(|| Arc::new(RequestSignal::new()))
            .clone()
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
}

impl Default for RecordingModelProvider {
    fn default() -> Self {
        Self::new(ModelCapabilities::default())
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

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
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

        if let Some(response) = self
            .responses
            .lock()
            .expect("responses lock should not be poisoned")
            .pop_front()
        {
            return Ok(response);
        }

        Ok(ChatResponse {
            text: Some(format!("echo:{user_text}")),
            tool_calls: Vec::new(),
            usage: None,
        })
    }
}

/// A tool invoker with no registered tools.
pub(crate) struct NoopToolInvoker;

impl ToolCatalog for NoopToolInvoker {
    fn lookup(&self, _name: &str) -> Option<&ToolSpec> {
        None
    }

    fn list_specs(&self) -> &[ToolSpec] {
        &[]
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
            execution_context: ExecutionContext::new(()),
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

/// A tool invoker backed by a fixed set of tool specifications.
pub(crate) struct StaticToolInvoker {
    specs: Vec<ToolSpec>,
}

impl StaticToolInvoker {
    /// Create a static tool catalog from the supplied specifications.
    #[must_use]
    pub(crate) fn new(specs: Vec<ToolSpec>) -> Self {
        Self { specs }
    }
}

impl ToolCatalog for StaticToolInvoker {
    fn lookup(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|spec| spec.name == name)
    }

    fn list_specs(&self) -> &[ToolSpec] {
        &self.specs
    }
}

impl ToolAuthorizer for StaticToolInvoker {
    fn authorize(
        &self,
        request: &ToolCallRequest,
        _context: &PolicyContext,
    ) -> AuthorizationDecision {
        AuthorizationDecision::Prepared(PreparedToolCall {
            request: request.clone(),
            execution_context: ExecutionContext::new(()),
        })
    }
}

#[async_trait]
impl ToolExecutor for StaticToolInvoker {
    async fn invoke(
        &self,
        _prepared: PreparedToolCall,
        _cancelled: Arc<AtomicBool>,
    ) -> Result<ToolInvokeResult, ToolInvokeError> {
        Ok(ToolInvokeResult {
            content: String::new(),
            is_error: false,
        })
    }
}

/// A manually-advanced clock for runtime manager tests.
#[derive(Clone)]
pub(crate) struct MockClock {
    now: Arc<Mutex<Instant>>,
}

impl MockClock {
    /// Create a clock pinned to the supplied instant.
    #[must_use]
    pub(crate) fn new(start: Instant) -> Self {
        Self {
            now: Arc::new(Mutex::new(start)),
        }
    }

    /// Advance the clock by the supplied duration.
    pub(crate) fn advance(&self, duration: Duration) {
        let mut now = self.now.lock().expect("clock lock should not be poisoned");
        *now += duration;
    }

    /// Return the current instant.
    #[must_use]
    pub(crate) fn now(&self) -> Instant {
        *self.now.lock().expect("clock lock should not be poisoned")
    }
}

/// Create and return a unique temporary directory path for tests.
pub(crate) fn temp_dir(prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "gemenr-{prefix}-{}-{timestamp}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&directory).expect("temp directory should be created");
    directory
}

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
