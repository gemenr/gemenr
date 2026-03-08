use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use gemenr_core::{
    AccessAdapter, AccessError, AccessOutbound, AccessRouter, AgentError, Config, ConversationId,
    CronJobConfig, LarkConfig, ReplyRoute, RuntimeBuilder, ToolInvoker,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_cron_scheduler::{Job, JobScheduler, JobSchedulerError};
use tracing::{error, info, warn};

const LARK_TOKEN_URL: &str =
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal";
const LARK_SEND_URL: &str = "https://open.feishu.cn/open-apis/im/v1/messages";

/// Cron daemon that triggers task-mode runtimes and routes results.
#[derive(Clone)]
pub struct CronDaemon {
    access_router: Arc<AccessRouter>,
    runtime_builder: RuntimeBuilder,
    tools: Arc<dyn ToolInvoker>,
    system_prompt: String,
}

/// Errors produced while registering or running cron jobs.
#[derive(Debug, Error)]
pub enum DaemonError {
    /// Scheduler setup or registration failed.
    #[error(transparent)]
    Scheduler(#[from] JobSchedulerError),
    /// Result delivery failed.
    #[error(transparent)]
    Access(#[from] AccessError),
    /// Waiting for the shutdown signal failed.
    #[error("failed to wait for shutdown signal: {0}")]
    Signal(#[from] std::io::Error),
    /// One runtime execution failed.
    #[error(transparent)]
    Runtime(#[from] AgentError),
}

#[derive(Debug, Clone)]
struct CachedTenantToken {
    token: String,
    refresh_at: Instant,
}

impl CachedTenantToken {
    fn from_ttl(token: String, ttl_seconds: u64, now: Instant) -> Self {
        let refresh_after = ttl_seconds.saturating_sub(60);
        Self {
            token,
            refresh_at: now + Duration::from_secs(refresh_after),
        }
    }

    fn should_refresh(&self, now: Instant) -> bool {
        now >= self.refresh_at
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponseBody {
    tenant_access_token: String,
    expire: u64,
}

/// Thin Lark adapter used by the daemon for report delivery.
pub struct LarkReportAdapter {
    http: Client,
    config: LarkConfig,
    tenant_token: RwLock<Option<CachedTenantToken>>,
}

impl LarkReportAdapter {
    /// Create a Lark report adapter from access config.
    #[must_use]
    pub fn new(config: LarkConfig) -> Self {
        Self {
            http: Client::new(),
            config,
            tenant_token: RwLock::new(None),
        }
    }

    fn route(chat_id: impl Into<String>, thread_id: Option<String>) -> ReplyRoute {
        let metadata = match thread_id {
            Some(thread_id) => json!({ "thread_id": thread_id }),
            None => json!({}),
        };
        ReplyRoute::new("lark", chat_id.into(), metadata)
    }

    async fn refresh_tenant_token(&self) -> Result<String, AccessError> {
        let now = Instant::now();
        if let Some(cached) = self.tenant_token.read().await.clone()
            && !cached.should_refresh(now)
        {
            return Ok(cached.token);
        }

        let response = self
            .http
            .post(LARK_TOKEN_URL)
            .json(&json!({
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            }))
            .send()
            .await
            .map_err(|error| AccessError::Delivery(error.to_string()))?
            .error_for_status()
            .map_err(|error| AccessError::Delivery(error.to_string()))?;
        let body: TokenResponseBody = response
            .json()
            .await
            .map_err(|error| AccessError::Delivery(error.to_string()))?;
        let cached =
            CachedTenantToken::from_ttl(body.tenant_access_token.clone(), body.expire, now);
        *self.tenant_token.write().await = Some(cached.clone());
        Ok(cached.token)
    }
}

#[async_trait]
impl AccessAdapter for LarkReportAdapter {
    fn name(&self) -> &'static str {
        "lark"
    }

    fn scheme(&self) -> &'static str {
        "lark"
    }

    fn parse_route(&self, raw: &str) -> Result<Option<ReplyRoute>, AccessError> {
        let Some(target) = raw.strip_prefix("lark:") else {
            return Ok(None);
        };

        if target.is_empty() {
            return Err(AccessError::InvalidRoute(raw.to_string()));
        }

        let (chat_id, thread_id) = match target.split_once('/') {
            Some((chat_id, thread_id)) if !chat_id.is_empty() && !thread_id.is_empty() => {
                (chat_id.to_string(), Some(thread_id.to_string()))
            }
            Some(_) => return Err(AccessError::InvalidRoute(raw.to_string())),
            None => (target.to_string(), None),
        };

        Ok(Some(Self::route(chat_id, thread_id)))
    }

    async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
        if !outbound.route.has_scheme(self.scheme()) {
            return Err(AccessError::Delivery(
                "lark adapter can only deliver lark routes".to_string(),
            ));
        }

        let chat_id = outbound.route.target.clone();
        if chat_id.is_empty() {
            return Err(AccessError::Delivery(
                "lark route is missing chat_id".to_string(),
            ));
        }
        let thread_id = outbound
            .route
            .metadata_string("thread_id")
            .map(str::to_string);

        let token = self.refresh_tenant_token().await?;
        let mut body = json!({
            "receive_id": chat_id,
            "msg_type": "text",
            "content": serde_json::to_string(&json!({"text": outbound.content}))
                .map_err(|error| AccessError::Delivery(error.to_string()))?,
        });
        if let Some(thread_id) = thread_id {
            body["thread_id"] = json!(thread_id);
        }

        self.http
            .post(LARK_SEND_URL)
            .bearer_auth(token)
            .query(&[("receive_id_type", "chat_id")])
            .json(&body)
            .send()
            .await
            .map_err(|error| AccessError::Delivery(error.to_string()))?
            .error_for_status()
            .map_err(|error| AccessError::Delivery(error.to_string()))?;
        Ok(())
    }
}

impl CronDaemon {
    /// Create a daemon with a shared base runtime builder and access router.
    #[must_use]
    pub fn new(
        access_router: Arc<AccessRouter>,
        runtime_builder: RuntimeBuilder,
        tools: Arc<dyn ToolInvoker>,
        system_prompt: String,
    ) -> Self {
        Self {
            access_router,
            runtime_builder,
            tools,
            system_prompt,
        }
    }

    /// Start the scheduler, register jobs from config, and wait until shutdown.
    pub async fn run(&self, config: &Config) -> Result<(), DaemonError> {
        let scheduler = JobScheduler::new().await?;
        self.register_jobs(&scheduler, &config.cron).await?;
        scheduler.start().await?;
        tokio::signal::ctrl_c().await?;
        Ok(())
    }

    /// Register all configured cron jobs with the scheduler.
    pub async fn register_jobs(
        &self,
        scheduler: &JobScheduler,
        jobs: &[CronJobConfig],
    ) -> Result<usize, DaemonError> {
        for job_config in jobs {
            let daemon = self.clone();
            let job = job_config.clone();
            let schedule = job.schedule.clone();
            scheduler
                .add(
                    Job::new_async(schedule.as_str(), move |_, _| {
                        let daemon = daemon.clone();
                        let job = job.clone();
                        Box::pin(async move {
                            if let Err(error) = daemon.execute_job(&job).await {
                                error!(job = %job.name, error = %error, "cron job execution failed");
                            }
                        })
                    })
                    .expect("cron expression should already be validated by config"),
                )
                .await?;
        }

        Ok(jobs.len())
    }

    /// Execute one cron job immediately.
    pub async fn execute_job(&self, job: &CronJobConfig) -> Result<(), DaemonError> {
        let builder = self.builder_for_job(job);
        let mut runtime = builder.build(task_system_prompt(&self.system_prompt, &job.prompt));
        match runtime.run_turn(&job.prompt).await {
            Ok(content) => {
                info!(job = %job.name, "cron job completed");
                self.report(job, content, false).await?;
                Ok(())
            }
            Err(error) => {
                warn!(job = %job.name, error = %error, "cron job failed");
                let message = format!("Cron job `{}` failed: {error}", job.name);
                self.report(job, message, true).await?;
                Err(DaemonError::Runtime(error))
            }
        }
    }

    fn builder_for_job(&self, job: &CronJobConfig) -> RuntimeBuilder {
        match job.tools.as_deref() {
            Some(allowed) => {
                self.runtime_builder
                    .clone()
                    .with_tools(gemenr_tools::allowlist_tool_invoker(
                        Arc::clone(&self.tools),
                        allowed,
                    ))
            }
            None => self.runtime_builder.clone(),
        }
    }

    async fn report(
        &self,
        job: &CronJobConfig,
        content: String,
        is_error: bool,
    ) -> Result<(), AccessError> {
        let Some(route) = job.report_to.as_deref() else {
            return Ok(());
        };
        let route = self.access_router.parse_route(route)?;
        self.access_router
            .deliver(AccessOutbound {
                conversation_id: ConversationId(format!("cron:{}", job.name)),
                route,
                content,
                metadata: if is_error {
                    json!({"job": job.name, "stream": "stderr"})
                } else {
                    json!({"job": job.name})
                },
            })
            .await
    }
}

fn task_system_prompt(default_prompt: &str, task: &str) -> String {
    format!(
        "{default_prompt}\n\nExecute the following task:\n\n{task}\n\nUse the available tools to complete the task. When done, provide a summary of what you accomplished."
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    use async_trait::async_trait;
    use gemenr_core::model::{
        ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
        ModelResponse,
    };
    use gemenr_core::{
        AccessAdapter, AccessError, AccessOutbound, AccessRouter, CronJobConfig, InMemoryTapeStore,
        PolicyContext, ReplyRoute, RuntimeBuilder, SandboxKind, SoulManager, TapeStore,
        ToolInvokeError, ToolInvokeResult, ToolInvoker, ToolSpec,
    };
    use serde_json::json;
    use tokio::sync::RwLock;
    use tokio_cron_scheduler::JobScheduler;

    use super::{CronDaemon, task_system_prompt};

    struct RecordingAdapter {
        scheme: &'static str,
        messages: Mutex<Vec<AccessOutbound>>,
    }

    impl RecordingAdapter {
        fn new(scheme: &'static str) -> Self {
            Self {
                scheme,
                messages: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl AccessAdapter for RecordingAdapter {
        fn name(&self) -> &'static str {
            self.scheme
        }

        fn scheme(&self) -> &'static str {
            self.scheme
        }

        fn parse_route(&self, raw: &str) -> Result<Option<ReplyRoute>, AccessError> {
            if self.scheme() == "stdio" {
                return Ok((raw == "stdio:").then(|| ReplyRoute::new("stdio", "", json!({}))));
            }

            let prefix = format!("{}:", self.scheme());
            let Some(target) = raw.strip_prefix(&prefix) else {
                return Ok(None);
            };
            if target.is_empty() {
                return Err(AccessError::InvalidRoute(raw.to_string()));
            }
            Ok(Some(ReplyRoute::new(self.scheme(), target, json!({}))))
        }

        async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
            self.messages.lock().expect("messages lock").push(outbound);
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingModelProvider {
        requests: Mutex<Vec<ChatRequest>>,
        fail: bool,
    }

    #[async_trait]
    impl ModelProvider for RecordingModelProvider {
        async fn complete(
            &self,
            request: ModelRequest,
        ) -> Result<ModelResponse, gemenr_core::ModelError> {
            Ok(ModelResponse {
                content: format!("{} messages", request.messages.len()),
                finish_reason: FinishReason::Stop,
            })
        }

        async fn chat(
            &self,
            request: ChatRequest,
        ) -> Result<ChatResponse, gemenr_core::ModelError> {
            self.requests
                .lock()
                .expect("requests lock")
                .push(request.clone());
            if self.fail {
                return Err(gemenr_core::ModelError::Timeout);
            }
            Ok(ChatResponse {
                text: Some("job ok".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                native_tool_calling: true,
                ..ModelCapabilities::default()
            }
        }
    }

    struct StaticToolInvoker {
        specs: Vec<ToolSpec>,
    }

    impl gemenr_core::ToolCatalog for StaticToolInvoker {
        fn lookup(&self, name: &str) -> Option<&ToolSpec> {
            self.specs.iter().find(|spec| spec.name == name)
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            self.specs.clone()
        }
    }

    impl gemenr_core::ToolAuthorizer for StaticToolInvoker {
        fn authorize(
            &self,
            request: &gemenr_core::ToolCallRequest,
            _context: &PolicyContext,
        ) -> gemenr_core::AuthorizationDecision {
            gemenr_core::AuthorizationDecision::Prepared(gemenr_core::PreparedToolCall {
                request: request.clone(),
                policy: gemenr_core::ExecutionPolicy::Allow {
                    sandbox: SandboxKind::None,
                },
            })
        }
    }

    #[async_trait]
    impl gemenr_core::ToolExecutor for StaticToolInvoker {
        async fn invoke(
            &self,
            prepared: gemenr_core::PreparedToolCall,
            _cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Ok(ToolInvokeResult {
                content: prepared.request.name,
                is_error: false,
            })
        }
    }

    #[derive(Default)]
    struct RecordingTapeStore {
        sessions: Mutex<HashSet<String>>,
        inner: InMemoryTapeStore,
    }

    #[async_trait]
    impl TapeStore for RecordingTapeStore {
        async fn append(
            &self,
            session_id: &gemenr_core::SessionId,
            event: gemenr_core::EventEnvelope,
        ) -> Result<(), gemenr_core::TapeError> {
            self.sessions
                .lock()
                .expect("sessions lock")
                .insert(session_id.0.clone());
            self.inner.append(session_id, event).await
        }

        async fn load_since_anchor(
            &self,
            session_id: &gemenr_core::SessionId,
        ) -> Result<Vec<gemenr_core::EventEnvelope>, gemenr_core::TapeError> {
            self.inner.load_since_anchor(session_id).await
        }

        async fn load_last_anchor(
            &self,
            session_id: &gemenr_core::SessionId,
        ) -> Result<Option<gemenr_core::AnchorEntry>, gemenr_core::TapeError> {
            self.inner.load_last_anchor(session_id).await
        }

        async fn load_all(
            &self,
            session_id: &gemenr_core::SessionId,
        ) -> Result<Vec<gemenr_core::EventEnvelope>, gemenr_core::TapeError> {
            self.inner.load_all(session_id).await
        }
    }

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: name.to_string(),
            input_schema: json!({"type": "object"}),
            risk_level: gemenr_core::RiskLevel::Low,
        }
    }

    fn builder(
        model: Arc<RecordingModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        tape_store: Arc<dyn TapeStore>,
    ) -> RuntimeBuilder {
        let workspace =
            std::env::temp_dir().join(format!("gemenr-daemon-tests-{}", std::process::id()));
        std::fs::create_dir_all(&workspace).expect("workspace exists");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&workspace).expect("soul loads"),
        ));
        RuntimeBuilder::new(model, tools, soul, tape_store).model_name("test-model".to_string())
    }

    fn job(name: &str) -> CronJobConfig {
        CronJobConfig {
            name: name.to_string(),
            schedule: "*/5 * * * * *".to_string(),
            prompt: format!("run {name}"),
            tools: None,
            report_to: None,
        }
    }

    #[tokio::test]
    async fn registers_all_cron_jobs() {
        let scheduler = JobScheduler::new()
            .await
            .expect("scheduler should initialize");
        let tools: Arc<dyn ToolInvoker> = Arc::new(StaticToolInvoker {
            specs: vec![spec("shell")],
        });
        let daemon = CronDaemon::new(
            Arc::new(AccessRouter::new()),
            builder(
                Arc::new(RecordingModelProvider::default()),
                tools.clone(),
                Arc::new(InMemoryTapeStore::new()),
            ),
            tools,
            "system".to_string(),
        );

        let count = daemon
            .register_jobs(&scheduler, &[job("a"), job("b")])
            .await
            .expect("jobs should register");
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn each_execution_uses_independent_runtime_session() {
        let tape_store = Arc::new(RecordingTapeStore::default());
        let tools: Arc<dyn ToolInvoker> = Arc::new(StaticToolInvoker {
            specs: vec![spec("shell")],
        });
        let daemon = CronDaemon::new(
            Arc::new(AccessRouter::new()),
            builder(
                Arc::new(RecordingModelProvider::default()),
                tools.clone(),
                tape_store.clone(),
            ),
            tools,
            "system".to_string(),
        );

        daemon
            .execute_job(&job("a"))
            .await
            .expect("first job should succeed");
        daemon
            .execute_job(&job("b"))
            .await
            .expect("second job should succeed");

        assert_eq!(tape_store.sessions.lock().expect("sessions lock").len(), 2);
    }

    #[tokio::test]
    async fn routes_reports_to_stdio_and_lark() {
        let stdio = Arc::new(RecordingAdapter::new("stdio"));
        let lark = Arc::new(RecordingAdapter::new("lark"));
        let router = AccessRouter::new()
            .with_adapter(stdio.clone())
            .with_adapter(lark.clone());
        let tools: Arc<dyn ToolInvoker> = Arc::new(StaticToolInvoker {
            specs: vec![spec("shell")],
        });
        let daemon = CronDaemon::new(
            Arc::new(router),
            builder(
                Arc::new(RecordingModelProvider::default()),
                tools.clone(),
                Arc::new(InMemoryTapeStore::new()),
            ),
            tools,
            "system".to_string(),
        );

        let mut stdio_job = job("stdio");
        stdio_job.report_to = Some("stdio:".to_string());
        daemon
            .execute_job(&stdio_job)
            .await
            .expect("stdio job should succeed");

        let mut lark_job = job("lark");
        lark_job.report_to = Some("lark:oc_test".to_string());
        daemon
            .execute_job(&lark_job)
            .await
            .expect("lark job should succeed");

        assert_eq!(stdio.messages.lock().expect("stdio messages").len(), 1);
        assert_eq!(lark.messages.lock().expect("lark messages").len(), 1);
    }

    #[tokio::test]
    async fn allowlist_limits_visible_tools() {
        let model = Arc::new(RecordingModelProvider::default());
        let tools: Arc<dyn ToolInvoker> = Arc::new(StaticToolInvoker {
            specs: vec![spec("shell"), spec("fs.read")],
        });
        let daemon = CronDaemon::new(
            Arc::new(AccessRouter::new()),
            builder(
                model.clone(),
                tools.clone(),
                Arc::new(InMemoryTapeStore::new()),
            ),
            tools,
            "system".to_string(),
        );

        let mut only_shell = job("allowlisted");
        only_shell.tools = Some(vec!["shell".to_string()]);
        daemon
            .execute_job(&only_shell)
            .await
            .expect("job should succeed");

        let requests = model.requests.lock().expect("requests lock");
        let request = &requests[0];
        if let Some(tools) = request.tools.as_ref() {
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name, "shell");
        } else {
            let rendered = request
                .messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join(
                    "
",
                );
            assert!(rendered.contains("shell"));
            assert!(!rendered.contains("fs.read"));
        }
    }

    #[tokio::test]
    async fn failures_emit_alerts_on_same_route() {
        let stdio = Arc::new(RecordingAdapter::new("stdio"));
        let router = AccessRouter::new().with_adapter(stdio.clone());
        let model = Arc::new(RecordingModelProvider {
            fail: true,
            ..RecordingModelProvider::default()
        });
        let tools: Arc<dyn ToolInvoker> = Arc::new(StaticToolInvoker {
            specs: vec![spec("shell")],
        });
        let daemon = CronDaemon::new(
            Arc::new(router),
            builder(model, tools.clone(), Arc::new(InMemoryTapeStore::new())),
            tools,
            "system".to_string(),
        );

        let mut failing = job("failing");
        failing.report_to = Some("stdio:".to_string());
        let error = daemon
            .execute_job(&failing)
            .await
            .expect_err("job should fail");
        assert!(
            error.to_string().contains("request timed out")
                || error.to_string().contains("timed out")
        );
        let messages = stdio.messages.lock().expect("messages lock");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].metadata["stream"], json!("stderr"));
    }

    #[test]
    fn task_prompt_wraps_prompt_text() {
        let prompt = task_system_prompt("system", "do work");
        assert!(prompt.contains("system"));
        assert!(prompt.contains("do work"));
    }
}
