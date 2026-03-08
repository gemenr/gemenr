use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{
    AccessAdapter, AccessError, AccessInbound, AccessOutbound, AccessRouter, AgentError,
    ApprovalHandler, Config, ConfigError, ConversationDriver, ConversationId, EventEnvelope,
    EventKind, EventSink, FallbackPlan, InMemoryTapeStore, JsonlTapeStore, ModelProvider,
    ModelRouter, ProviderType, ReplyRoute, RuntimeBuilder, SessionId, SoulManager, TapeStore,
    ToolInvoker,
};
use gemenr_tools::{ToolPlane, builtin};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

mod daemon;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// Gemenr — an LLM-based agent runtime.
#[derive(Debug, Parser)]
#[command(name = "gemenr", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Supported CLI subcommands.
#[derive(Debug, Subcommand)]
enum Commands {
    /// Start an interactive chat session with the LLM.
    Chat,

    /// Execute a task autonomously — the agent plans and calls tools to complete it.
    Run {
        /// Restore and continue a previously persisted session.
        #[arg(long)]
        session: Option<String>,

        /// The task description for the agent to execute.
        task: String,
    },

    /// Run the cron daemon.
    Daemon,
}

#[tokio::main]
async fn main() {
    init_tracing();

    let cli = Cli::parse();
    let config = match Config::load() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Chat => run_chat(&config).await,
        Commands::Run { session, task } => run_task(&task, session.as_deref(), &config).await,
        Commands::Daemon => run_daemon(&config).await,
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();
}

struct StdinApprovalHandler;

impl ApprovalHandler for StdinApprovalHandler {
    fn confirm(&self, message: &str) -> bool {
        eprintln!("{message} [y/N]");
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return false;
        }

        input.trim().eq_ignore_ascii_case("y")
    }
}

struct StdioAdapter {
    stdout: Mutex<Box<dyn Write + Send>>,
    stderr: Mutex<Box<dyn Write + Send>>,
}

impl StdioAdapter {
    fn new(stdout: Box<dyn Write + Send>, stderr: Box<dyn Write + Send>) -> Self {
        Self {
            stdout: Mutex::new(stdout),
            stderr: Mutex::new(stderr),
        }
    }

    fn route() -> ReplyRoute {
        ReplyRoute::new("stdio", "", serde_json::json!({}))
    }
}

impl Default for StdioAdapter {
    fn default() -> Self {
        Self::new(Box::new(io::stdout()), Box::new(io::stderr()))
    }
}

#[async_trait]
impl AccessAdapter for StdioAdapter {
    fn name(&self) -> &'static str {
        "stdio"
    }

    fn scheme(&self) -> &'static str {
        "stdio"
    }

    fn parse_route(&self, raw: &str) -> Result<Option<ReplyRoute>, AccessError> {
        Ok((raw == "stdio:").then(Self::route))
    }

    async fn send(&self, outbound: AccessOutbound) -> Result<(), AccessError> {
        if !outbound.route.has_scheme(self.scheme()) {
            return Err(AccessError::Delivery(
                "stdio adapter can only deliver stdio routes".to_string(),
            ));
        }
        let use_stderr = outbound
            .metadata
            .get("stream")
            .and_then(serde_json::Value::as_str)
            == Some("stderr");
        let mutex = if use_stderr {
            &self.stderr
        } else {
            &self.stdout
        };
        let mut writer = mutex
            .lock()
            .map_err(|_| AccessError::Delivery("stdio writer lock poisoned".to_string()))?;
        writeln!(writer, "{}", outbound.content)
            .map_err(|error| AccessError::Delivery(error.to_string()))?;
        writer
            .flush()
            .map_err(|error| AccessError::Delivery(error.to_string()))
    }
}

struct RuntimeConversationDriver {
    runtime: tokio::sync::Mutex<gemenr_core::AgentRuntime>,
    restore_on_first_turn: AtomicBool,
}

impl RuntimeConversationDriver {
    fn new(runtime: gemenr_core::AgentRuntime, restore_on_first_turn: bool) -> Self {
        Self {
            runtime: tokio::sync::Mutex::new(runtime),
            restore_on_first_turn: AtomicBool::new(restore_on_first_turn),
        }
    }
}

#[async_trait]
impl ConversationDriver for RuntimeConversationDriver {
    async fn handle(&self, inbound: AccessInbound) -> Result<AccessOutbound, AccessError> {
        let mut runtime = self.runtime.lock().await;
        if self.restore_on_first_turn.swap(false, Ordering::Relaxed) {
            runtime
                .restore_from_tape()
                .await
                .map_err(|error| AccessError::Driver(display_agent_error(&error)))?;
        }

        let content = runtime
            .run_turn(&inbound.text)
            .await
            .map_err(|error| AccessError::Driver(display_agent_error(&error)))?;

        Ok(AccessOutbound {
            conversation_id: inbound.conversation_id,
            route: inbound.route,
            content,
            metadata: serde_json::json!({}),
        })
    }
}

fn build_stdio_router() -> AccessRouter {
    AccessRouter::new().with_adapter(Arc::new(StdioAdapter::default()))
}

fn build_stdio_inbound(conversation_id: ConversationId, text: impl Into<String>) -> AccessInbound {
    AccessInbound {
        conversation_id,
        user_id: "stdio-user".to_string(),
        text: text.into(),
        route: StdioAdapter::route(),
        metadata: serde_json::json!({}),
    }
}

fn task_conversation_id(session: Option<&str>) -> ConversationId {
    match session {
        Some(session) => ConversationId(session.to_string()),
        None => ConversationId(SessionId::new().0),
    }
}

async fn handle_stdio_message(
    driver: &dyn ConversationDriver,
    router: &AccessRouter,
    inbound: AccessInbound,
) -> Result<(), AccessError> {
    let outbound = driver.handle(inbound).await?;
    router.deliver(outbound).await
}

struct StdioEventSink;

impl EventSink for StdioEventSink {
    fn publish(&self, event: &EventEnvelope) {
        match &event.kind {
            EventKind::UserInput => {
                if let Some(text) = event.payload.get("text").and_then(|value| value.as_str()) {
                    eprintln!("[user_input] {text}");
                }
            }
            EventKind::ModelResponse => {
                if let Some(text) = event.payload.get("text").and_then(|value| value.as_str()) {
                    eprintln!("[model_response] {text}");
                }
            }
            EventKind::AssistantToolCalls => {
                let text = event
                    .payload
                    .get("text")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let tool_count = event
                    .payload
                    .get("tool_calls")
                    .and_then(|value| value.as_array())
                    .map_or(0, Vec::len);
                eprintln!("[assistant_tool_calls] {tool_count} tool(s): {text}");
            }
            EventKind::ToolStarted => {
                let name = event
                    .payload
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("tool");
                eprintln!("[tool_started] {name}");
            }
            EventKind::ToolCompleted => {
                let name = event
                    .payload
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("tool");
                let result = event
                    .payload
                    .get("result")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                eprintln!("[tool_completed] {name}: {result}");
            }
            EventKind::ToolFailed => {
                let name = event
                    .payload
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("tool");
                let result = event
                    .payload
                    .get("result")
                    .or_else(|| event.payload.get("error"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                eprintln!("[tool_failed] {name}: {result}");
            }
            EventKind::ToolDenied => {
                let name = event
                    .payload
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("tool");
                let reason = event
                    .payload
                    .get("error")
                    .or_else(|| event.payload.get("reason"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                eprintln!("[tool_denied] {name}: {reason}");
            }
            EventKind::ToolTimedOut => {
                let name = event
                    .payload
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("tool");
                eprintln!("[tool_timed_out] {name}");
            }
            EventKind::ContextSummarized => {
                eprintln!("[context_summarized]");
            }
            EventKind::TurnCompleted => {
                eprintln!("[turn_completed]");
            }
            EventKind::TurnFailed => {
                let error = event
                    .payload
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                eprintln!("[turn_failed] {error}");
            }
            EventKind::AnchorCreated | EventKind::Custom(_) => {}
        }
    }
}

async fn run_chat(config: &Config) {
    tracing::info!(target: "gemenr::cli", "starting chat session");

    let stdin = io::stdin();
    let mut input = String::new();
    let builder = match build_runtime_builder(config, None) {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };
    let runtime = builder.build(DEFAULT_SYSTEM_PROMPT.to_string());
    let conversation_id = ConversationId(runtime.session_id().0.clone());
    let driver = RuntimeConversationDriver::new(runtime, false);
    let router = build_stdio_router();

    loop {
        eprint!("> ");
        if let Err(error) = io::stderr().flush() {
            eprintln!("Failed to flush prompt: {error}");
            break;
        }

        input.clear();
        match stdin.read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                eprintln!("Failed to read input: {error}");
                break;
            }
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        tracing::debug!(target: "gemenr::cli", "running chat turn through stdio access adapter");

        let inbound = build_stdio_inbound(conversation_id.clone(), trimmed.to_string());
        if let Err(error) = handle_stdio_message(&driver, &router, inbound).await {
            tracing::warn!(target: "gemenr::cli", error = %error, "chat turn failed");
            eprintln!("Error: {error}");
        }
    }

    tracing::info!(target: "gemenr::cli", "chat session ended");
}

async fn run_task(task: &str, session_id: Option<&str>, config: &Config) {
    let builder = match build_runtime_builder(config, Some(4096)) {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    let system_prompt = format!(
        "You are an autonomous agent. Execute the following task:

{task}

Use the available tools to complete the task. When done, provide a summary of what you accomplished."
    );
    let conversation_id = task_conversation_id(session_id);
    let runtime = match session_id {
        Some(session_id) => {
            builder.build_with_session(system_prompt, SessionId(session_id.to_string()))
        }
        None => builder.build(system_prompt),
    };

    let cancellation_handle = runtime.cancellation_handle();
    let driver = RuntimeConversationDriver::new(runtime, session_id.is_some());
    let router = build_stdio_router();
    let interrupt_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancellation_handle.store(true, Ordering::Relaxed);
        }
    });

    tracing::info!(target: "gemenr::cli", task = task, "starting task execution");

    let inbound = build_stdio_inbound(conversation_id, task.to_string());
    let result = driver.handle(inbound).await;
    interrupt_task.abort();

    match result {
        Ok(mut outbound) => {
            outbound.content = format!(
                "
{}",
                outbound.content
            );
            if let Err(error) = router.deliver(outbound).await {
                eprintln!("Task failed: {error}");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("Task failed: {error}");
            std::process::exit(1);
        }
    }

    tracing::info!(target: "gemenr::cli", "task execution completed");
}

async fn run_daemon(config: &Config) {
    let workspace = current_workspace();
    let app_dir = workspace.join(".gemenr");
    let soul = load_soul_manager(&app_dir);
    let tools = match build_tool_invoker(config, Arc::clone(&soul)) {
        Ok(tools) => tools,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };
    let tape_store: Arc<dyn TapeStore> = match JsonlTapeStore::new(app_dir.join("tapes")) {
        Ok(store) => Arc::new(store),
        Err(error) => {
            tracing::warn!(target: "gemenr::cli", error = %error, "failed to create tape store; using in-memory fallback");
            Arc::new(InMemoryTapeStore::new())
        }
    };
    let provider = match build_model_provider(config) {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };
    let builder = match configure_runtime_builder(
        RuntimeBuilder::new(provider, Arc::clone(&tools), soul, tape_store),
        config,
        Some(4096),
    ) {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    let mut router = AccessRouter::new().with_adapter(Arc::new(StdioAdapter::default()));
    if let Some(lark) = config.access.lark.clone() {
        router = router.with_adapter(Arc::new(daemon::LarkReportAdapter::new(lark)));
    }

    let daemon = daemon::CronDaemon::new(
        Arc::new(router),
        builder,
        tools,
        DEFAULT_SYSTEM_PROMPT.to_string(),
    );
    if let Err(error) = daemon.run(config).await {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn current_workspace() -> PathBuf {
    match std::env::current_dir() {
        Ok(path) => path,
        Err(error) => {
            tracing::warn!(
                target: "gemenr::cli",
                error = %error,
                "failed to resolve current workspace; using current relative directory"
            );
            PathBuf::from(".")
        }
    }
}

fn load_soul_manager(app_dir: &Path) -> Arc<RwLock<SoulManager>> {
    match SoulManager::load(app_dir) {
        Ok(soul) => Arc::new(RwLock::new(soul)),
        Err(error) => {
            tracing::warn!(
                target: "gemenr::cli",
                path = %app_dir.display(),
                error = %error,
                "failed to load SOUL.md; using temporary fallback"
            );
            let fallback_dir =
                std::env::temp_dir().join(format!("gemenr-soul-fallback-{}", std::process::id()));

            match SoulManager::load(&fallback_dir) {
                Ok(soul) => Arc::new(RwLock::new(soul)),
                Err(fallback_error) => {
                    eprintln!("Error initializing SOUL.md: {fallback_error}");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn build_runtime_builder(
    config: &Config,
    default_max_tokens: Option<u32>,
) -> Result<RuntimeBuilder, ConfigError> {
    let provider = build_model_provider(config)?;

    let workspace = current_workspace();
    let app_dir = workspace.join(".gemenr");
    let soul = load_soul_manager(&app_dir);
    let tools = build_tool_invoker(config, Arc::clone(&soul))?;

    let tape_store: Arc<dyn TapeStore> = match JsonlTapeStore::new(app_dir.join("tapes")) {
        Ok(store) => Arc::new(store),
        Err(error) => {
            tracing::warn!(
                target: "gemenr::cli",
                error = %error,
                "failed to create tape store; using in-memory fallback"
            );
            Arc::new(InMemoryTapeStore::new())
        }
    };

    let builder = RuntimeBuilder::new(provider, tools, soul, tape_store);
    configure_runtime_builder(builder, config, default_max_tokens)
}

fn build_tool_invoker(
    _config: &Config,
    soul: Arc<RwLock<SoulManager>>,
) -> Result<Arc<dyn ToolInvoker>, ConfigError> {
    let mut tool_plane = ToolPlane::new();
    builtin::register_builtin_tools(&mut tool_plane, soul);
    Ok(Arc::new(tool_plane))
}

fn build_model_provider(config: &Config) -> Result<Arc<dyn ModelProvider>, ConfigError> {
    let selected_model = config.selected_model()?;
    let selected_provider = config.selected_provider()?;

    let provider: Arc<dyn ModelProvider> = match selected_provider.provider_type {
        ProviderType::Anthropic => Arc::new(AnthropicProvider::new(config)?),
    };

    let mut router = ModelRouter::new(selected_model.provider.clone(), provider);

    if let Some(fallback) = &config.fallback {
        if fallback.primary != selected_model.provider {
            return Err(ConfigError::Invalid(format!(
                "fallback.primary `{}` must match selected model provider `{}`",
                fallback.primary, selected_model.provider
            )));
        }

        for backup in &fallback.backups {
            let provider_config = config.providers.get(backup).ok_or_else(|| {
                ConfigError::Invalid(format!(
                    "fallback.backups references unknown provider `{backup}`"
                ))
            })?;
            let provider: Arc<dyn ModelProvider> = match provider_config.provider_type {
                ProviderType::Anthropic => Arc::new(AnthropicProvider::from_parts(
                    selected_model,
                    provider_config,
                )?),
            };
            router.add_provider(backup.clone(), provider);
        }

        router.set_fallback_plan(FallbackPlan {
            primary: fallback.primary.clone(),
            backups: fallback.backups.clone(),
        })?;
    }

    Ok(Arc::new(router))
}

fn configure_runtime_builder(
    builder: RuntimeBuilder,
    config: &Config,
    default_max_tokens: Option<u32>,
) -> Result<RuntimeBuilder, ConfigError> {
    let selected_model = config.selected_model()?;

    let builder = builder
        .model_name(selected_model.model.clone())
        .tool_dispatcher(config.tool_dispatcher.clone())
        .approval_handler(Arc::new(StdinApprovalHandler))
        .event_sink(Arc::new(StdioEventSink));

    Ok(match selected_model.max_tokens.or(default_max_tokens) {
        Some(max_tokens) => builder.max_tokens(max_tokens),
        None => builder,
    })
}

fn display_model_error(error: &gemenr_core::ModelError) -> String {
    match error {
        gemenr_core::ModelError::Auth(message) => format!("authentication failed: {message}"),
        gemenr_core::ModelError::Api {
            status: 401 | 403,
            message,
        } => {
            format!("authentication failed: {message}")
        }
        _ => error.to_string(),
    }
}

fn display_agent_error(error: &AgentError) -> String {
    match error {
        AgentError::Model(model_error) => display_model_error(model_error),
        _ => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use async_trait::async_trait;
    use clap::{Parser, error::ErrorKind};
    use tokio::sync::RwLock;

    use super::{
        Cli, Commands, DEFAULT_SYSTEM_PROMPT, StdioAdapter, build_stdio_inbound,
        configure_runtime_builder, display_agent_error, display_model_error, handle_stdio_message,
        task_conversation_id,
    };
    use gemenr_core::{
        AccessAdapter, AccessError, AccessInbound, AccessOutbound, AccessRouter, AgentError,
        ChatRequest, ChatResponse, Config, ConversationDriver, ConversationId, InMemoryTapeStore,
        ModelCapabilities, ModelConfig, ModelError, ModelProvider, ProviderConfig, ProviderType,
        SoulManager, TapeStore, ToolInvokeError, ToolInvokeResult, ToolSpec,
    };

    #[test]
    fn cli_parses_chat_subcommand() {
        let cli = Cli::try_parse_from(["gemenr", "chat"]).expect("chat command should parse");

        assert!(matches!(cli.command, Commands::Chat));
    }

    #[test]
    fn cli_parses_run_subcommand() {
        let cli = Cli::try_parse_from(["gemenr", "run", "list files"]).expect("run should parse");

        match cli.command {
            Commands::Run { session, task } => {
                assert_eq!(session, None);
                assert_eq!(task, "list files");
            }
            Commands::Chat | Commands::Daemon => panic!("expected run command"),
        }
    }

    #[test]
    fn cli_parses_run_subcommand_with_session() {
        let cli = Cli::try_parse_from(["gemenr", "run", "--session", "session-123", "list files"])
            .expect("run with session should parse");

        match cli.command {
            Commands::Run { session, task } => {
                assert_eq!(session.as_deref(), Some("session-123"));
                assert_eq!(task, "list files");
            }
            Commands::Chat | Commands::Daemon => panic!("expected run command"),
        }
    }

    #[derive(Clone, Default)]
    struct SharedBuffer {
        inner: Arc<std::sync::Mutex<Vec<u8>>>,
    }

    impl SharedBuffer {
        fn as_string(&self) -> String {
            String::from_utf8(
                self.inner
                    .lock()
                    .expect("buffer lock should not be poisoned")
                    .clone(),
            )
            .expect("buffer should contain utf-8")
        }
    }

    impl std::io::Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.inner
                .lock()
                .expect("buffer lock should not be poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingDriver {
        seen: std::sync::Mutex<Vec<AccessInbound>>,
    }

    #[async_trait]
    impl ConversationDriver for RecordingDriver {
        async fn handle(&self, inbound: AccessInbound) -> Result<AccessOutbound, AccessError> {
            self.seen
                .lock()
                .expect("seen lock should not be poisoned")
                .push(inbound.clone());
            Ok(AccessOutbound {
                conversation_id: inbound.conversation_id,
                route: inbound.route,
                content: format!("driver:{}", inbound.text),
                metadata: serde_json::json!({}),
            })
        }
    }

    #[tokio::test]
    async fn chat_uses_stdio_access_contract() {
        let driver = RecordingDriver::default();
        let stdout = SharedBuffer::default();
        let stderr = SharedBuffer::default();
        let router = AccessRouter::new().with_adapter(Arc::new(StdioAdapter::new(
            Box::new(stdout.clone()),
            Box::new(stderr),
        )));
        let inbound = build_stdio_inbound(ConversationId("chat-1".to_string()), "hello");

        handle_stdio_message(&driver, &router, inbound.clone())
            .await
            .expect("stdio access flow should succeed");

        let seen = driver
            .seen
            .lock()
            .expect("seen lock should not be poisoned")
            .clone();
        assert_eq!(seen, vec![inbound]);
        assert_eq!(
            stdout.as_string(),
            "driver:hello
"
        );
    }

    #[test]
    fn run_session_keeps_stable_conversation_id() {
        assert_eq!(
            task_conversation_id(Some("session-123")),
            ConversationId("session-123".to_string())
        );
    }

    #[tokio::test]
    async fn stdio_adapter_writes_stdout_and_stderr() {
        let stdout = SharedBuffer::default();
        let stderr = SharedBuffer::default();
        let adapter = StdioAdapter::new(Box::new(stdout.clone()), Box::new(stderr.clone()));

        adapter
            .send(AccessOutbound {
                conversation_id: ConversationId("conv-1".to_string()),
                route: StdioAdapter::route(),
                content: "hello".to_string(),
                metadata: serde_json::json!({}),
            })
            .await
            .expect("stdout delivery should succeed");
        adapter
            .send(AccessOutbound {
                conversation_id: ConversationId("conv-1".to_string()),
                route: StdioAdapter::route(),
                content: "oops".to_string(),
                metadata: serde_json::json!({"stream": "stderr"}),
            })
            .await
            .expect("stderr delivery should succeed");

        assert_eq!(
            stdout.as_string(),
            "hello
"
        );
        assert_eq!(
            stderr.as_string(),
            "oops
"
        );
    }

    #[test]
    fn cli_requires_task_argument_for_run_subcommand() {
        let error = Cli::try_parse_from(["gemenr", "run"]).expect_err("run should require task");

        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[tokio::test]
    async fn configure_runtime_builder_uses_selected_model_config_for_chat() {
        let config = test_config();
        let model = Arc::new(RecordingModelProvider::new());
        let model_for_assertions = Arc::clone(&model);
        let tools = Arc::new(StaticToolInvoker);
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let soul = test_soul();
        let builder = gemenr_core::RuntimeBuilder::new(model, tools, soul, tape_store);

        let builder = configure_runtime_builder(builder, &config, None)
            .expect("runtime builder should configure");
        let mut runtime = builder.build(DEFAULT_SYSTEM_PROMPT.to_string());

        runtime
            .run_turn("Hello")
            .await
            .expect("turn should succeed");

        let requests = model_for_assertions.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "claude-haiku-4-5-20251001");
        assert_eq!(requests[0].max_tokens, Some(256));
    }

    #[tokio::test]
    async fn configure_runtime_builder_uses_fallback_max_tokens_for_run() {
        let config = test_config_without_max_tokens();
        let model = Arc::new(RecordingModelProvider::new());
        let model_for_assertions = Arc::clone(&model);
        let tools = Arc::new(StaticToolInvoker);
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let soul = test_soul();
        let builder = gemenr_core::RuntimeBuilder::new(model, tools, soul, tape_store);

        let builder = configure_runtime_builder(builder, &config, Some(4096))
            .expect("runtime builder should configure");
        let mut runtime = builder.build("system".to_string());

        runtime
            .run_turn("Hello")
            .await
            .expect("turn should succeed");

        let requests = model_for_assertions.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].max_tokens, Some(4096));
    }

    #[tokio::test]
    async fn configure_runtime_builder_uses_default_system_prompt_for_chat_runtime() {
        let config = test_config();
        let model = Arc::new(RecordingModelProvider::new());
        let model_for_assertions = Arc::clone(&model);
        let tools = Arc::new(StaticToolInvoker);
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let soul = test_soul();
        let builder = gemenr_core::RuntimeBuilder::new(model, tools, soul, tape_store);

        let builder = configure_runtime_builder(builder, &config, None)
            .expect("runtime builder should configure");
        let mut runtime = builder.build(DEFAULT_SYSTEM_PROMPT.to_string());

        runtime
            .run_turn("Hello")
            .await
            .expect("turn should succeed");

        let requests = model_for_assertions.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].messages[0]
                .content
                .contains(DEFAULT_SYSTEM_PROMPT)
        );
    }

    #[test]
    fn display_model_error_maps_auth_status_codes_to_user_friendly_message() {
        let unauthorized = ModelError::Api {
            status: 401,
            message: "bad key".to_string(),
        };
        let forbidden = ModelError::Api {
            status: 403,
            message: "request not allowed".to_string(),
        };

        assert_eq!(
            display_model_error(&unauthorized),
            "authentication failed: bad key"
        );
        assert_eq!(
            display_model_error(&forbidden),
            "authentication failed: request not allowed"
        );
    }

    #[test]
    fn display_model_error_leaves_non_auth_errors_unchanged() {
        let timeout = ModelError::Timeout;

        assert_eq!(display_model_error(&timeout), "request timed out");
    }

    #[test]
    fn display_agent_error_uses_model_message_verbatim() {
        let error = AgentError::Model(ModelError::Network("provider exploded".to_string()));

        assert_eq!(
            display_agent_error(&error),
            "network error: provider exploded"
        );
    }

    #[test]
    fn display_agent_error_leaves_non_model_errors_unchanged() {
        let error = AgentError::Cancelled;

        assert_eq!(display_agent_error(&error), "turn cancelled");
    }

    fn test_config() -> Config {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_string(),
            ProviderConfig {
                provider_type: ProviderType::Anthropic,
                api_key: "test-key".to_string(),
                api_endpoint: None,
            },
        );

        let mut models = HashMap::new();
        models.insert(
            "default".to_string(),
            ModelConfig {
                provider: "anthropic".to_string(),
                model: "claude-haiku-4-5-20251001".to_string(),
                max_tokens: Some(256),
            },
        );

        Config {
            model: "default".to_string(),
            providers,
            models,
            tool_dispatcher: "auto".to_string(),
            access: Default::default(),
            cron: Vec::new(),
            policy: Default::default(),
            fallback: None,
            mcp: Default::default(),
        }
    }

    fn test_config_without_max_tokens() -> Config {
        let mut config = test_config();
        config
            .models
            .get_mut("default")
            .expect("default model should exist")
            .max_tokens = None;
        config
    }

    fn test_soul() -> Arc<RwLock<SoulManager>> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!(
            "gemenr-cli-test-soul-{}-{timestamp}",
            std::process::id()
        ));

        std::fs::create_dir_all(&workspace).expect("test soul workspace should exist");

        Arc::new(RwLock::new(
            SoulManager::load(&workspace).expect("test SOUL.md should load"),
        ))
    }

    #[derive(Debug)]
    struct RecordingModelProvider {
        requests: std::sync::Mutex<Vec<ChatRequest>>,
    }

    impl RecordingModelProvider {
        fn new() -> Self {
            Self {
                requests: std::sync::Mutex::new(Vec::new()),
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
        async fn complete(
            &self,
            _request: gemenr_core::ModelRequest,
        ) -> Result<gemenr_core::ModelResponse, ModelError> {
            unreachable!("runtime builder tests should use chat(), not complete()")
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities::default()
        }

        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .push(request);

            Ok(ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[derive(Debug, Default)]
    struct StaticToolInvoker;

    impl gemenr_core::ToolCatalog for StaticToolInvoker {
        fn lookup(&self, _name: &str) -> Option<&ToolSpec> {
            None
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            Vec::new()
        }
    }

    impl gemenr_core::ToolAuthorizer for StaticToolInvoker {
        fn authorize(
            &self,
            request: &gemenr_core::ToolCallRequest,
            _context: &gemenr_core::PolicyContext,
        ) -> gemenr_core::AuthorizationDecision {
            gemenr_core::AuthorizationDecision::Prepared(gemenr_core::PreparedToolCall {
                request: request.clone(),
                policy: gemenr_core::ExecutionPolicy::Allow {
                    sandbox: gemenr_core::SandboxKind::None,
                },
            })
        }
    }

    #[async_trait]
    impl gemenr_core::ToolExecutor for StaticToolInvoker {
        async fn invoke(
            &self,
            _prepared: gemenr_core::PreparedToolCall,
            _cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Ok(ToolInvokeResult {
                content: String::new(),
                is_error: false,
            })
        }
    }
}
