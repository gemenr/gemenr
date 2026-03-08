use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{
    AgentError, ApprovalHandler, Config, ConfigError, EventEnvelope, EventKind, EventSink,
    InMemoryTapeStore, JsonlTapeStore, ModelProvider, RuntimeBuilder, SoulManager,
    TapeStore, ToolInvoker,
};
use gemenr_tools::{ToolPlane, builtin};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

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
        /// The task description for the agent to execute.
        task: String,
    },
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
        Commands::Run { task } => run_task(&task, &config).await,
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
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                eprintln!("[tool_failed] {name}: {result}");
            }
            EventKind::ContextSummarized => {
                eprintln!("[context_summarized]");
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
    let mut runtime = builder.build(DEFAULT_SYSTEM_PROMPT.to_string());

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

        tracing::debug!(target: "gemenr::cli", "running chat turn through agent runtime");

        match runtime.run_turn(trimmed).await {
            Ok(response) => {
                println!("{response}");
            }
            Err(error) => {
                tracing::warn!(
                    target: "gemenr::cli",
                    error = %error,
                    "chat turn failed"
                );
                eprintln!("Error: {}", display_agent_error(&error));
            }
        }
    }

    tracing::info!(target: "gemenr::cli", "chat session ended");
}

async fn run_task(task: &str, config: &Config) {
    let builder = match build_runtime_builder(config, Some(4096)) {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    let system_prompt = format!(
        "You are an autonomous agent. Execute the following task:\n\n{task}\n\nUse the available tools to complete the task. When done, provide a summary of what you accomplished."
    );
    let mut runtime = builder.build(system_prompt);
    let cancellation_handle = runtime.cancellation_handle();
    let interrupt_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancellation_handle.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });

    tracing::info!(target: "gemenr::cli", task = task, "starting task execution");

    let result = runtime.run_turn(task).await;
    interrupt_task.abort();

    match result {
        Ok(result) => println!("\n{result}"),
        Err(error) => {
            eprintln!("Task failed: {error}");
            std::process::exit(1);
        }
    }

    tracing::info!(target: "gemenr::cli", "task execution completed");
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
    let provider: Arc<dyn ModelProvider> = Arc::new(AnthropicProvider::new(config)?);

    let workspace = current_workspace();
    let app_dir = workspace.join(".gemenr");
    let soul = load_soul_manager(&app_dir);

    let mut tool_plane = ToolPlane::new();
    builtin::register_builtin_tools(&mut tool_plane, Arc::clone(&soul));
    let tools: Arc<dyn ToolInvoker> = Arc::new(tool_plane);

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

#[cfg(test)]
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
        AgentError::Model(message) => message.clone(),
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
        Cli, Commands, DEFAULT_SYSTEM_PROMPT, configure_runtime_builder, display_agent_error,
        display_model_error,
    };
    use gemenr_core::{
        AgentError, ChatRequest, ChatResponse, Config, InMemoryTapeStore, ModelCapabilities,
        ModelConfig, ModelError, ModelProvider, ProviderConfig, ProviderType, SoulManager,
        TapeStore, ToolInvokeError, ToolInvokeResult, ToolInvoker, ToolSpec,
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
            Commands::Run { task } => assert_eq!(task, "list files"),
            Commands::Chat => panic!("expected run command"),
        }
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
        let error = AgentError::Model("provider exploded".to_string());

        assert_eq!(display_agent_error(&error), "provider exploded");
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
        let workspace =
            std::env::temp_dir().join(format!("gemenr-cli-test-soul-{}", std::process::id()));

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

    #[async_trait]
    impl ToolInvoker for StaticToolInvoker {
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
        ) -> gemenr_core::PolicyDecision {
            gemenr_core::PolicyDecision::Allow
        }

        async fn invoke(
            &self,
            _call_id: &str,
            _name: &str,
            _arguments: serde_json::Value,
            _cancelled: Arc<AtomicBool>,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Ok(ToolInvokeResult {
                content: String::new(),
                is_error: false,
            })
        }
    }
}
