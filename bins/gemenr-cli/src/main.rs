use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{
    ChatMessage, Config, ConfigError, InMemoryTapeStore, JsonlTapeStore, ModelError, ModelProvider,
    ModelRequest, RuntimeBuilder, SoulManager, TapeStore, ToolInvoker,
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
        Commands::Chat => {
            let provider = match AnthropicProvider::new(&config) {
                Ok(provider) => provider,
                Err(error) => {
                    eprintln!("Error: {error}");
                    std::process::exit(1);
                }
            };
            run_chat(&provider, &config).await;
        }
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

async fn run_chat(provider: &dyn ModelProvider, config: &Config) {
    tracing::info!(target: "gemenr::cli", "starting chat session");

    let stdin = io::stdin();
    let mut history = initial_history();
    let mut input = String::new();

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

        history.push(ChatMessage::user(trimmed));
        let request = match build_model_request(&history, config) {
            Ok(request) => request,
            Err(error) => {
                eprintln!("Error: {error}");
                history.pop();
                break;
            }
        };
        tracing::debug!(
            target: "gemenr::cli",
            selected_model = %config.model,
            remote_model = %request.model,
            message_count = history.len(),
            "sending request to model"
        );

        match provider.complete(request).await {
            Ok(response) => {
                println!("{}", response.content);
                tracing::debug!(
                    target: "gemenr::cli",
                    content_length = response.content.len(),
                    "received model response"
                );
                history.push(ChatMessage::assistant(response.content));
            }
            Err(error) => {
                tracing::warn!(
                    target: "gemenr::cli",
                    error = %error,
                    "model request failed"
                );
                eprintln!("Error: {}", display_model_error(&error));
                history.pop();
            }
        }
    }

    tracing::info!(target: "gemenr::cli", "chat session ended");
}

async fn run_task(task: &str, config: &Config) {
    let provider: Arc<dyn ModelProvider> = match AnthropicProvider::new(config) {
        Ok(provider) => Arc::new(provider),
        Err(error) => {
            eprintln!("Error creating provider: {error}");
            std::process::exit(1);
        }
    };

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

    let selected_model = match config.selected_model() {
        Ok(model) => model,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    let builder = RuntimeBuilder::new(provider, tools, soul, tape_store)
        .model_name(selected_model.model.clone())
        .max_tokens(selected_model.max_tokens.unwrap_or(4096))
        .tool_dispatcher(config.tool_dispatcher.clone());

    let system_prompt = format!(
        "You are an autonomous agent. Execute the following task:\n\n{task}\n\nUse the available tools to complete the task. When done, provide a summary of what you accomplished."
    );
    let mut runtime = builder.build(system_prompt);

    tracing::info!(target: "gemenr::cli", task = task, "starting task execution");

    match runtime.run_turn(task).await {
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

fn build_model_request(
    history: &[ChatMessage],
    config: &Config,
) -> Result<ModelRequest, ConfigError> {
    let selected_model = config.selected_model()?;

    Ok(ModelRequest {
        messages: history.to_vec(),
        model: selected_model.model.clone(),
        max_tokens: selected_model.max_tokens,
    })
}

fn initial_history() -> Vec<ChatMessage> {
    vec![ChatMessage::system(DEFAULT_SYSTEM_PROMPT)]
}

fn display_model_error(error: &ModelError) -> String {
    match error {
        ModelError::Auth(message) => format!("authentication failed: {message}"),
        ModelError::Api {
            status: 401 | 403,
            message,
        } => {
            format!("authentication failed: {message}")
        }
        _ => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use clap::{Parser, error::ErrorKind};

    use super::{
        Cli, Commands, DEFAULT_SYSTEM_PROMPT, build_model_request, display_model_error,
        initial_history,
    };
    use gemenr_core::{ChatMessage, Config, ModelConfig, ModelError, ProviderConfig, ProviderType};

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

    #[test]
    fn build_model_request_uses_selected_model_config() {
        let history = vec![
            ChatMessage::system("Be concise."),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
        ];
        let config = test_config();

        let request = build_model_request(&history, &config).expect("request should build");

        assert_eq!(request.messages, history);
        assert_eq!(request.model, "claude-haiku-4-5-20251001");
        assert_eq!(request.max_tokens, Some(256));
    }

    #[test]
    fn initial_history_contains_default_system_prompt() {
        let history = initial_history();

        assert_eq!(history, vec![ChatMessage::system(DEFAULT_SYSTEM_PROMPT)]);
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
}
