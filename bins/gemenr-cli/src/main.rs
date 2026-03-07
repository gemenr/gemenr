use std::io::{self, Write};

use clap::{Parser, Subcommand};
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{ChatMessage, Config, ConfigError, ModelError, ModelProvider, ModelRequest};
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
    let provider = match AnthropicProvider::new(&config) {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Chat => run_chat(&provider, &config).await,
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

    use super::{DEFAULT_SYSTEM_PROMPT, build_model_request, display_model_error, initial_history};
    use gemenr_core::{ChatMessage, Config, ModelConfig, ModelError, ProviderConfig, ProviderType};

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
        }
    }
}
