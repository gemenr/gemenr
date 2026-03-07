use std::io::{self, Write};

use clap::{Parser, Subcommand};
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{ChatMessage, Config, ModelError, ModelProvider, ModelRequest};
use tracing_subscriber::EnvFilter;

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
    let provider = AnthropicProvider::new(&config);

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
    let mut history: Vec<ChatMessage> = Vec::new();
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
        tracing::debug!(
            target: "gemenr::cli",
            message_count = history.len(),
            model = %config.model,
            "sending request to model"
        );

        let request = ModelRequest {
            messages: history.clone(),
            model: config.model.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
        };

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
