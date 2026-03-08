use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{
    AccessError, AccessInbound, AccessOutbound, Config, ConfigError, ConversationDriver,
    DenyAllApprovals, InMemoryTapeStore, JsonlTapeStore, ModelProvider, ModelRouter, ProviderType,
    RuntimeBuilder, RuntimeManager, SoulManager, TapeStore,
};
use gemenr_tools::{ToolPlane, builtin};
use tokio::sync::{Mutex, RwLock};
use tracing_subscriber::EnvFilter;

mod lark;
mod service;

use lark::LarkAdapter;
use service::{IdleCollector, LarkService};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

#[tokio::main]
async fn main() {
    init_tracing();

    let config = match Config::load() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    let lark_config = match config.access.lark.clone() {
        Some(config) => config,
        None => {
            eprintln!("Error: missing [access.lark] configuration");
            std::process::exit(1);
        }
    };

    let builder = match build_runtime_builder(&config) {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };
    let manager = Arc::new(Mutex::new(RuntimeManager::new(
        builder,
        DEFAULT_SYSTEM_PROMPT.to_string(),
    )));
    let driver = Arc::new(RuntimeManagerDriver::new(manager));
    let adapter = Arc::new(LarkAdapter::new(lark_config));
    let service = LarkService::new(adapter, driver.clone(), driver)
        .idle_timeout(Duration::from_secs(600))
        .reconnect_backoff(Duration::from_secs(1), Duration::from_secs(30));

    if let Err(error) = service.run().await {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();
}

struct RuntimeManagerDriver {
    manager: Arc<Mutex<RuntimeManager>>,
}

impl RuntimeManagerDriver {
    fn new(manager: Arc<Mutex<RuntimeManager>>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl ConversationDriver for RuntimeManagerDriver {
    async fn handle(&self, inbound: AccessInbound) -> Result<AccessOutbound, AccessError> {
        self.manager
            .lock()
            .await
            .dispatch(inbound)
            .await
            .map_err(|error| AccessError::Driver(error.to_string()))
    }
}

#[async_trait]
impl IdleCollector for RuntimeManagerDriver {
    async fn hibernate_idle(&self, max_idle: Duration) -> usize {
        self.manager
            .lock()
            .await
            .hibernate_idle(max_idle)
            .map(|ids| ids.len())
            .unwrap_or(0)
    }
}

fn current_workspace() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn load_soul_manager(app_dir: &Path) -> Arc<RwLock<SoulManager>> {
    match SoulManager::load(app_dir) {
        Ok(soul) => Arc::new(RwLock::new(soul)),
        Err(_) => {
            let fallback_dir = std::env::temp_dir()
                .join(format!("gemenr-im-soul-fallback-{}", std::process::id()));
            Arc::new(RwLock::new(
                SoulManager::load(&fallback_dir).expect("fallback SOUL.md should load"),
            ))
        }
    }
}

fn build_runtime_builder(config: &Config) -> Result<RuntimeBuilder, ConfigError> {
    let provider = build_model_provider(config)?;
    let workspace = current_workspace();
    let app_dir = workspace.join(".gemenr");
    let soul = load_soul_manager(&app_dir);

    let mut tool_plane = ToolPlane::new();
    builtin::register_builtin_tools(&mut tool_plane, Arc::clone(&soul));
    let tools = Arc::new(tool_plane);

    let tape_store: Arc<dyn TapeStore> = match JsonlTapeStore::new(app_dir.join("tapes")) {
        Ok(store) => Arc::new(store),
        Err(_) => Arc::new(InMemoryTapeStore::new()),
    };

    let selected_model = config.selected_model()?;
    Ok(RuntimeBuilder::new(provider, tools, soul, tape_store)
        .model_name(selected_model.model.clone())
        .tool_dispatcher(config.tool_dispatcher.clone())
        .approval_handler(Arc::new(DenyAllApprovals)))
}

fn build_model_provider(config: &Config) -> Result<Arc<dyn ModelProvider>, ConfigError> {
    let selected_model = config.selected_model()?;
    let selected_provider = config.selected_provider()?;

    let provider: Arc<dyn ModelProvider> = match selected_provider.provider_type {
        ProviderType::Anthropic => Arc::new(AnthropicProvider::new(config)?),
    };
    let mut router = ModelRouter::new(selected_model.provider.clone(), provider);

    if let Some(fallback) = &config.fallback {
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
        router.set_fallback_plan(gemenr_core::FallbackPlan {
            primary: fallback.primary.clone(),
            backups: fallback.backups.clone(),
        })?;
    }

    Ok(Arc::new(router))
}
