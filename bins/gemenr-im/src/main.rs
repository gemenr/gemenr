use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use gemenr_core::model::AnthropicProvider;
use gemenr_core::{
    AccessError, AccessInbound, AccessOutbound, Config, ConfigError, ConversationDriver,
    DenyAllApprovals, InMemoryTapeStore, JsonlTapeStore, ModelProvider, ModelRouter, ProviderType,
    RuntimeBuilder, RuntimeManager, SoulManager, TapeStore, ToolInvoker,
};
use gemenr_tools::{RuleBasedPolicyEvaluator, ToolPlane, builtin};
use tokio::sync::RwLock;
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

    let builder = match build_runtime_builder(&config).await {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };
    let manager = Arc::new(RuntimeManager::new(
        builder,
        DEFAULT_SYSTEM_PROMPT.to_string(),
    ));
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
    manager: Arc<RuntimeManager>,
}

impl RuntimeManagerDriver {
    fn new(manager: Arc<RuntimeManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl ConversationDriver for RuntimeManagerDriver {
    async fn handle(&self, inbound: AccessInbound) -> Result<AccessOutbound, AccessError> {
        self.manager
            .dispatch(inbound)
            .await
            .map_err(|error| AccessError::Driver(error.to_string()))
    }
}

#[async_trait]
impl IdleCollector for RuntimeManagerDriver {
    async fn hibernate_idle(&self, max_idle: Duration) -> usize {
        self.manager
            .hibernate_idle(max_idle)
            .await
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

async fn build_runtime_builder(config: &Config) -> Result<RuntimeBuilder, ConfigError> {
    let provider = build_model_provider(config)?;
    let workspace = current_workspace();
    let app_dir = workspace.join(".gemenr");
    let soul = load_soul_manager(&app_dir);
    let tools = build_tool_invoker(config, Arc::clone(&soul)).await?;

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

async fn build_tool_invoker(
    config: &Config,
    soul: Arc<RwLock<SoulManager>>,
) -> Result<Arc<dyn ToolInvoker>, ConfigError> {
    let tooling = config.tooling_view();
    let mut tool_plane = ToolPlane::with_policy_evaluator(Arc::new(
        RuleBasedPolicyEvaluator::from_config(&tooling.policy),
    ));
    builtin::register_builtin_tools(&mut tool_plane, soul);
    tool_plane
        .register_mcp_servers(&tooling.mcp)
        .await
        .map_err(|error| {
            ConfigError::Invalid(format!("failed to register enabled MCP servers: {error}"))
        })?;
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use serde_json::json;
    use tokio::sync::RwLock;

    use super::{build_tool_invoker, load_soul_manager};
    use gemenr_core::{
        AuthorizationDecision, Config, ConfigError, McpServerConfig, ModelConfig, PolicyContext,
        PolicyEffect, PolicyRuleConfig, ProviderConfig, ProviderType, SandboxKind,
        ScopedPolicyConfig,
    };

    #[tokio::test]
    async fn builder_wires_rule_based_policy_evaluator_from_config() {
        let mut config = test_config();
        config.policy.conversations = vec![ScopedPolicyConfig {
            id: "conv-1".to_string(),
            rules: vec![PolicyRuleConfig {
                tool: "shell".to_string(),
                effect: PolicyEffect::Deny,
                sandbox: SandboxKind::None,
            }],
        }];

        let tools = build_tool_invoker(&config, test_soul())
            .await
            .expect("tool invoker should build");
        let decision = tools.authorize(
            &gemenr_core::ToolCallRequest {
                call_id: "call-1".to_string(),
                name: "shell".to_string(),
                arguments: json!({}),
            },
            &PolicyContext {
                conversation_id: Some("conv-1".to_string()),
                ..PolicyContext::default()
            },
        );

        assert!(matches!(decision, AuthorizationDecision::Denied { .. }));
    }

    #[tokio::test]
    async fn builder_registers_enabled_mcp_servers() {
        let mut config = test_config();
        config.mcp.servers = vec![mock_mcp_server_config()];

        let tools = build_tool_invoker(&config, test_soul())
            .await
            .expect("tool invoker should build");

        assert!(
            tools
                .list_specs()
                .iter()
                .any(|spec| spec.name == "mcp.mock.echo")
        );
    }

    #[tokio::test]
    async fn missing_mcp_server_configuration_is_not_silently_ignored() {
        let mut config = test_config();
        config.mcp.servers = vec![McpServerConfig {
            name: "missing".to_string(),
            command: "/definitely-missing-gemenr-mcp".to_string(),
            args: Vec::new(),
            env: HashMap::new(),
            enabled: true,
        }];

        let error = build_tool_invoker(&config, test_soul())
            .await
            .err()
            .expect("missing MCP executable should fail");

        assert!(matches!(error, ConfigError::Invalid(_)));
        assert!(
            error
                .to_string()
                .contains("failed to register enabled MCP servers")
        );
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

    fn test_soul() -> Arc<RwLock<gemenr_core::SoulManager>> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!(
            "gemenr-im-test-soul-{}-{timestamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&workspace).expect("test soul workspace should exist");
        load_soul_manager(&workspace)
    }

    fn mock_mcp_server_config() -> McpServerConfig {
        McpServerConfig {
            name: "mock".to_string(),
            command: "python3".to_string(),
            args: vec![
                "-u".to_string(),
                "-c".to_string(),
                framed_server_script().to_string(),
            ],
            env: HashMap::new(),
            enabled: true,
        }
    }

    fn framed_server_script() -> &'static str {
        r#"
import json, sys

def read_msg():
    headers = b''
    while b'\r\n\r\n' not in headers:
        chunk = sys.stdin.buffer.read(1)
        if not chunk:
            return None
        headers += chunk
    header_text = headers.decode('utf-8')
    length = 0
    for line in header_text.split('\r\n'):
        if line.lower().startswith('content-length:'):
            length = int(line.split(':', 1)[1].strip())
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode('utf-8'))

def write_msg(payload):
    body = json.dumps(payload).encode('utf-8')
    sys.stdout.buffer.write(f'Content-Length: {len(body)}\r\n\r\n'.encode('utf-8'))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()

while True:
    message = read_msg()
    if message is None:
        break
    method = message.get('method')
    if method == 'initialize':
        write_msg({'jsonrpc':'2.0','id':message['id'],'result':{'serverInfo':{'name':'mock'}}})
    elif method == 'tools/list':
        write_msg({'jsonrpc':'2.0','id':message['id'],'result':{'tools':[{'name':'echo','description':'Echo text','inputSchema':{'type':'object'}}]}})
    elif method == 'tools/call':
        write_msg({'jsonrpc':'2.0','id':message['id'],'result':{'content':[{'type':'text','text':'echo'}],'isError':False}})
"#
    }
}
