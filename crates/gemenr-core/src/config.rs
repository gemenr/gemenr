use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
use tracing::debug;

const CONFIG_FILE_NAME: &str = "gemenr.toml";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_ENDPOINT_ENV: &str = "ANTHROPIC_API_ENDPOINT";
const GEMENR_MODEL_ENV: &str = "GEMENR_MODEL";
const DEFAULT_LARK_DEBOUNCE_MS: u64 = 300;

/// Application configuration for Gemenr.
///
/// Configuration is organized into provider definitions, model definitions,
/// access-layer integration, and Phase 2 execution controls.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The selected model configuration identifier from [`Config::models`].
    pub model: String,
    /// Available provider definitions keyed by provider identifier.
    pub providers: HashMap<String, ProviderConfig>,
    /// Available model definitions keyed by model identifier.
    pub models: HashMap<String, ModelConfig>,
    /// Tool dispatcher strategy override.
    ///
    /// Valid values are `native`, `xml`, and `auto`. The default is `auto`,
    /// which lets the runtime choose based on provider capabilities.
    pub tool_dispatcher: String,
    /// Access-layer configuration.
    pub access: AccessConfig,
    /// Cron-triggered jobs.
    pub cron: Vec<CronJobConfig>,
    /// Policy rule configuration.
    pub policy: PolicyConfig,
    /// Phase 2 provider fallback configuration.
    pub fallback: Option<ModelFallbackConfig>,
    /// External MCP server configuration.
    pub mcp: McpConfig,
}

/// Read-only runtime configuration needed by the core composition root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRuntimeConfigView {
    /// Selected model identifier from [`Config::models`].
    pub model_id: String,
    /// Tool dispatcher strategy used to build the runtime.
    pub tool_dispatcher: String,
}

/// Read-only tool-plane configuration used by binary composition roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolingConfigView {
    /// Policy rules used to evaluate tool execution.
    pub policy: PolicyConfig,
    /// External MCP server registrations.
    pub mcp: McpConfig,
}

/// Read-only access-layer configuration used by binary composition roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessConfigView {
    /// Transport-specific access configuration.
    pub access: AccessConfig,
    /// Cron-triggered jobs and their reporting routes.
    pub cron: Vec<CronJobConfig>,
}

/// Access-layer related configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AccessConfig {
    /// Lark / Feishu long-connection settings.
    pub lark: Option<LarkConfig>,
}

/// Lark access configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LarkConfig {
    /// App ID issued by Lark.
    pub app_id: String,
    /// App secret used to fetch tenant access tokens.
    pub app_secret: String,
    /// Optional WebSocket endpoint override for tests.
    pub ws_endpoint: Option<String>,
    /// Milliseconds to debounce bursts of inbound messages in one conversation.
    pub debounce_ms: u64,
}

/// Cron-triggered task definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronJobConfig {
    /// Stable job name.
    pub name: String,
    /// Cron expression.
    pub schedule: String,
    /// Prompt sent into task mode.
    pub prompt: String,
    /// Optional allowlist of tools visible to the runtime.
    pub tools: Option<Vec<String>>,
    /// Optional route such as `stdio:` or `lark:<chat_id>`.
    pub report_to: Option<String>,
}

/// Root policy configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PolicyConfig {
    /// Organization-scoped rule groups.
    pub organizations: Vec<ScopedPolicyConfig>,
    /// Workspace-scoped rule groups.
    pub workspaces: Vec<ScopedPolicyConfig>,
    /// Conversation-scoped rule groups.
    pub conversations: Vec<ScopedPolicyConfig>,
}

/// Policy rules attached to a specific scope identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopedPolicyConfig {
    /// Stable scope identifier.
    pub id: String,
    /// Rules evaluated within the scope.
    pub rules: Vec<PolicyRuleConfig>,
}

/// One configured policy rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRuleConfig {
    /// Tool name targeted by this rule.
    pub tool: String,
    /// Rule effect.
    pub effect: PolicyEffect,
    /// Sandbox backend selected when the rule applies.
    pub sandbox: PolicySandboxKind,
}

/// Sandbox backend configured for one policy rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicySandboxKind {
    /// Run the tool without a sandbox wrapper.
    None,
    /// Run inside a macOS Seatbelt sandbox.
    Seatbelt,
    /// Run inside a Linux Landlock sandbox.
    Landlock,
}

/// Effect configured for one policy rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyEffect {
    /// Allow the tool call.
    Allow,
    /// Require confirmation before execution.
    NeedConfirmation,
    /// Deny the tool call.
    Deny,
}

/// Phase 2 model fallback configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFallbackConfig {
    /// Primary provider key.
    pub primary: String,
    /// Backup provider keys in failover order.
    pub backups: Vec<String>,
}

/// External stdio MCP server definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    /// Stable server name.
    pub name: String,
    /// Command used to start the server process.
    pub command: String,
    /// Command-line arguments passed to the server process.
    pub args: Vec<String>,
    /// Environment variables injected into the server process.
    pub env: HashMap<String, String>,
    /// Whether this server should be started.
    pub enabled: bool,
}

/// Root MCP configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct McpConfig {
    /// Configured stdio servers.
    pub servers: Vec<McpServerConfig>,
}

/// Configuration for a model provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    /// The provider implementation type.
    pub provider_type: ProviderType,
    /// API key used to authenticate provider requests.
    pub api_key: String,
    /// Optional API endpoint override.
    pub api_endpoint: Option<String>,
}

/// Supported provider types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    /// Anthropic Claude provider.
    Anthropic,
}

/// Configuration for a selectable model entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    /// Provider identifier in [`Config::providers`].
    pub provider: String,
    /// Remote model name understood by the provider.
    pub model: String,
    /// Maximum tokens to generate. `None` means provider default.
    pub max_tokens: Option<u32>,
}

/// Errors that can occur while loading configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// No API key was found for the selected Anthropic provider.
    #[error(
        "API key not found for the selected Anthropic provider: set ANTHROPIC_API_KEY or add providers.<id>.api_key to gemenr.toml"
    )]
    ApiKeyMissing,

    /// Reading the configuration file failed.
    #[error("failed to read config file: {0}")]
    FileRead(#[from] std::io::Error),

    /// Parsing the configuration file failed.
    #[error("failed to parse config file: {0}")]
    FileParse(#[from] toml::de::Error),

    /// A configuration value failed validation or parsing.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    model: Option<String>,
    #[serde(default = "default_tool_dispatcher")]
    tool_dispatcher: String,
    #[serde(default)]
    providers: HashMap<String, RawProviderConfig>,
    #[serde(default)]
    models: HashMap<String, RawModelConfig>,
    #[serde(default)]
    access: RawAccessConfig,
    #[serde(default)]
    cron: Vec<RawCronJobConfig>,
    #[serde(default)]
    policy: RawPolicyConfig,
    fallback: Option<RawModelFallbackConfig>,
    #[serde(default)]
    mcp: RawMcpConfig,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            model: None,
            tool_dispatcher: default_tool_dispatcher(),
            providers: HashMap::new(),
            models: HashMap::new(),
            access: RawAccessConfig::default(),
            cron: Vec::new(),
            policy: RawPolicyConfig::default(),
            fallback: None,
            mcp: RawMcpConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProviderConfig {
    #[serde(rename = "type")]
    provider_type: String,
    api_key: Option<String>,
    api_endpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelConfig {
    provider: String,
    model: String,
    max_tokens: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAccessConfig {
    lark: Option<RawLarkConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLarkConfig {
    app_id: String,
    app_secret: String,
    ws_endpoint: Option<String>,
    #[serde(default = "default_lark_debounce_ms")]
    debounce_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCronJobConfig {
    name: String,
    schedule: String,
    prompt: String,
    tools: Option<Vec<String>>,
    report_to: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyConfig {
    #[serde(default)]
    organizations: Vec<RawScopedPolicyConfig>,
    #[serde(default)]
    workspaces: Vec<RawScopedPolicyConfig>,
    #[serde(default)]
    conversations: Vec<RawScopedPolicyConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScopedPolicyConfig {
    id: String,
    #[serde(default)]
    rules: Vec<RawPolicyRuleConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyRuleConfig {
    tool: String,
    effect: String,
    #[serde(default = "default_sandbox_kind")]
    sandbox: PolicySandboxKind,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelFallbackConfig {
    primary: String,
    backups: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMcpConfig {
    #[serde(default)]
    servers: Vec<RawMcpServerConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMcpServerConfig {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default = "default_enabled")]
    enabled: bool,
}

impl Config {
    /// Load configuration from environment variables, config file, and defaults.
    pub fn load() -> Result<Self, ConfigError> {
        let file_config = discover_config_path()
            .map(|path| {
                debug!(path = %path.display(), "loading configuration file");
                read_config_file(&path)
            })
            .transpose()?;

        Self::from_sources(file_config)
    }

    /// Load configuration from a specific config file path.
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        debug!(path = %path.display(), "loading configuration from explicit path");
        let file_config = read_config_file(path)?;
        Self::from_sources(Some(file_config))
    }

    /// Returns the selected model definition.
    pub fn selected_model(&self) -> Result<&ModelConfig, ConfigError> {
        self.models.get(&self.model).ok_or_else(|| {
            ConfigError::Invalid(format!("model `{}` not found in [models]", self.model))
        })
    }

    /// Returns the provider definition for the selected model.
    pub fn selected_provider(&self) -> Result<&ProviderConfig, ConfigError> {
        let selected_model = self.selected_model()?;
        self.providers.get(&selected_model.provider).ok_or_else(|| {
            ConfigError::Invalid(format!(
                "provider `{}` not found in [providers]",
                selected_model.provider
            ))
        })
    }

    /// Returns the runtime-facing subset of configuration for core assembly.
    pub fn core_runtime_view(&self) -> CoreRuntimeConfigView {
        CoreRuntimeConfigView {
            model_id: self.model.clone(),
            tool_dispatcher: self.tool_dispatcher.clone(),
        }
    }

    /// Returns the tool-plane subset of configuration for policy and MCP wiring.
    pub fn tooling_view(&self) -> ToolingConfigView {
        ToolingConfigView {
            policy: self.policy.clone(),
            mcp: self.mcp.clone(),
        }
    }

    /// Returns the access-layer subset of configuration for transports and cron.
    pub fn access_view(&self) -> AccessConfigView {
        AccessConfigView {
            access: self.access.clone(),
            cron: self.cron.clone(),
        }
    }

    fn from_sources(file_config: Option<RawConfig>) -> Result<Self, ConfigError> {
        let raw_config = file_config.unwrap_or_default();
        let models = build_models(raw_config.models)?;
        let tool_dispatcher = parse_tool_dispatcher(raw_config.tool_dispatcher)?;

        let selected_model_id = normalize_optional_string(raw_config.model)
            .or(optional_env_string(GEMENR_MODEL_ENV)?)
            .ok_or_else(|| {
                ConfigError::Invalid("root `model` must select one entry from [models]".to_string())
            })?;

        let selected_model = models.get(&selected_model_id).ok_or_else(|| {
            ConfigError::Invalid(format!("model `{selected_model_id}` not found in [models]"))
        })?;
        let selected_provider_id = selected_model.provider.clone();

        let providers = build_providers(raw_config.providers, &selected_provider_id)?;
        validate_model_references(&models, &providers)?;

        Ok(Self {
            model: selected_model_id,
            providers: providers.clone(),
            models,
            tool_dispatcher,
            access: build_access_config(raw_config.access)?,
            cron: build_cron_jobs(raw_config.cron)?,
            policy: build_policy_config(raw_config.policy)?,
            fallback: build_fallback_config(raw_config.fallback, &providers)?,
            mcp: build_mcp_config(raw_config.mcp)?,
        })
    }
}

fn default_tool_dispatcher() -> String {
    "auto".to_string()
}

fn default_lark_debounce_ms() -> u64 {
    DEFAULT_LARK_DEBOUNCE_MS
}

fn default_sandbox_kind() -> PolicySandboxKind {
    PolicySandboxKind::None
}

fn default_enabled() -> bool {
    true
}

fn parse_tool_dispatcher(tool_dispatcher: String) -> Result<String, ConfigError> {
    let normalized = normalize_required_string(tool_dispatcher, || {
        "root `tool_dispatcher` must not be empty".to_string()
    })?;

    match normalized.as_str() {
        "auto" | "native" | "xml" => Ok(normalized),
        _ => Err(ConfigError::Invalid(
            "root `tool_dispatcher` must be one of `auto`, `native`, or `xml`".to_string(),
        )),
    }
}

fn build_models(
    raw_models: HashMap<String, RawModelConfig>,
) -> Result<HashMap<String, ModelConfig>, ConfigError> {
    let mut models = HashMap::with_capacity(raw_models.len());

    for (model_id, raw_model) in raw_models {
        let provider = normalize_required_string(raw_model.provider, || {
            format!("models.{model_id}.provider must not be empty")
        })?;
        let model_name = normalize_required_string(raw_model.model, || {
            format!("models.{model_id}.model must not be empty")
        })?;
        models.insert(
            model_id,
            ModelConfig {
                provider,
                model: model_name,
                max_tokens: raw_model.max_tokens,
            },
        );
    }

    Ok(models)
}

fn build_providers(
    raw_providers: HashMap<String, RawProviderConfig>,
    selected_provider_id: &str,
) -> Result<HashMap<String, ProviderConfig>, ConfigError> {
    let mut providers = HashMap::with_capacity(raw_providers.len());

    for (provider_id, raw_provider) in raw_providers {
        let provider_type = parse_provider_type(&provider_id, &raw_provider.provider_type)?;
        let is_selected = provider_id == selected_provider_id;

        let api_key = if is_selected && provider_type == ProviderType::Anthropic {
            normalize_optional_string(raw_provider.api_key)
                .or(optional_env_string(ANTHROPIC_API_KEY_ENV)?)
        } else {
            normalize_optional_string(raw_provider.api_key)
        };

        let api_endpoint = if is_selected && provider_type == ProviderType::Anthropic {
            normalize_optional_string(raw_provider.api_endpoint)
                .or(optional_env_string(ANTHROPIC_API_ENDPOINT_ENV)?)
        } else {
            normalize_optional_string(raw_provider.api_endpoint)
        };

        let api_key = api_key.ok_or_else(|| {
            if is_selected && provider_type == ProviderType::Anthropic {
                ConfigError::ApiKeyMissing
            } else {
                ConfigError::Invalid(format!("providers.{provider_id}.api_key must not be empty"))
            }
        })?;

        providers.insert(
            provider_id,
            ProviderConfig {
                provider_type,
                api_key,
                api_endpoint,
            },
        );
    }

    Ok(providers)
}

fn build_access_config(raw_access: RawAccessConfig) -> Result<AccessConfig, ConfigError> {
    Ok(AccessConfig {
        lark: raw_access.lark.map(build_lark_config).transpose()?,
    })
}

fn build_lark_config(raw_lark: RawLarkConfig) -> Result<LarkConfig, ConfigError> {
    Ok(LarkConfig {
        app_id: normalize_required_string(raw_lark.app_id, || {
            "access.lark.app_id must not be empty".to_string()
        })?,
        app_secret: normalize_required_string(raw_lark.app_secret, || {
            "access.lark.app_secret must not be empty".to_string()
        })?,
        ws_endpoint: normalize_optional_string(raw_lark.ws_endpoint),
        debounce_ms: raw_lark.debounce_ms,
    })
}

fn build_cron_jobs(raw_cron: Vec<RawCronJobConfig>) -> Result<Vec<CronJobConfig>, ConfigError> {
    raw_cron
        .into_iter()
        .enumerate()
        .map(|(index, job)| {
            Ok(CronJobConfig {
                name: normalize_required_string(job.name, || {
                    format!("cron[{index}].name must not be empty")
                })?,
                schedule: normalize_required_string(job.schedule, || {
                    format!("cron[{index}].schedule must not be empty")
                })?,
                prompt: normalize_required_string(job.prompt, || {
                    format!("cron[{index}].prompt must not be empty")
                })?,
                tools: normalize_optional_vec(job.tools),
                report_to: normalize_optional_string(job.report_to),
            })
        })
        .collect()
}

fn build_policy_config(raw_policy: RawPolicyConfig) -> Result<PolicyConfig, ConfigError> {
    Ok(PolicyConfig {
        organizations: build_scoped_policy_configs(
            raw_policy.organizations,
            "policy.organizations",
        )?,
        workspaces: build_scoped_policy_configs(raw_policy.workspaces, "policy.workspaces")?,
        conversations: build_scoped_policy_configs(
            raw_policy.conversations,
            "policy.conversations",
        )?,
    })
}

fn build_scoped_policy_configs(
    raw_groups: Vec<RawScopedPolicyConfig>,
    scope_name: &str,
) -> Result<Vec<ScopedPolicyConfig>, ConfigError> {
    raw_groups
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            Ok(ScopedPolicyConfig {
                id: normalize_required_string(group.id, || {
                    format!("{scope_name}[{index}].id must not be empty")
                })?,
                rules: build_policy_rules(group.rules, scope_name, index)?,
            })
        })
        .collect()
}

fn build_policy_rules(
    raw_rules: Vec<RawPolicyRuleConfig>,
    scope_name: &str,
    scope_index: usize,
) -> Result<Vec<PolicyRuleConfig>, ConfigError> {
    raw_rules
        .into_iter()
        .enumerate()
        .map(|(rule_index, rule)| {
            Ok::<PolicyRuleConfig, ConfigError>(PolicyRuleConfig {
                tool: normalize_required_string(rule.tool, || {
                    format!(
                        "{scope_name}[{scope_index}].rules[{rule_index}].tool must not be empty"
                    )
                })?,
                effect: parse_policy_effect(&rule.effect, scope_name, scope_index, rule_index)?,
                sandbox: rule.sandbox,
            })
        })
        .collect()
}

fn parse_policy_effect(
    value: &str,
    scope_name: &str,
    scope_index: usize,
    rule_index: usize,
) -> Result<PolicyEffect, ConfigError> {
    match value.trim() {
        "allow" => Ok(PolicyEffect::Allow),
        "confirm" | "need_confirmation" => Ok(PolicyEffect::NeedConfirmation),
        "deny" => Ok(PolicyEffect::Deny),
        other => Err(ConfigError::Invalid(format!(
            "{scope_name}[{scope_index}].rules[{rule_index}].effect `{other}` is not supported"
        ))),
    }
}

fn build_fallback_config(
    raw_fallback: Option<RawModelFallbackConfig>,
    providers: &HashMap<String, ProviderConfig>,
) -> Result<Option<ModelFallbackConfig>, ConfigError> {
    let Some(raw_fallback) = raw_fallback else {
        return Ok(None);
    };

    let primary = normalize_required_string(raw_fallback.primary, || {
        "fallback.primary must not be empty".to_string()
    })?;
    if !providers.contains_key(&primary) {
        return Err(ConfigError::Invalid(format!(
            "fallback.primary references unknown provider `{primary}`"
        )));
    }

    let backups = normalize_required_vec(raw_fallback.backups, || {
        "fallback.backups must contain at least one provider".to_string()
    })?;

    for backup in &backups {
        if !providers.contains_key(backup) {
            return Err(ConfigError::Invalid(format!(
                "fallback.backups references unknown provider `{backup}`"
            )));
        }
    }

    Ok(Some(ModelFallbackConfig { primary, backups }))
}

fn build_mcp_config(raw_mcp: RawMcpConfig) -> Result<McpConfig, ConfigError> {
    let servers = raw_mcp
        .servers
        .into_iter()
        .enumerate()
        .map(|(index, server)| {
            Ok::<McpServerConfig, ConfigError>(McpServerConfig {
                name: normalize_required_string(server.name, || {
                    format!("mcp.servers[{index}].name must not be empty")
                })?,
                command: normalize_required_string(server.command, || {
                    format!("mcp.servers[{index}].command must not be empty")
                })?,
                args: normalize_string_vec(server.args),
                env: normalize_string_map(server.env),
                enabled: server.enabled,
            })
        })
        .collect::<Result<Vec<_>, ConfigError>>()?;

    Ok(McpConfig { servers })
}

fn validate_model_references(
    models: &HashMap<String, ModelConfig>,
    providers: &HashMap<String, ProviderConfig>,
) -> Result<(), ConfigError> {
    for (model_id, model) in models {
        if !providers.contains_key(&model.provider) {
            return Err(ConfigError::Invalid(format!(
                "models.{model_id}.provider references unknown provider `{}`",
                model.provider
            )));
        }
    }

    Ok(())
}

fn parse_provider_type(provider_id: &str, value: &str) -> Result<ProviderType, ConfigError> {
    match value.trim() {
        "anthropic" => Ok(ProviderType::Anthropic),
        other => Err(ConfigError::Invalid(format!(
            "providers.{provider_id}.type `{other}` is not supported"
        ))),
    }
}

fn discover_config_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join(CONFIG_FILE_NAME));
    }

    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(
            PathBuf::from(home)
                .join(".config/gemenr")
                .join(CONFIG_FILE_NAME),
        );
    }

    let path = candidates.into_iter().find(|candidate| candidate.is_file());
    if let Some(found) = path.as_ref() {
        debug!(path = %found.display(), "discovered configuration file");
    }
    path
}

fn read_config_file(path: &Path) -> Result<RawConfig, ConfigError> {
    let contents = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&contents)?)
}

fn optional_env_string(key: &str) -> Result<Option<String>, ConfigError> {
    match std::env::var(key) {
        Ok(value) => Ok(normalize_optional_string(Some(value))),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(ConfigError::Invalid(format!(
            "failed to read environment variable {key}: {error}"
        ))),
    }
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    })
}

fn normalize_required_string(
    value: String,
    error_message: impl FnOnce() -> String,
) -> Result<String, ConfigError> {
    normalize_optional_string(Some(value)).ok_or_else(|| ConfigError::Invalid(error_message()))
}

fn normalize_optional_vec(values: Option<Vec<String>>) -> Option<Vec<String>> {
    values.and_then(|values| {
        let normalized = normalize_string_vec(values);
        (!normalized.is_empty()).then_some(normalized)
    })
}

fn normalize_required_vec(
    values: Vec<String>,
    error_message: impl FnOnce() -> String,
) -> Result<Vec<String>, ConfigError> {
    let normalized = normalize_string_vec(values);
    if normalized.is_empty() {
        Err(ConfigError::Invalid(error_message()))
    } else {
        Ok(normalized)
    }
}

fn normalize_string_vec(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| normalize_optional_string(Some(value)))
        .collect()
}

fn normalize_string_map(values: HashMap<String, String>) -> HashMap<String, String> {
    values
        .into_iter()
        .filter_map(|(key, value)| {
            let normalized_key = normalize_optional_string(Some(key))?;
            let normalized_value = normalize_optional_string(Some(value))?;
            Some((normalized_key, normalized_value))
        })
        .collect()
}

#[cfg(test)]
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_ENDPOINT_ENV, ANTHROPIC_API_KEY_ENV, AccessConfig, AccessConfigView,
        CONFIG_FILE_NAME, Config, ConfigError, CoreRuntimeConfigView, CronJobConfig, ENV_MUTEX,
        GEMENR_MODEL_ENV, LarkConfig, McpConfig, McpServerConfig, PolicyConfig, PolicyEffect,
        PolicyRuleConfig, ProviderType, ScopedPolicyConfig, ToolingConfigView,
    };
    use std::env;
    use std::ffi::{OsStr, OsString};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_ENV_KEYS: [&str; 3] = [
        ANTHROPIC_API_KEY_ENV,
        ANTHROPIC_API_ENDPOINT_ENV,
        GEMENR_MODEL_ENV,
    ];

    struct TestIsolation {
        original_cwd: PathBuf,
        original_home: Option<OsString>,
        original_env: Vec<(&'static str, Option<OsString>)>,
        temp_dir: PathBuf,
    }

    impl TestIsolation {
        fn new() -> Self {
            let temp_dir = unique_temp_dir();
            fs::create_dir_all(&temp_dir).expect("temporary directory should be created");

            let original_cwd = env::current_dir().expect("current directory should be readable");
            let original_home = env::var_os("HOME");
            let original_env = TEST_ENV_KEYS
                .iter()
                .map(|&key| (key, env::var_os(key)))
                .collect();

            env::set_current_dir(&temp_dir).expect("current directory should switch to temp dir");
            set_env_var("HOME", &temp_dir);
            clear_test_env();

            Self {
                original_cwd,
                original_home,
                original_env,
                temp_dir,
            }
        }

        fn temp_dir(&self) -> &Path {
            &self.temp_dir
        }
    }

    impl Drop for TestIsolation {
        fn drop(&mut self) {
            env::set_current_dir(&self.original_cwd)
                .expect("current directory should restore after test");

            match &self.original_home {
                Some(value) => set_env_var("HOME", value),
                None => remove_env_var("HOME"),
            }

            for (key, value) in &self.original_env {
                match value {
                    Some(value) => set_env_var(key, value),
                    None => remove_env_var(key),
                }
            }

            let _ = fs::remove_dir_all(&self.temp_dir);
        }
    }

    #[test]
    fn load_from_parses_provider_and_model_tables() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"
api_endpoint = "https://file.example/v1/messages"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
max_tokens = 256
"#,
        );

        let config = Config::load_from(&config_path).expect("config file should parse");
        let selected_model = config
            .selected_model()
            .expect("selected model should exist");
        let selected_provider = config
            .selected_provider()
            .expect("selected provider should exist");

        assert_eq!(config.model, "default");
        assert_eq!(config.tool_dispatcher, "auto");
        assert_eq!(selected_model.provider, "anthropic");
        assert_eq!(selected_model.model, "claude-haiku-4-5-20251001");
        assert_eq!(selected_model.max_tokens, Some(256));
        assert_eq!(selected_provider.provider_type, ProviderType::Anthropic);
        assert_eq!(selected_provider.api_key, "file-api-key");
        assert_eq!(
            selected_provider.api_endpoint.as_deref(),
            Some("https://file.example/v1/messages")
        );
        assert!(config.access.lark.is_none());
        assert!(config.cron.is_empty());
        assert!(config.fallback.is_none());
        assert!(config.mcp.servers.is_empty());
    }

    #[test]
    fn load_requires_root_model_selection() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let _isolation = TestIsolation::new();

        let error = Config::load().expect_err("missing root model should error");
        assert!(matches!(error, ConfigError::Invalid(message) if message.contains("root `model`")));
    }

    #[test]
    fn load_defaults_tool_dispatcher_to_auto() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
"#,
        );

        let config = Config::load_from(&config_path).expect("config should load");

        assert_eq!(config.tool_dispatcher, "auto");
    }

    #[test]
    fn load_accepts_custom_tool_dispatcher() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"
tool_dispatcher = "native"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
"#,
        );

        let config = Config::load_from(&config_path).expect("config should load");

        assert_eq!(config.tool_dispatcher, "native");
    }

    #[test]
    fn load_old_config_without_tool_dispatcher_still_works() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
max_tokens = 256
"#,
        );

        let config =
            Config::load_from(&config_path).expect("config should stay backward compatible");

        assert_eq!(config.model, "default");
        assert_eq!(config.tool_dispatcher, "auto");
    }

    #[test]
    fn load_parses_lark_access_config() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[access.lark]
app_id = "cli-app"
app_secret = "cli-secret"
ws_endpoint = "wss://example.invalid/ws"
debounce_ms = 750
"#,
        );

        let config = Config::load_from(&config_path).expect("config should parse");
        let lark = config.access.lark.expect("lark config should exist");

        assert_eq!(lark.app_id, "cli-app");
        assert_eq!(lark.app_secret, "cli-secret");
        assert_eq!(
            lark.ws_endpoint.as_deref(),
            Some("wss://example.invalid/ws")
        );
        assert_eq!(lark.debounce_ms, 750);
    }

    #[test]
    fn load_parses_cron_jobs() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[[cron]]
name = "daily-system-check"
schedule = "0 9 * * *"
prompt = "run checks"
tools = ["shell", "fs.read"]
report_to = "stdio:"

[[cron]]
name = "weekly-report"
schedule = "0 18 * * 5"
prompt = "send report"
"#,
        );

        let config = Config::load_from(&config_path).expect("config should parse");

        assert_eq!(config.cron.len(), 2);
        assert_eq!(
            config.cron[0].tools.as_deref(),
            Some(&["shell".to_string(), "fs.read".to_string()][..])
        );
        assert_eq!(config.cron[0].report_to.as_deref(), Some("stdio:"));
        assert_eq!(config.cron[1].tools, None);
        assert_eq!(config.cron[1].report_to, None);
    }

    #[test]
    fn load_parses_mcp_servers() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[[mcp.servers]]
name = "filesystem"
command = "node"
args = ["server.js", "--stdio"]
enabled = false

[mcp.servers.env]
NODE_ENV = "test"
ROOT = "/tmp/project"
"#,
        );

        let config = Config::load_from(&config_path).expect("config should parse");
        let server = &config.mcp.servers[0];

        assert_eq!(config.mcp.servers.len(), 1);
        assert_eq!(server.name, "filesystem");
        assert_eq!(server.command, "node");
        assert_eq!(server.args, vec!["server.js", "--stdio"]);
        assert_eq!(server.env.get("NODE_ENV").map(String::as_str), Some("test"));
        assert_eq!(
            server.env.get("ROOT").map(String::as_str),
            Some("/tmp/project")
        );
        assert!(!server.enabled);
    }

    #[test]
    fn core_runtime_view_exposes_only_runtime_fields() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"
tool_dispatcher = "native"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[[cron]]
name = "daily-system-check"
schedule = "0 9 * * *"
prompt = "run checks"

[policy]

[[mcp.servers]]
name = "filesystem"
command = "node"
"#,
        );

        let config = Config::load_from(&config_path).expect("config should parse");
        let view = config.core_runtime_view();
        let CoreRuntimeConfigView {
            model_id,
            tool_dispatcher,
        } = view;

        assert_eq!(model_id, "default");
        assert_eq!(tool_dispatcher, "native");
    }

    #[test]
    fn tooling_view_contains_policy_and_mcp_without_model_selection() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"
tool_dispatcher = "xml"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[[policy.organizations]]
id = "org-1"

[[policy.organizations.rules]]
tool = "shell"
effect = "allow"

[[mcp.servers]]
name = "filesystem"
command = "node"
args = ["server.js"]
"#,
        );

        let config = Config::load_from(&config_path).expect("config should parse");
        let view = config.tooling_view();
        let ToolingConfigView { policy, mcp } = view;

        assert_eq!(
            policy,
            PolicyConfig {
                organizations: vec![ScopedPolicyConfig {
                    id: "org-1".to_string(),
                    rules: vec![PolicyRuleConfig {
                        tool: "shell".to_string(),
                        effect: PolicyEffect::Allow,
                        sandbox: super::PolicySandboxKind::None,
                    }],
                }],
                workspaces: Vec::new(),
                conversations: Vec::new(),
            }
        );
        assert_eq!(
            mcp,
            McpConfig {
                servers: vec![McpServerConfig {
                    name: "filesystem".to_string(),
                    command: "node".to_string(),
                    args: vec!["server.js".to_string()],
                    env: Default::default(),
                    enabled: true,
                }],
            }
        );
    }

    #[test]
    fn access_view_contains_access_and_cron() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[access.lark]
app_id = "cli-app"
app_secret = "cli-secret"
debounce_ms = 750

[[cron]]
name = "daily-system-check"
schedule = "0 9 * * *"
prompt = "run checks"
tools = ["shell"]
report_to = "lark:chat-id"
"#,
        );

        let config = Config::load_from(&config_path).expect("config should parse");
        let view = config.access_view();
        let AccessConfigView { access, cron } = view;

        assert_eq!(
            access,
            AccessConfig {
                lark: Some(LarkConfig {
                    app_id: "cli-app".to_string(),
                    app_secret: "cli-secret".to_string(),
                    ws_endpoint: None,
                    debounce_ms: 750,
                }),
            }
        );
        assert_eq!(
            cron,
            vec![CronJobConfig {
                name: "daily-system-check".to_string(),
                schedule: "0 9 * * *".to_string(),
                prompt: "run checks".to_string(),
                tools: Some(vec!["shell".to_string()]),
                report_to: Some("lark:chat-id".to_string()),
            }]
        );
    }

    #[test]
    fn legacy_phase_one_config_still_loads_with_views_available() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
max_tokens = 256
"#,
        );

        let config =
            Config::load_from(&config_path).expect("config should stay backward compatible");

        assert_eq!(
            config.core_runtime_view(),
            CoreRuntimeConfigView {
                model_id: "default".to_string(),
                tool_dispatcher: "auto".to_string(),
            }
        );
        assert_eq!(
            config.tooling_view(),
            ToolingConfigView {
                policy: PolicyConfig::default(),
                mcp: McpConfig::default(),
            }
        );
        assert_eq!(
            config.access_view(),
            AccessConfigView {
                access: AccessConfig::default(),
                cron: Vec::new(),
            }
        );
    }

    #[test]
    fn load_rejects_empty_fallback_backups() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.primary]
type = "anthropic"
api_key = "primary-key"

[providers.backup]
type = "anthropic"
api_key = "backup-key"

[models.default]
provider = "primary"
model = "claude-haiku-4-5-20251001"

[fallback]
primary = "primary"
backups = []
"#,
        );

        let error = Config::load_from(&config_path)
            .expect_err("config should reject empty fallback backups");
        assert!(
            matches!(error, ConfigError::Invalid(message) if message.contains("fallback.backups"))
        );
    }

    #[test]
    fn load_rejects_unknown_fallback_provider() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.primary]
type = "anthropic"
api_key = "primary-key"

[models.default]
provider = "primary"
model = "claude-haiku-4-5-20251001"

[fallback]
primary = "primary"
backups = ["missing"]
"#,
        );

        let error = Config::load_from(&config_path)
            .expect_err("config should reject unknown fallback provider");
        assert!(
            matches!(error, ConfigError::Invalid(message) if message.contains("unknown provider `missing`"))
        );
    }

    #[test]
    fn environment_selects_model_and_fills_selected_provider_credentials() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
[providers.primary]
type = "anthropic"

[providers.secondary]
type = "anthropic"
api_key = "secondary-key"

[models.primary]
provider = "primary"
model = "claude-haiku-4-5-20251001"

[models.secondary]
provider = "secondary"
model = "claude-sonnet-4-20250514"
"#,
        );

        set_env_var(GEMENR_MODEL_ENV, "primary");
        set_env_var(ANTHROPIC_API_KEY_ENV, "env-api-key");
        set_env_var(
            ANTHROPIC_API_ENDPOINT_ENV,
            "https://env.example/v1/messages",
        );

        let config = Config::load_from(&config_path).expect("config should load");
        let selected_provider = config
            .selected_provider()
            .expect("selected provider should exist");

        assert_eq!(config.model, "primary");
        assert_eq!(selected_provider.api_key, "env-api-key");
        assert_eq!(
            selected_provider.api_endpoint.as_deref(),
            Some("https://env.example/v1/messages")
        );
        assert_eq!(
            config.providers["secondary"].api_key, "secondary-key",
            "non-selected providers keep file values"
        );
    }

    #[test]
    fn config_file_overrides_environment_values() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"
api_endpoint = "https://file.example/v1/messages"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
max_tokens = 256
"#,
        );

        set_env_var(GEMENR_MODEL_ENV, "env-selected");
        set_env_var(ANTHROPIC_API_KEY_ENV, "env-api-key");
        set_env_var(
            ANTHROPIC_API_ENDPOINT_ENV,
            "https://env.example/v1/messages",
        );

        let config = Config::load_from(&config_path).expect("config should load");
        let selected_provider = config
            .selected_provider()
            .expect("selected provider should exist");

        assert_eq!(config.model, "default");
        assert_eq!(selected_provider.api_key, "file-api-key");
        assert_eq!(
            selected_provider.api_endpoint.as_deref(),
            Some("https://file.example/v1/messages")
        );
    }

    #[test]
    fn load_rejects_unknown_selected_model() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "missing"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
"#,
        );

        let error = Config::load_from(&config_path).expect_err("unknown model should error");
        assert!(
            matches!(error, ConfigError::Invalid(message) if message.contains("model `missing` not found"))
        );
    }

    #[test]
    fn load_rejects_unknown_provider_reference() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.anthropic]
type = "anthropic"
api_key = "file-api-key"

[models.default]
provider = "missing"
model = "claude-haiku-4-5-20251001"
"#,
        );

        let error = Config::load_from(&config_path).expect_err("unknown provider should error");
        assert!(
            matches!(error, ConfigError::Invalid(message) if message.contains("unknown provider `missing`"))
        );
    }

    #[test]
    fn load_rejects_unsupported_provider_type() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
model = "default"

[providers.openai]
type = "openai"
api_key = "file-api-key"

[models.default]
provider = "openai"
model = "gpt-4.1"
"#,
        );

        let error = Config::load_from(&config_path).expect_err("unsupported provider should error");
        assert!(
            matches!(error, ConfigError::Invalid(message) if message.contains("not supported"))
        );
    }

    #[test]
    fn load_rejects_old_flat_schema() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
api_key = "flat-key"
model = "claude-haiku-4-5-20251001"
"#,
        );

        let error = Config::load_from(&config_path).expect_err("old schema should be rejected");
        assert!(matches!(error, ConfigError::FileParse(_)));
    }

    #[test]
    fn example_config_matches_schema() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            include_str!("../../../gemenr.toml.example"),
        );

        let config = Config::load_from(&config_path).expect("example config should parse");
        let selected_model = config
            .selected_model()
            .expect("selected model should exist");
        let selected_provider = config
            .selected_provider()
            .expect("selected provider should exist");

        assert_eq!(config.model, "default");
        assert_eq!(selected_model.provider, "anthropic");
        assert_eq!(selected_provider.provider_type, ProviderType::Anthropic);
    }

    fn write_config(dir: &Path, contents: &str) -> PathBuf {
        let path = dir.join(CONFIG_FILE_NAME);
        fs::write(&path, contents).expect("config file should be written");
        path
    }

    fn clear_test_env() {
        for key in TEST_ENV_KEYS {
            remove_env_var(key);
        }
    }

    fn unique_temp_dir() -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "gemenr-config-test-{}-{timestamp}",
            std::process::id()
        ))
    }

    fn set_env_var(key: &str, value: impl AsRef<OsStr>) {
        // SAFETY: All callers hold ENV_MUTEX, which serializes process-wide env mutation.
        unsafe { env::set_var(key, value) };
    }

    fn remove_env_var(key: &str) {
        // SAFETY: All callers hold ENV_MUTEX, which serializes process-wide env mutation.
        unsafe { env::remove_var(key) };
    }
}
