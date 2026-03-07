use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
use tracing::debug;

const CONFIG_FILE_NAME: &str = "gemenr.toml";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_ENDPOINT_ENV: &str = "ANTHROPIC_API_ENDPOINT";
const GEMENR_MODEL_ENV: &str = "GEMENR_MODEL";

/// Application configuration for Gemenr.
///
/// Configuration is organized into provider definitions, model definitions,
/// and a root model selector that chooses the active model entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The selected model configuration identifier from [`Config::models`].
    pub model: String,
    /// Available provider definitions keyed by provider identifier.
    pub providers: HashMap<String, ProviderConfig>,
    /// Available model definitions keyed by model identifier.
    pub models: HashMap<String, ModelConfig>,
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
#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    model: Option<String>,
    #[serde(default)]
    providers: HashMap<String, RawProviderConfig>,
    #[serde(default)]
    models: HashMap<String, RawModelConfig>,
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

    fn from_sources(file_config: Option<RawConfig>) -> Result<Self, ConfigError> {
        let raw_config = file_config.unwrap_or_default();
        let models = build_models(raw_config.models)?;

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
            providers,
            models,
        })
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

#[cfg(test)]
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_ENDPOINT_ENV, ANTHROPIC_API_KEY_ENV, CONFIG_FILE_NAME, Config, ConfigError,
        ENV_MUTEX, GEMENR_MODEL_ENV, ProviderType,
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
        assert_eq!(selected_model.provider, "anthropic");
        assert_eq!(selected_model.model, "claude-haiku-4-5-20251001");
        assert_eq!(selected_model.max_tokens, Some(256));
        assert_eq!(selected_provider.provider_type, ProviderType::Anthropic);
        assert_eq!(selected_provider.api_key, "file-api-key");
        assert_eq!(
            selected_provider.api_endpoint.as_deref(),
            Some("https://file.example/v1/messages")
        );
    }

    #[test]
    fn load_requires_root_model_selection() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let _isolation = TestIsolation::new();

        let error = Config::load().expect_err("missing root model should error");
        assert!(matches!(error, ConfigError::Invalid(message) if message.contains("root `model`")));
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
