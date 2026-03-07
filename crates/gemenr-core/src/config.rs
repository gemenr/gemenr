use std::fmt::Display;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
use tracing::debug;

const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
const DEFAULT_TEMPERATURE: f64 = 0.7;
const CONFIG_FILE_NAME: &str = "gemenr.toml";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const GEMENR_MODEL_ENV: &str = "GEMENR_MODEL";
const GEMENR_TEMPERATURE_ENV: &str = "GEMENR_TEMPERATURE";
const GEMENR_MAX_TOKENS_ENV: &str = "GEMENR_MAX_TOKENS";

/// Application configuration for Gemenr.
///
/// Configuration is loaded with the following priority (highest first):
/// 1. Environment variables
/// 2. Configuration file (`gemenr.toml`)
/// 3. Built-in defaults
#[derive(Debug, Clone)]
pub struct Config {
    /// API key for the model provider.
    pub api_key: String,
    /// Model identifier (e.g., `claude-sonnet-4-20250514`).
    pub model: String,
    /// Sampling temperature (0.0 - 1.0).
    pub temperature: f64,
    /// Maximum tokens to generate. `None` means provider default.
    pub max_tokens: Option<u32>,
}

/// Errors that can occur while loading configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// No API key was found in the environment or configuration file.
    #[error(
        "API key not found: set ANTHROPIC_API_KEY environment variable or add it to gemenr.toml"
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
struct ConfigFile {
    api_key: Option<String>,
    model: Option<String>,
    temperature: Option<f64>,
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

    fn from_sources(file_config: Option<ConfigFile>) -> Result<Self, ConfigError> {
        let file_config = file_config.unwrap_or_default();

        let api_key = optional_env_string(ANTHROPIC_API_KEY_ENV)?
            .or_else(|| normalize_optional_string(file_config.api_key))
            .ok_or(ConfigError::ApiKeyMissing)?;

        let model = optional_env_string(GEMENR_MODEL_ENV)?
            .or_else(|| normalize_optional_string(file_config.model))
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let temperature = parse_optional_env::<f64>(GEMENR_TEMPERATURE_ENV)?
            .or(file_config.temperature)
            .unwrap_or(DEFAULT_TEMPERATURE);
        validate_temperature(temperature)?;

        let max_tokens =
            parse_optional_env::<u32>(GEMENR_MAX_TOKENS_ENV)?.or(file_config.max_tokens);

        Ok(Self {
            api_key,
            model,
            temperature,
            max_tokens,
        })
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

fn read_config_file(path: &Path) -> Result<ConfigFile, ConfigError> {
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

fn parse_optional_env<T>(key: &str) -> Result<Option<T>, ConfigError>
where
    T: std::str::FromStr,
    T::Err: Display,
{
    optional_env_string(key)?
        .map(|value| {
            value.parse().map_err(|error| {
                ConfigError::Invalid(format!(
                    "failed to parse environment variable {key}: {error}"
                ))
            })
        })
        .transpose()
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    })
}

fn validate_temperature(temperature: f64) -> Result<(), ConfigError> {
    if !temperature.is_finite() || !(0.0..=1.0).contains(&temperature) {
        return Err(ConfigError::Invalid(format!(
            "temperature must be between 0.0 and 1.0, got {temperature}"
        )));
    }

    Ok(())
}

#[cfg(test)]
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_KEY_ENV, CONFIG_FILE_NAME, Config, ConfigError, DEFAULT_MODEL,
        DEFAULT_TEMPERATURE, ENV_MUTEX, GEMENR_MAX_TOKENS_ENV, GEMENR_MODEL_ENV,
        GEMENR_TEMPERATURE_ENV,
    };
    use std::env;
    use std::ffi::{OsStr, OsString};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_ENV_KEYS: [&str; 4] = [
        ANTHROPIC_API_KEY_ENV,
        GEMENR_MODEL_ENV,
        GEMENR_TEMPERATURE_ENV,
        GEMENR_MAX_TOKENS_ENV,
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
    fn load_reads_values_from_environment() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let _isolation = TestIsolation::new();

        set_env_var(ANTHROPIC_API_KEY_ENV, "env-api-key");
        set_env_var(GEMENR_MODEL_ENV, "claude-test-model");
        set_env_var(GEMENR_TEMPERATURE_ENV, "0.3");
        set_env_var(GEMENR_MAX_TOKENS_ENV, "512");

        let config = Config::load().expect("config should load from environment");

        assert_eq!(config.api_key, "env-api-key");
        assert_eq!(config.model, "claude-test-model");
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.max_tokens, Some(512));
    }

    #[test]
    fn load_requires_api_key() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let _isolation = TestIsolation::new();

        let error = Config::load().expect_err("missing API key should error");
        assert!(matches!(error, ConfigError::ApiKeyMissing));
    }

    #[test]
    fn load_applies_default_values() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let _isolation = TestIsolation::new();

        set_env_var(ANTHROPIC_API_KEY_ENV, "env-api-key");

        let config = Config::load().expect("config should load with defaults");

        assert_eq!(config.api_key, "env-api-key");
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(config.max_tokens, None);
    }

    #[test]
    fn load_from_parses_toml_configuration() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
api_key = "file-api-key"
model = "claude-file-model"
temperature = 0.2
max_tokens = 256
"#,
        );

        let config = Config::load_from(&config_path).expect("config file should parse");

        assert_eq!(config.api_key, "file-api-key");
        assert_eq!(config.model, "claude-file-model");
        assert_eq!(config.temperature, 0.2);
        assert_eq!(config.max_tokens, Some(256));
    }

    #[test]
    fn environment_overrides_config_file_values() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex should lock");
        let isolation = TestIsolation::new();
        let config_path = write_config(
            isolation.temp_dir(),
            r#"
api_key = "file-api-key"
model = "claude-file-model"
temperature = 0.2
max_tokens = 256
"#,
        );

        set_env_var(ANTHROPIC_API_KEY_ENV, "env-api-key");
        set_env_var(GEMENR_MODEL_ENV, "claude-env-model");
        set_env_var(GEMENR_TEMPERATURE_ENV, "0.9");
        set_env_var(GEMENR_MAX_TOKENS_ENV, "1024");

        let config = Config::load_from(&config_path).expect("env vars should override file values");

        assert_eq!(config.api_key, "env-api-key");
        assert_eq!(config.model, "claude-env-model");
        assert_eq!(config.temperature, 0.9);
        assert_eq!(config.max_tokens, Some(1024));
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
