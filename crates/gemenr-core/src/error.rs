use std::time::Duration;

use thiserror::Error;

/// Errors that can occur during model API interaction.
#[derive(Debug, Error)]
pub enum ModelError {
    /// API key is invalid or missing.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// Rate limit exceeded by the API provider.
    #[error("rate limited{}", .retry_after.map(|d| format!(", retry after {d:?}")).unwrap_or_default())]
    RateLimit {
        /// Suggested wait duration before retrying, if provided by the API.
        retry_after: Option<Duration>,
    },

    /// Request timed out.
    #[error("request timed out")]
    Timeout,

    /// Network-level error (DNS, connection refused, etc.).
    #[error("network error: {0}")]
    Network(String),

    /// API returned a non-success HTTP status.
    #[error("API error (status {status}): {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Error message from the API.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::ModelError;
    use std::time::Duration;

    #[test]
    fn display_formats_all_error_variants() {
        let auth = ModelError::Auth("bad key".to_string());
        let timeout = ModelError::Timeout;
        let network = ModelError::Network("connection refused".to_string());
        let api = ModelError::Api {
            status: 429,
            message: "too many requests".to_string(),
        };

        assert_eq!(auth.to_string(), "authentication failed: bad key");
        assert_eq!(timeout.to_string(), "request timed out");
        assert_eq!(network.to_string(), "network error: connection refused");
        assert_eq!(api.to_string(), "API error (status 429): too many requests");
    }

    #[test]
    fn rate_limit_display_handles_optional_retry_after() {
        let without_retry = ModelError::RateLimit { retry_after: None };
        let with_retry = ModelError::RateLimit {
            retry_after: Some(Duration::from_secs(30)),
        };

        assert_eq!(without_retry.to_string(), "rate limited");
        assert_eq!(with_retry.to_string(), "rate limited, retry after 30s");
    }
}
