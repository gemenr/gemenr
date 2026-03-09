use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use gemenr_core::config::McpServerConfig;
use serde_json::{Value, json};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// A remote MCP tool exposed by one stdio server.
#[derive(Debug, Clone, PartialEq)]
pub struct McpRemoteTool {
    /// Remote tool name.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// Input schema advertised by the MCP server.
    pub input_schema: Value,
}

/// A tool result returned by one MCP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolResult {
    /// Rendered text content returned by the tool.
    pub content: String,
    /// Whether the remote server marked this as an error result.
    pub is_error: bool,
}

/// Errors that can occur while speaking the MCP stdio protocol.
#[derive(Debug, Error)]
pub enum McpError {
    /// Starting the stdio child process failed.
    #[error("failed to start MCP server: {0}")]
    Start(#[source] std::io::Error),
    /// The child process did not expose piped stdio.
    #[error("MCP server must expose piped stdin/stdout")]
    MissingPipes,
    /// Reading or writing protocol bytes failed.
    #[error("I/O error while speaking MCP: {0}")]
    Io(#[from] std::io::Error),
    /// JSON decoding failed.
    #[error("invalid MCP JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    /// The server returned an invalid or incomplete frame.
    #[error("invalid MCP protocol frame: {0}")]
    Protocol(String),
    /// The request timed out.
    #[error("MCP request timed out")]
    Timeout,
    /// The server returned a JSON-RPC error.
    #[error("MCP remote error: {0}")]
    Remote(String),
}

/// Minimal stdio JSON-RPC client for MCP servers.
pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: AtomicU64,
    timeout: Duration,
}

impl McpClient {
    /// Start one stdio MCP server process.
    pub async fn start(config: &McpServerConfig) -> Result<Self, McpError> {
        let mut command = Command::new(&config.command);
        command.args(&config.args);
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        command.envs(config.env.iter());

        let mut child = command.spawn().map_err(McpError::Start)?;
        let stdin = child.stdin.take().ok_or(McpError::MissingPipes)?;
        let stdout = child.stdout.take().ok_or(McpError::MissingPipes)?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: AtomicU64::new(1),
            timeout: DEFAULT_TIMEOUT,
        })
    }

    /// Send MCP initialize.
    pub async fn initialize(&mut self) -> Result<(), McpError> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gemenr", "version": "0.1.0"}
            }),
        )
        .await
        .map(|_| ())
    }

    /// List tools exposed by the remote server.
    pub async fn list_tools(&mut self) -> Result<Vec<McpRemoteTool>, McpError> {
        let value = self.request("tools/list", json!({})).await?;
        let tools = value
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| McpError::Protocol("missing tools array".to_string()))?;

        tools
            .iter()
            .map(|tool| {
                Ok(McpRemoteTool {
                    name: tool
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| McpError::Protocol("tool missing name".to_string()))?
                        .to_string(),
                    description: tool
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input_schema: tool
                        .get("inputSchema")
                        .cloned()
                        .or_else(|| tool.get("input_schema").cloned())
                        .unwrap_or_else(|| json!({"type": "object"})),
                })
            })
            .collect()
    }

    /// Call one remote MCP tool.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<McpToolResult, McpError> {
        let value = self
            .request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;
        Ok(parse_tool_result(&value))
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_frame(&request).await?;
        let response = tokio::time::timeout(self.timeout, self.read_frame())
            .await
            .map_err(|_| McpError::Timeout)??;

        if response.get("id").and_then(Value::as_u64) != Some(id) {
            return Err(McpError::Protocol("response id mismatch".to_string()));
        }

        if let Some(error) = response.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown MCP error")
                .to_string();
            return Err(McpError::Remote(message));
        }

        response
            .get("result")
            .cloned()
            .ok_or_else(|| McpError::Protocol("missing result field".to_string()))
    }

    async fn write_frame(&mut self, payload: &Value) -> Result<(), McpError> {
        let body = serde_json::to_vec(payload)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(&body).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_frame(&mut self) -> Result<Value, McpError> {
        let mut header_bytes = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            let read = self.stdout.read(&mut byte).await?;
            if read == 0 {
                return Err(McpError::Protocol("unexpected EOF".to_string()));
            }
            header_bytes.push(byte[0]);
            if header_bytes.ends_with(b"\r\n\r\n") {
                break;
            }
        }

        let header = String::from_utf8(header_bytes)
            .map_err(|_| McpError::Protocol("headers must be valid utf-8".to_string()))?;
        let content_length = parse_content_length(&header)?;
        let mut body = vec![0u8; content_length];
        self.stdout.read_exact(&mut body).await?;
        serde_json::from_slice(&body).map_err(McpError::from)
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn parse_content_length(headers: &str) -> Result<usize, McpError> {
    for line in headers.split("\r\n") {
        if let Some(value) = line.strip_prefix("Content-Length:") {
            return value
                .trim()
                .parse::<usize>()
                .map_err(|_| McpError::Protocol("invalid Content-Length header".to_string()));
        }
    }

    Err(McpError::Protocol(
        "missing Content-Length header".to_string(),
    ))
}

fn parse_tool_result(value: &Value) -> McpToolResult {
    let content = value
        .get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|content| !content.is_empty())
        .or_else(|| {
            value
                .get("content")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .unwrap_or_default();

    McpToolResult {
        content,
        is_error: value
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{McpClient, McpError};
    use gemenr_core::config::McpServerConfig;
    use serde_json::json;

    fn mock_server_config(script: &str) -> McpServerConfig {
        McpServerConfig {
            name: "mock".to_string(),
            command: "python3".to_string(),
            args: vec!["-u".to_string(), "-c".to_string(), script.to_string()],
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
        text = message['params']['arguments'].get('text', '')
        write_msg({'jsonrpc':'2.0','id':message['id'],'result':{'content':[{'type':'text','text':f'echo:{text}'}],'isError':False}})
    else:
        write_msg({'jsonrpc':'2.0','id':message['id'],'error':{'message':'unsupported method'}})
"#
    }

    #[tokio::test]
    async fn initialize_request_round_trips() {
        let mut client = McpClient::start(&mock_server_config(framed_server_script()))
            .await
            .expect("client should start");

        client
            .initialize()
            .await
            .expect("initialize should succeed");
    }

    #[tokio::test]
    async fn list_tools_parses_remote_catalog() {
        let mut client = McpClient::start(&mock_server_config(framed_server_script()))
            .await
            .expect("client should start");
        client
            .initialize()
            .await
            .expect("initialize should succeed");

        let tools = client
            .list_tools()
            .await
            .expect("list_tools should succeed");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description, "Echo text");
        assert_eq!(tools[0].input_schema, json!({"type": "object"}));
    }

    #[tokio::test]
    async fn call_tool_round_trips_result() {
        let mut client = McpClient::start(&mock_server_config(framed_server_script()))
            .await
            .expect("client should start");
        client
            .initialize()
            .await
            .expect("initialize should succeed");

        let result = client
            .call_tool("echo", json!({"text": "hello"}))
            .await
            .expect("tool call should succeed");

        assert_eq!(result.content, "echo:hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn invalid_json_or_protocol_errors_are_reported() {
        let invalid_json = r#"
import sys
body = b'{not json}'
sys.stdout.buffer.write(f'Content-Length: {len(body)}\r\n\r\n'.encode('utf-8'))
sys.stdout.buffer.write(body)
sys.stdout.buffer.flush()
"#;
        let mut client = McpClient::start(&mock_server_config(invalid_json))
            .await
            .expect("client should start");

        let error = client
            .initialize()
            .await
            .expect_err("initialize should fail");
        assert!(matches!(error, McpError::Json(_) | McpError::Protocol(_)));
    }
}
