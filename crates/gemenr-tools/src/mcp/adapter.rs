use std::sync::Arc;

use async_trait::async_trait;
use gemenr_core::{McpConfig, RiskLevel, ToolSpec};
use tokio::sync::Mutex;

use crate::ToolPlane;
use crate::handler::{ExecContext, ToolError, ToolHandler, ToolOutput};
use crate::mcp::client::{McpClient, McpError, McpRemoteTool};

/// Build the namespaced local tool name for one remote MCP tool.
#[must_use]
pub fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!("mcp.{server_name}.{tool_name}")
}

/// Adapter that forwards one local tool invocation to a remote MCP tool.
pub struct McpToolAdapter {
    client: Arc<Mutex<McpClient>>,
    server_name: String,
    remote_name: String,
}

impl McpToolAdapter {
    /// Create a new adapter for one namespaced MCP tool.
    #[must_use]
    pub fn new(client: Arc<Mutex<McpClient>>, server_name: String, remote_name: String) -> Self {
        Self {
            client,
            server_name,
            remote_name,
        }
    }
}

#[async_trait]
impl ToolHandler for McpToolAdapter {
    async fn execute(
        &self,
        _ctx: &ExecContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let mut client = self.client.lock().await;
        let result = client
            .call_tool(&self.remote_name, args)
            .await
            .map_err(mcp_error_to_tool_error)?;

        if result.is_error {
            Err(ToolError::Execution {
                exit_code: None,
                stderr: format!(
                    "remote MCP tool `{}` on server `{}` returned an error: {}",
                    self.remote_name, self.server_name, result.content
                ),
            })
        } else {
            Ok(ToolOutput {
                content: result.content,
            })
        }
    }
}

/// Start enabled MCP servers and register their remote tools into one tool plane.
pub async fn register_mcp_servers(
    plane: &mut ToolPlane,
    config: &McpConfig,
) -> Result<(), McpError> {
    for server in config.servers.iter().filter(|server| server.enabled) {
        let mut client = McpClient::start(server).await?;
        client.initialize().await?;
        let tools = client.list_tools().await?;
        let shared_client = Arc::new(Mutex::new(client));

        for remote in tools {
            let spec = remote_tool_spec(&server.name, &remote);
            let handler = McpToolAdapter::new(
                Arc::clone(&shared_client),
                server.name.clone(),
                remote.name.clone(),
            );
            plane.register(spec, Box::new(handler));
        }
    }

    Ok(())
}

fn remote_tool_spec(server_name: &str, remote: &McpRemoteTool) -> ToolSpec {
    ToolSpec {
        name: mcp_tool_name(server_name, &remote.name),
        description: remote.description.clone(),
        input_schema: remote.input_schema.clone(),
        risk_level: RiskLevel::Low,
    }
}

fn mcp_error_to_tool_error(error: McpError) -> ToolError {
    ToolError::Execution {
        exit_code: None,
        stderr: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use gemenr_core::{McpServerConfig, ToolInvoker};
    use serde_json::json;
    use tokio::sync::Mutex;

    use super::{McpToolAdapter, mcp_tool_name, register_mcp_servers};
    use crate::ToolPlane;
    use crate::handler::{ExecContext, ToolHandler};
    use crate::mcp::client::{McpClient, McpRemoteTool};

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
"#
    }

    fn server_config() -> McpServerConfig {
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

    #[test]
    fn namespaced_tool_names_are_stable() {
        assert_eq!(mcp_tool_name("server", "echo"), "mcp.server.echo");
    }

    #[tokio::test]
    async fn tool_handler_bridges_to_mcp_client() {
        let mut client = McpClient::start(&server_config())
            .await
            .expect("client should start");
        client
            .initialize()
            .await
            .expect("initialize should succeed");
        let adapter = McpToolAdapter::new(
            Arc::new(Mutex::new(client)),
            "mock".to_string(),
            "echo".to_string(),
        );

        let output = adapter
            .execute(&ExecContext::default(), json!({"text": "hello"}))
            .await
            .expect("adapter call should succeed");

        assert_eq!(output.content, "echo:hello");
    }

    #[tokio::test]
    async fn enabled_servers_register_namespaced_tools() {
        let mut plane = ToolPlane::new();
        let config = gemenr_core::McpConfig {
            servers: vec![server_config()],
        };

        register_mcp_servers(&mut plane, &config)
            .await
            .expect("registration should succeed");

        assert!(plane.lookup("mcp.mock.echo").is_some());
        assert!(
            plane
                .list_specs()
                .iter()
                .any(|spec| spec.name == "mcp.mock.echo")
        );
    }

    #[test]
    fn remote_tool_schema_maps_into_tool_spec_shape() {
        let remote = McpRemoteTool {
            name: "echo".to_string(),
            description: "Echo text".to_string(),
            input_schema: json!({"type": "object"}),
        };

        let spec = super::remote_tool_spec("mock", &remote);
        assert_eq!(spec.name, "mcp.mock.echo");
        assert_eq!(spec.input_schema, json!({"type": "object"}));
    }
}
