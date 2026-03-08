//! MCP stdio client and tool adapter integration.

/// MCP tool-plane adapters.
pub mod adapter;
/// MCP stdio client primitives.
pub mod client;

pub use adapter::{McpToolAdapter, mcp_tool_name, register_mcp_servers};
pub use client::{McpClient, McpError, McpRemoteTool, McpToolResult};
