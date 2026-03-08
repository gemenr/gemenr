//! Agent loop components such as tool dispatch strategies.

pub mod dispatcher;

pub use dispatcher::{
    ConversationMessage, NativeToolDispatcher, ParsedToolCall, ToolDispatcher, ToolExecutionResult,
    XmlToolDispatcher,
};
