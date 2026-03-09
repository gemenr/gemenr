//! Agent loop components such as tool dispatch strategies.

/// Tool dispatcher implementations and conversation message projections.
pub mod dispatcher;

pub use dispatcher::{
    ConversationMessage, NativeToolDispatcher, ParsedToolCall, SelectedToolDispatcher,
    ToolExecutionResult, XmlToolDispatcher,
};
