use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tokio::sync::RwLock;

use crate::agent::dispatcher::{NativeToolDispatcher, ToolDispatcher, XmlToolDispatcher};
use crate::context::{ContextManager, SoulManager, TapeStore, TokenBudget};
use crate::kernel::{AgentRuntime, PromptComposer, TurnController};
use crate::model::ModelProvider;
use crate::protocol::SessionId;
use crate::tool_invoker::ToolInvoker;

/// Assembles [`AgentRuntime`] instances with shared resources.
pub struct RuntimeBuilder {
    /// Shared model provider.
    model: Arc<dyn ModelProvider>,
    /// Shared tool invoker.
    tools: Arc<dyn ToolInvoker>,
    /// Shared SOUL manager.
    soul: Arc<RwLock<SoulManager>>,
    /// Shared tape store.
    tape_store: Arc<dyn TapeStore>,
    /// Tool dispatcher override (`native`, `xml`, or `auto`).
    tool_dispatcher_config: String,
    /// Model identifier for requests.
    model_name: String,
    /// Maximum tokens for requests.
    max_tokens: Option<u32>,
    /// Token budget for context building.
    budget: TokenBudget,
}

impl RuntimeBuilder {
    /// Create a new builder with shared resources.
    #[must_use]
    pub fn new(
        model: Arc<dyn ModelProvider>,
        tools: Arc<dyn ToolInvoker>,
        soul: Arc<RwLock<SoulManager>>,
        tape_store: Arc<dyn TapeStore>,
    ) -> Self {
        Self {
            model,
            tools,
            soul,
            tape_store,
            tool_dispatcher_config: "auto".to_string(),
            model_name: String::new(),
            max_tokens: None,
            budget: TokenBudget::default(),
        }
    }

    /// Set the model identifier used for requests.
    #[must_use]
    pub fn model_name(mut self, name: String) -> Self {
        self.model_name = name;
        self
    }

    /// Set the maximum tokens used for requests.
    #[must_use]
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set the tool dispatcher selection mode.
    #[must_use]
    pub fn tool_dispatcher(mut self, config: String) -> Self {
        self.tool_dispatcher_config = config;
        self
    }

    /// Set the token budget used for context building.
    #[must_use]
    pub fn token_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Build an independent runtime for one task or conversation.
    #[must_use]
    pub fn build(&self, system_prompt: String) -> AgentRuntime {
        let tool_dispatcher: Box<dyn ToolDispatcher> = match self.tool_dispatcher_config.as_str() {
            "native" => Box::new(NativeToolDispatcher),
            "xml" => Box::new(XmlToolDispatcher),
            _ if self.model.supports_native_tools() => Box::new(NativeToolDispatcher),
            _ => Box::new(XmlToolDispatcher),
        };

        AgentRuntime::new(
            ContextManager::new(SessionId::new(), self.tape_store.clone(), self.soul.clone()),
            self.model.clone(),
            self.tools.clone(),
            tool_dispatcher,
            PromptComposer,
            TurnController,
            self.budget.clone(),
            system_prompt,
            self.model_name.clone(),
            self.max_tokens,
            Arc::new(AtomicBool::new(false)),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use tokio::sync::RwLock;

    use super::RuntimeBuilder;
    use crate::context::{InMemoryTapeStore, SoulManager, TapeStore};
    use crate::error::ModelError;
    use crate::model::{
        ChatRequest, ChatResponse, FinishReason, ModelCapabilities, ModelProvider, ModelRequest,
        ModelResponse,
    };
    use crate::tool_invoker::{PolicyDecision, ToolInvokeError, ToolInvokeResult, ToolInvoker};
    use crate::tool_spec::{RiskLevel, ToolSpec};

    struct RecordingModelProvider {
        responses: Mutex<VecDeque<ChatResponse>>,
        requests: Mutex<Vec<ChatRequest>>,
        capabilities: ModelCapabilities,
    }

    impl RecordingModelProvider {
        fn new(capabilities: ModelCapabilities) -> Self {
            Self {
                responses: Mutex::new(
                    vec![ChatResponse {
                        text: Some("done".to_string()),
                        tool_calls: Vec::new(),
                        usage: None,
                    }]
                    .into(),
                ),
                requests: Mutex::new(Vec::new()),
                capabilities,
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl ModelProvider for RecordingModelProvider {
        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
            Ok(ModelResponse {
                content: request
                    .messages
                    .last()
                    .map(|message| message.content.clone())
                    .unwrap_or_default(),
                finish_reason: FinishReason::Stop,
            })
        }

        fn capabilities(&self) -> ModelCapabilities {
            self.capabilities
        }

        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            self.requests
                .lock()
                .expect("requests lock should not be poisoned")
                .push(request);
            self.responses
                .lock()
                .expect("responses lock should not be poisoned")
                .pop_front()
                .ok_or_else(|| ModelError::Api {
                    status: 500,
                    message: "no scripted response available".to_string(),
                })
        }
    }

    struct StaticToolInvoker {
        specs: HashMap<String, ToolSpec>,
    }

    impl StaticToolInvoker {
        fn new(specs: Vec<ToolSpec>) -> Self {
            Self {
                specs: specs
                    .into_iter()
                    .map(|spec| (spec.name.clone(), spec))
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl ToolInvoker for StaticToolInvoker {
        fn lookup(&self, name: &str) -> Option<&ToolSpec> {
            self.specs.get(name)
        }

        fn list_specs(&self) -> Vec<ToolSpec> {
            self.specs.values().cloned().collect()
        }

        fn check_policy(&self, _name: &str, _arguments: &serde_json::Value) -> PolicyDecision {
            PolicyDecision::Allow
        }

        async fn invoke(
            &self,
            _call_id: &str,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<ToolInvokeResult, ToolInvokeError> {
            Ok(ToolInvokeResult {
                content: String::new(),
                is_error: false,
            })
        }
    }

    fn sample_tool() -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: "Echo content".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
            risk_level: RiskLevel::Low,
        }
    }

    fn soul() -> Arc<RwLock<SoulManager>> {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-builder");
        std::fs::create_dir_all(&workspace).expect("test workspace should be created");
        Arc::new(RwLock::new(
            SoulManager::load(&workspace).expect("SOUL.md should load"),
        ))
    }

    #[tokio::test]
    async fn auto_selects_native_dispatcher_when_provider_supports_it() {
        let model = Arc::new(RecordingModelProvider::new(ModelCapabilities {
            native_tool_calling: true,
            vision: false,
        }));
        let model_for_assertions = model.clone();
        let tools = Arc::new(StaticToolInvoker::new(vec![sample_tool()]));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let builder = RuntimeBuilder::new(model, tools, soul(), tape_store)
            .tool_dispatcher("auto".to_string())
            .model_name("test-model".to_string());
        let mut runtime = builder.build("system".to_string());

        runtime
            .run_turn("hello")
            .await
            .expect("turn should succeed");

        assert!(model_for_assertions.requests()[0].tools.is_some());
    }

    #[tokio::test]
    async fn auto_selects_xml_dispatcher_when_provider_lacks_native_tools() {
        let model = Arc::new(RecordingModelProvider::new(ModelCapabilities::default()));
        let model_for_assertions = model.clone();
        let tools = Arc::new(StaticToolInvoker::new(vec![sample_tool()]));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let builder = RuntimeBuilder::new(model, tools, soul(), tape_store)
            .tool_dispatcher("auto".to_string())
            .model_name("test-model".to_string());
        let mut runtime = builder.build("system".to_string());

        runtime
            .run_turn("hello")
            .await
            .expect("turn should succeed");

        let request = &model_for_assertions.requests()[0];
        assert!(request.tools.is_none());
        assert!(
            request.messages[0]
                .content
                .contains("To call a tool, output a <tool_call> tag")
        );
    }

    #[tokio::test]
    async fn explicit_dispatcher_configuration_overrides_provider_capability() {
        let model = Arc::new(RecordingModelProvider::new(ModelCapabilities {
            native_tool_calling: true,
            vision: false,
        }));
        let model_for_assertions = model.clone();
        let tools = Arc::new(StaticToolInvoker::new(vec![sample_tool()]));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let builder = RuntimeBuilder::new(model, tools, soul(), tape_store)
            .tool_dispatcher("xml".to_string())
            .model_name("test-model".to_string());
        let mut runtime = builder.build("system".to_string());

        runtime
            .run_turn("hello")
            .await
            .expect("turn should succeed");

        assert!(model_for_assertions.requests()[0].tools.is_none());
    }

    #[test]
    fn build_creates_independent_sessions() {
        let model = Arc::new(RecordingModelProvider::new(ModelCapabilities::default()));
        let tools = Arc::new(StaticToolInvoker::new(vec![sample_tool()]));
        let tape_store: Arc<dyn TapeStore> = Arc::new(InMemoryTapeStore::new());
        let builder = RuntimeBuilder::new(model, tools, soul(), tape_store)
            .tool_dispatcher("auto".to_string())
            .model_name("test-model".to_string());

        let runtime_one = builder.build("system".to_string());
        let runtime_two = builder.build("system".to_string());

        assert_ne!(runtime_one.session_id(), runtime_two.session_id());
    }
}
