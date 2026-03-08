# AGENTS.md — Gemenr Engineering Guide

This file defines the working protocol for coding agents in this repository.
Scope: entire repository.

## 1) Project Snapshot

Gemenr is now a **multi-entry Rust workspace** for running an LLM agent in three modes:

- **interactive CLI** (`gemenr-cli chat` / `gemenr-cli run`)
- **scheduled automation** (`gemenr-cli daemon`)
- **IM / Lark bot service** (`gemenr-im`)

The core architecture is already beyond the earlier “minimal kernel only” phase:

- `gemenr-core` contains the runtime kernel, context/tape persistence, model abstractions, access-layer types, and runtime/session management
- `gemenr-tools` contains the actual tool registry, built-in tools, policy evaluation, sandbox adapters, and MCP client code
- `gemenr-cli` and `gemenr-im` are thin composition layers that assemble providers, tools, persistence, and transport adapters

## 2) Current Workspace Layout

```text
gemenr/
├── Cargo.toml
├── gemenr.toml(.example)       # runtime config
├── bins/
│   ├── gemenr-cli/             # chat, run, daemon
│   └── gemenr-im/              # Lark / Feishu long-connection service
└── crates/
    ├── gemenr-core/            # runtime, context, config, provider/router, access layer
    └── gemenr-tools/           # tool plane, builtins, policy, sandbox, MCP
```

Runtime state is typically stored under a workspace-local `.gemenr/` directory:

- `.gemenr/SOUL.md` — long-lived agent memory/preferences
- `.gemenr/tapes/` — persisted JSONL session tapes when disk persistence is available

## 3) Architecture Map

### 3.1 `gemenr-core`

Treat `gemenr-core` as the stable center of the system.

- `builder.rs` — assembles `AgentRuntime` from shared `Arc` dependencies
- `kernel/` — prompt composition, turn loop, cancellation, approvals, event emission
- `context/` — tape-backed event history, anchors, summarization thresholds, `SOUL.md`
- `model/` — `ModelProvider`, `ModelRouter`, Anthropic-compatible provider implementation
- `agent/dispatcher.rs` — native-tool vs XML-tool parsing/formatting strategy
- `tool_invoker.rs` / `tool_spec.rs` — runtime-facing tool contract and risk metadata
- `access/` — normalized inbound/outbound messages plus route parsing and transport abstraction
- `runtime_manager.rs` — multi-conversation lifecycle for long-lived transports like Lark
- `config.rs` — validated `gemenr.toml` loader for models, fallback, access, cron, policy, MCP

### 3.2 `gemenr-tools`

Treat `gemenr-tools` as the execution plane.

- `ToolPlane` is the concrete `ToolInvoker` implementation used by binaries
- built-ins currently registered by binaries:
  - `shell`
  - `fs.read`
  - `fs.write`
  - `update_soul`
- `policy.rs` converts scoped config into `ExecutionPolicy`
- `sandbox/` selects `Seatbelt` on macOS or `Landlock` on Linux when policy requires it
- `mcp/` implements stdio MCP client + remote tool adapter

Important: MCP support exists in the library layer, but the current binary builders only register built-in tools. Do not assume configured MCP servers are live unless you wire them into the entrypoint explicitly.

### 3.3 Binaries

- `bins/gemenr-cli`
  - `chat` uses stdio transport for interactive conversation
  - `run` executes one task, optionally restoring a prior session
  - `daemon` schedules cron jobs and can report back to `stdio:` or `lark:...`
- `bins/gemenr-im`
  - runs the Lark adapter/service loop
  - uses `RuntimeManager` to multiplex many conversations onto shared runtime resources
  - hibernates idle conversations and restores them from tape on demand

## 4) Runtime Flow You Should Preserve

The real runtime path today is:

1. load validated config from `gemenr.toml`
2. build model provider (usually `AnthropicProvider` wrapped by `ModelRouter`)
3. build `SoulManager`, `TapeStore`, and `ToolPlane`
4. create `RuntimeBuilder`
5. select tool dispatcher mode: `native`, `xml`, or `auto`
6. `AgentRuntime::run_turn()` appends user input to tape
7. context is rebuilt from anchor + post-anchor events
8. if budget threshold is exceeded, summarize and create a new anchor
9. prompt is composed from `SOUL.md` + system prompt + context + tool instructions/specs
10. model response is parsed into text and/or tool calls
11. policy + approval gates are checked before tool execution
12. tool results are appended as events and fed back into the next loop iteration

Do not bypass this pipeline casually. If you need new behavior, prefer extending an existing seam (`ModelProvider`, `ToolInvoker`, `ApprovalHandler`, `EventSink`, `AccessAdapter`, `ConversationDriver`) instead of inserting ad-hoc side paths.

## 5) Architectural Boundaries

- `gemenr-core` must not depend on `gemenr-tools`, `gemenr-cli`, or `gemenr-im`
- `gemenr-tools` may depend on `gemenr-core`, but not on binaries
- binaries are composition roots; keep integration glue there
- transport-specific logic belongs in `bins/` or `access/`, not in the kernel
- tool execution policy belongs in `gemenr-tools`, not in model/provider code
- config parsing/validation belongs in `config.rs`; avoid duplicating config rules elsewhere

If you find yourself adding cross-crate helpers, first ask whether the change is actually an entrypoint concern that should stay in a binary.

## 6) Current Design Facts That Matter

- the runtime is **per session / per conversation**, not global
- shared heavy resources are injected via `Arc`
- the context model is **event tape + optional anchor summary**, not a mutable transcript object
- summarization is budget-driven and uses the model itself
- tool calling supports two provider styles:
  - native structured tool calls
  - XML-tagged tool calls embedded in plain text
- provider fallback is already supported via `ModelRouter`
- cron jobs may apply a per-job tool allowlist via `allowlist_tool_invoker(...)`
- `gemenr-im` is intentionally long-lived and reconnecting; avoid request/response assumptions that only make sense for CLI mode

## 7) Configuration Rules

`gemenr.toml` is part of the public operator surface.

- prefer extending the typed config model in `crates/gemenr-core/src/config.rs`
- keep `serde(deny_unknown_fields)` discipline for new config sections unless there is a strong compatibility reason not to
- validate eagerly and return explicit `ConfigError::Invalid(...)`
- preserve env var override behavior where it already exists
- do not log, print, or commit secrets from config or environment

Be careful with examples and tests: use placeholder credentials only.

## 8) Code Style and Conventions

- use standard Rust naming and module layout
- all public items should have doc comments, matching existing crate style
- use `thiserror` for library-facing error types
- prefer `tracing` over `println!` / `eprintln!` outside CLI-facing UX paths
- keep async boundaries explicit; favor small focused async functions over deeply nested control flow
- prefer straightforward structs/enums over speculative abstractions
- preserve the project’s habit of colocated unit tests in the same file as the implementation

## 9) Change Guidelines

### 9.1 When modifying the runtime

- preserve event emission and tape persistence semantics
- keep cancellation checks in long-running loops / tool execution paths
- preserve the max-step protection in the turn loop unless changing it intentionally
- if you add new event kinds or payload shapes, update both producers and any context reconstruction logic that depends on them

### 9.2 When adding tools

- register them through `ToolPlane`
- provide a stable `ToolSpec` with accurate `risk_level`
- ensure policy behavior is sensible for low/medium/high-risk tools
- if the tool shells out or touches the filesystem, think through sandbox behavior and failure surfacing
- prefer small JSON schemas with explicit required fields

### 9.3 When adding transports or long-lived services

- normalize inbound traffic into `AccessInbound`
- return outbound traffic as `AccessOutbound`
- use `RuntimeManager` when one service must host many conversations over time
- keep reconnection/backoff and idle reclamation outside the kernel

## 10) Build, Test, and Validate

Run these from `gemenr/` when relevant:

```bash
cargo fmt --all
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Useful entrypoints:

```bash
cargo run --bin gemenr-cli -- chat
cargo run --bin gemenr-cli -- run "your task"
cargo run --bin gemenr-cli -- daemon
cargo run --bin gemenr-im
```

## 11) Practical Advice for Agents

- first identify whether the change belongs to `core`, `tools`, or a binary composition root
- prefer updating `RuntimeBuilder` assembly code rather than duplicating wiring in multiple places
- if behavior differs between CLI and IM, keep the shared mechanism in `core` and the policy/transport differences in the binaries
- when a feature appears “configured but inactive”, verify whether the entrypoint actually wires it up before assuming it works
- keep docs and config examples aligned with the real implementation status; this repository has already drifted once

## 12) References

- `crates/gemenr-core/src/kernel/mod.rs` — runtime turn loop and event flow
- `crates/gemenr-core/src/context/mod.rs` — tape/anchor context model
- `crates/gemenr-core/src/config.rs` — config schema and validation
- `crates/gemenr-tools/src/lib.rs` — tool plane and allowlist wrapper
- `bins/gemenr-cli/src/main.rs` — CLI composition root
- `bins/gemenr-cli/src/daemon.rs` — cron execution flow
- `bins/gemenr-im/src/main.rs` — IM composition root
- `bins/gemenr-im/src/service.rs` — reconnect/hibernate loop
