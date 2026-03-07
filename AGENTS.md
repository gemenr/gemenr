# AGENTS.md — Gemenr Engineering Protocol

This file defines the working protocol for coding agents in this repository.
Scope: entire repository.

## 1) Project Overview

Gemenr is a tool-calling agent runtime built in Rust. It enables an LLM-based agent
to autonomously plan, select tools, and execute multi-step tasks.

Key characteristics:

- **Minimal kernel** — agent loop, context management, and event protocol live in `gemenr-core`; tools, policies, and platform integrations are external
- **Non-streaming model interface** — `ModelProvider::complete()` returns a full response; providers may use streaming APIs internally but accumulate before returning
- **Per-task isolation** — each task/conversation gets an independent `AgentRuntime` with its own context; shared resources (`ModelProvider`, `ToolPlane`) are injected via `Arc`
- **Tape + anchor context** — append-only event log with anchor checkpoints for stage boundaries and automatic summarization
- **SOUL.md continuous learning** — agent-editable markdown file for identity, preferences, and accumulated experience

## 2) Crate Structure

```
gemenr/
├── Cargo.toml              # workspace root
├── crates/
│   └── gemenr-core/        # message types, ModelProvider trait, provider impls, config
├── bins/
│   └── gemenr-cli/         # CLI entry point (chat mode, task mode)
```

Phase 1 will add `crates/gemenr-tools/` for the tool system (registration, execution, built-in tools).

Key extension points (Phase 1+):

- `gemenr-core` — `ModelProvider` trait (model interaction), `TapeStore` trait (event persistence)
- `gemenr-tools` — `ToolHandler` trait (tool execution), `ToolSpec` (tool definition)

## 3) Build & Test

```bash
# Check compilation
cargo check --workspace

# Format code
cargo fmt --all

# Lint — fix ALL warnings before committing
cargo clippy --workspace --all-targets -- -D warnings

# Run all tests
cargo test --workspace

# Run a specific test
cargo test --workspace test_name

# Run with logging
RUST_LOG=gemenr=debug cargo run --bin gemenr-cli
```

## 4) Engineering Principles

### 4.1 KISS — Keep It Simple

- Prefer straightforward control flow over clever abstractions.
- Prefer explicit `match` branches and typed structs over dynamic dispatch when possible.
- Keep error paths obvious and localized.

### 4.2 YAGNI — You Aren't Gonna Need It

- Do not add config keys, trait methods, or feature flags without a concrete current use case.
- Do not introduce speculative abstractions without at least one real caller.
- Unsupported paths should error out explicitly, not silently degrade.

### 4.3 DRY + Rule of Three

- Duplicate small, local logic when it preserves clarity.
- Extract shared utilities only after repeated, stable patterns (three or more occurrences).
- When extracting, preserve module boundaries and avoid hidden coupling.

### 4.4 Single Responsibility

- Each module focuses on one concern.
- Extend behavior by implementing existing narrow traits.
- Avoid god-modules that mix policy, transport, and storage.

### 4.5 Fail Fast + Explicit Errors

- Use typed errors (`thiserror`); avoid opaque `anyhow` in library code.
- Never silently swallow errors or broaden behavior on failure.
- Document intentional fallback behavior when it exists.

## 5) Code Style

- Follow standard Rust naming: `snake_case` for modules/functions/variables, `PascalCase` for types/traits/enums, `SCREAMING_SNAKE_CASE` for constants.
- All public items must have doc comments.
- Keep `unsafe` usage to zero unless absolutely justified with a `// SAFETY:` comment.
- Prefer `#[must_use]` on functions that return values callers should not ignore.
- Use `tracing` for structured logging. Avoid `println!`/`eprintln!` in library code.

## 6) Architecture Boundaries

- Concrete integrations depend on trait/config layers, not on each other.
- `gemenr-core` must not depend on `gemenr-tools` or `gemenr-cli`.
- `gemenr-tools` depends on `gemenr-core` for types; `gemenr-cli` depends on both.
- New shared abstractions require rule-of-three justification.
- Config keys and CLI flags are public API — treat additions and removals as breaking changes.

## 7) Commit & PR Discipline

- Use conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Keep commits small and focused — one concern per commit.
- Run `cargo fmt`, `cargo clippy`, and `cargo test` before committing.
- Do not mix formatting-only changes with functional changes.
- Do not modify unrelated modules in the same commit.

## 8) Design References

Architecture and requirements are documented in `../design-docs/`:

- `requirement.md` — phased requirements (MVP → Phase 1 → Phase 2 → Phase 3)
- `agent-design.md` — architecture design, module layouts, type definitions, state machines
