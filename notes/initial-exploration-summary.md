# Initial Exploration Summary (2026-04-24)

## What This Repository Is Building

This repository implements an OpenEnv-compatible software engineering environment for FrontierSWE-style tasks. The current production task is a PostgreSQL wire-compatibility challenge where the agent must implement a Zig binary that behaves like PostgreSQL server-side tools (`postgres`, `initdb`, `pg_ctl`) while using SQLite underneath.

The system is designed to support:
- Interactive agent execution through a gym-like environment API.
- Structured episode workflows (plan, execute, submit, advance).
- Multi-layer scoring (deterministic checks plus LLM judge layers).
- Task-specific containers and verifier harnesses.
- Future task onboarding with minimal core-env changes.

## Core Runtime Shape

- OpenEnv FastAPI app exposes environment endpoints.
- A mounted FastMCP server at `/tools/mcp` is used by pi-mcp-adapter.
- The environment class owns episode state and scoring logic.
- The pi agent runs as a subprocess through `PiHarnessAdapter`.

High-level episode phases:
1. `PLANNING`: agent must call `submit_plan`.
2. `EXECUTING`: agent iterates on subtasks using `submit_subtask` and `advance`.
3. `DONE`: final episode reward is computed.

## Scoring Model (Current)

Per-subtask scoring pipeline:
- L1a gate checks (script, deterministic).
- L1b visible compatibility tests (deterministic).
- L2 code review (LLM judge over git diff).
- Blended subtask score = `(1 - l2_weight) * L1 + l2_weight * L2`.
- Best blended score per subtask is frozen.

Episode-level reward includes:
- L3 plan quality score.
- Mean frozen subtask score.
- Completion ratio.
- Tool usage density.

## Current Task Scope

Task: `postgres-sqlite-wire-adapter`
- Workspace stub starts from a minimal Zig program that currently exits with an error.
- Build path is driven by `build.sh` using `zig build-exe`.
- Visible tests: `pg_compat_test.sh` (72 checks across 9 tiers).
- Hidden verifier: PostgreSQL 18 regression + TAP harness integration.
- Hard-fail constraints enforce source hygiene, dependency restrictions, and binary availability.

## Notable Design Direction

Recent merged architecture refactor (PR #10) confirms task-agnostic intent:
- Task configs moved to a registry model.
- L1/L2/L3 rubrics parameterized for reusable task onboarding.
- Core environment and MCP contract intended to remain stable while tasks vary.

## Immediate Risks Observed

- Reward-hacking risk is explicitly tracked as an open issue.
- README is empty, so external onboarding and reproducibility docs are currently weak.
- No open PRs at snapshot time, suggesting issue threads are currently the active planning surface.
