# Exploration Checkpoints And Learnings

## Completed Checkpoints

- Mapped repository structure and identified core runtime surfaces.
- Traced environment lifecycle: reset, first-step instruction injection, state transitions, timeout handling.
- Verified MCP tool contract and transport split (`/mcp` vs `/tools/mcp`).
- Decomposed scoring stack (gate, visible tests, L2 judge, L3 plan, episode reward blend).
- Traced task packaging from Docker build through verifier and reward writer.
- Collected GitHub issue/PR context to understand direction and risks.

## Key Learnings

- This repo is no longer PG-only at the architecture layer; it is intentionally task-agnostic now.
- The current concrete benchmark task remains the PostgreSQL-over-SQLite Zig challenge.
- The visible test suite is meaningful but only partial; hidden regression and TAP suites dominate final realism.
- L2/L3 graders are OpenAI-compatible and configurable, which makes eval behavior highly environment-dependent.
- Reward-hacking is a known concern and already acknowledged in project planning.

## Operational Notes For Later Execution

- If behavior looks inconsistent, check both:
  - Task-level `TaskConfig` values.
  - Runtime `FSWE_AGENT_*` and `FSWE_GRADER_*` env vars.
- pi must connect through streamable MCP endpoint (`/tools/mcp`) for adapter compatibility.
- Workspace reset relies on git checkout + clean inside the task workspace.
- The current Zig workspace starts as a failing stub by design; successful episodes require incremental implementation.

## Suggested Next Checkpoints (When Continuing)

1. Add a concise, technical top-level README with architecture + runbook + HF links.
2. Create a task-onboarding template for adding new tasks with minimal core changes.
3. Add anti-reward-hacking mitigations tracked in issue #9 and test them explicitly.
4. Validate one end-to-end baseline run and archive logs plus metrics under docs or artifacts.
5. Add a lightweight architecture decision log for future task additions and scoring changes.
