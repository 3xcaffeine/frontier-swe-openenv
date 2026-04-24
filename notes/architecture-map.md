# Architecture Map

## Major Components

1. Environment API and Orchestration
- `frontier_swe_env/server/app.py`
- `frontier_swe_env/server/frontier_swe_env_environment.py`
- `frontier_swe_env/server/mcp_tools.py`

Responsibilities:
- Boot FastAPI OpenEnv app.
- Mount FastMCP streamable endpoint for pi tool transport.
- Maintain one active environment session.
- Execute state transitions and scoring.

2. Data Models and Client
- `frontier_swe_env/models.py`
- `frontier_swe_env/client.py`

Responsibilities:
- Action/observation/state schemas.
- Host-side typed client for reset/step/state interactions.

3. Rubrics (Scoring Layers)
- `frontier_swe_env/rubrics/gate_checks.py`
- `frontier_swe_env/rubrics/l1_tests.py`
- `frontier_swe_env/rubrics/l2_code_review.py`
- `frontier_swe_env/rubrics/l3_plan_review.py`
- `frontier_swe_env/rubrics/episode_rubric.py`

Responsibilities:
- L1 deterministic checks via local subprocesses.
- L2/L3 LLM-based grading via OpenAI-compatible API.
- Final reward aggregation.

4. Task Registry and Config
- `frontier_swe_env/task_config.py`
- `frontier_swe_env/tasks/__init__.py`
- `frontier_swe_env/tasks/pg.py`

Responsibilities:
- Task metadata and scoring knobs.
- Registry-based task selection (`task_name`, `mode`).

5. Task Assets and Verifier
- `tasks/postgres-sqlite-wire-adapter/...`

Responsibilities:
- Agent prompt/instruction.
- Environment workspace scaffold.
- Visible compatibility tests.
- Hidden verifier and reward computation.

6. Container Build Stack
- `docker/Dockerfile.base`
- `docker/Dockerfile.pg`
- `docker/openenv_entrypoint.sh`

Responsibilities:
- Base runtime with OpenEnv, pi, and adapters.
- Task image with Zig + PostgreSQL docs/client + verifier artifacts.
- Runtime env-var wiring for agent and grader endpoints.

## Runtime Flow (Condensed)

1. Container starts OpenEnv FastAPI app.
2. `reset()` initializes episode, resets workspace git state, boots pi harness.
3. First `step()` prepends task instruction and sends to pi.
4. pi uses MCP tools (`submit_plan`, `submit_subtask`, `get_status`, `advance`).
5. Environment computes scores and phase transitions.
6. On completion or timeout, episode reward is produced.

## Contracts That Matter Later

MCP tools are the critical execution contract:
- `submit_plan(subtasks)`
- `submit_subtask(subtask_id)`
- `get_status()`
- `advance()`

State constraints:
- Plan can only be submitted in `PLANNING`.
- Subtask submission and advance only in `EXECUTING`.
- Attempts are capped per subtask.

## Configuration Surfaces

Agent model path:
- `FSWE_AGENT_MODEL`
- `FSWE_AGENT_PROVIDER`
- `FSWE_AGENT_API_URL`
- `FSWE_AGENT_API_KEY`

Grader model path:
- `FSWE_GRADER_MODEL`
- `FSWE_GRADER_API_URL`
- `FSWE_GRADER_API_KEY`

Task-level knobs live in `TaskConfig` (timeouts, weights, scripts, dimensions, etc.).
