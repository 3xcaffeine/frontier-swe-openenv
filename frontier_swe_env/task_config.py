"""Task configuration for FrontierSWE environments."""

from pathlib import Path

from pydantic import BaseModel


class TaskConfig(BaseModel):
    task_name: str
    docker_image: str
    instruction: str
    workspace_dir: str
    build_command: str
    gate_script_path: str
    visible_test_command: str
    visible_test_total: int
    max_subtasks: int
    max_attempts_per_subtask: int
    episode_timeout_s: float
    # Scoring weights
    gate_weight: float = 0.30
    l1_weight: float = 0.70
    l2_weight: float = 0.30
    plan_weight: float = 0.25
    subtask_weight: float = 0.60
    completion_weight: float = 0.10
    tool_weight: float = 0.05
    # Agent LLM config (the model pi uses — the one being trained/evaluated)
    agent_model: str | None = None
    agent_provider: str | None = None
    agent_api_base_url: str | None = None
    agent_api_key: str | None = None
    # LLM judge config (L2/L3 rubrics — a separate, typically stronger model)
    grader_model: str | None = None
    grader_api_base_url: str | None = None
    grader_api_key: str | None = None
    # Container config
    container_port: int = 8000
    cpus: int = 8
    memory_mb: int = 32768


PG_TRAINING_INSTRUCTION = """
# PostgreSQL Wire Adapter — Basic Connection

Your workspace is `/app/postgres-sqlite`. It contains a Zig stub in `src/main.zig`.

Goal: Make the binary handle argv[0] dispatch and respond to basic queries.

1. When invoked as `initdb`, create the directory passed via `-D <path>`
2. When invoked as `pg_ctl start`, fork a background process on the port from `-p`
3. When invoked as `postgres`, listen on TCP on the given port
4. Handle the PostgreSQL wire protocol startup: StartupMessage → AuthenticationOk → ReadyForQuery
5. Handle simple query mode: Query message → parse SQL → execute via SQLite → return RowDescription + DataRow + CommandComplete

Build: `bash build.sh`
Smoke test: `bash /app/smoke_test.sh`
Compat test: `PG_PORT=55432 bash /app/pg_compat_test.sh`
Reference: `w3m /reference/postgresql-docs/html/protocol-flow.html`

You have 15 minutes. Get as many pg_compat_test.sh tiers passing as possible.
""".strip()


def pg_training_config() -> TaskConfig:
    return TaskConfig(
        task_name="postgres-sqlite-wire-adapter",
        docker_image="frontier-swe-pg:latest",
        instruction=PG_TRAINING_INSTRUCTION,
        workspace_dir="/app/postgres-sqlite",
        build_command="cd /app/postgres-sqlite && bash build.sh -Doptimize=ReleaseSafe",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="PG_PORT=55432 bash /app/pg_compat_test.sh",
        visible_test_total=72,
        max_subtasks=2,
        max_attempts_per_subtask=2,
        episode_timeout_s=900,
    )


def pg_demo_config() -> TaskConfig:
    instruction_path = (
        Path(__file__).parent.parent
        / "tasks"
        / "postgres-sqlite-wire-adapter"
        / "instruction.md"
    )
    instruction = (
        instruction_path.read_text()
        if instruction_path.exists()
        else PG_TRAINING_INSTRUCTION
    )

    return TaskConfig(
        task_name="postgres-sqlite-wire-adapter",
        docker_image="frontier-swe-pg:latest",
        instruction=instruction,
        workspace_dir="/app/postgres-sqlite",
        build_command="cd /app/postgres-sqlite && bash build.sh -Doptimize=ReleaseFast",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="PG_PORT=55432 bash /app/pg_compat_test.sh",
        visible_test_total=72,
        max_subtasks=4,
        max_attempts_per_subtask=3,
        episode_timeout_s=5400,
    )
