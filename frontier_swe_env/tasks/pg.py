"""PostgreSQL wire-adapter task configuration."""

from pathlib import Path

from ..task_config import TaskConfig

PG_TRAINING_INSTRUCTION = """
# PostgreSQL Wire Adapter — Basic Connection

Your workspace is `/app/postgres-sqlite`. It contains a Zig stub in `src/main.zig`.

## Goal

Make the binary handle argv[0] dispatch and respond to basic queries.

1. When invoked as `initdb`, create the directory passed via `-D <path>`
2. When invoked as `pg_ctl start`, fork a background process on the port from `-p`
3. When invoked as `postgres`, listen on TCP on the given port
4. Handle the PostgreSQL wire protocol startup: StartupMessage → AuthenticationOk → ReadyForQuery
5. Handle simple query mode: Query message → parse SQL → execute via SQLite → return RowDescription + DataRow + CommandComplete

## Useful commands

- Build: `bash build.sh`
- Smoke test: `bash /app/smoke_test.sh`
- Compat test: `PG_PORT=55432 bash /app/pg_compat_test.sh`
- PG wire protocol docs: `w3m /reference/postgresql-docs/html/protocol-flow.html`

## Episode workflow

You MUST follow this workflow — your code is only scored when you use these tools.

**IMPORTANT: You have only 15 minutes. Break work into 3-5 small subtasks that
you can each code, test, and submit within 2-3 minutes. Do NOT plan huge subtasks
that try to do everything at once. Submit early and often — even partial progress
gets scored. A submitted imperfect subtask is worth infinitely more than an
unsubmitted perfect one.**

1. **Plan first.** Call `submit_plan` with 3-5 small, incremental subtasks.
   Each subtask needs:
   - `id`: a short identifier (e.g. "S1")
   - `description`: one specific thing you'll implement
   - `acceptance_criteria`: how to know it works

   Good plan (small, incremental):
   ```
   submit_plan({"subtasks": [
     {"id": "S1", "description": "argv[0] dispatch: initdb creates dir, pg_ctl forks", "acceptance_criteria": "bash build.sh succeeds and initdb -D /tmp/test creates dir"},
     {"id": "S2", "description": "TCP listener on given port", "acceptance_criteria": "nc -z 127.0.0.1 PORT succeeds"},
     {"id": "S3", "description": "Wire protocol handshake: StartupMessage, AuthOk, ReadyForQuery", "acceptance_criteria": "psql can connect without hanging"},
     {"id": "S4", "description": "Simple query: SELECT 1 returns result", "acceptance_criteria": "pg_compat_test.sh tier 1 passes"}
   ]})
   ```

   Bad plan (too broad): "Implement everything" in 1-2 subtasks.

2. **Code the current subtask.** Keep changes small and focused.
   Build and test frequently: `bash build.sh && bash /app/smoke_test.sh`

3. **Submit for scoring as soon as basic functionality works.**
   Call `submit_subtask` with the current subtask id:
   ```
   submit_subtask({"subtask_id": "S1"})
   ```
   The response contains:
   - `score`: your blended score (0.0-1.0)
   - `feedback`: specific issues — **read this carefully**
   - `attempts_remaining`: retries left

   You get **2 attempts per subtask**. If your score is low and you have
   attempts remaining, fix the issues from `feedback` and resubmit.
   Do NOT call `advance` on a low score when you still have attempts left.

4. **Advance.** Call `advance` to freeze your score and move on.
   Only advance when satisfied or out of attempts.

5. **Check progress.** Call `get_status` to see phase, scores, remaining time.

You have 15 minutes. Get as many pg_compat_test.sh tiers passing as possible.

**Remember: submit_subtask early. An imperfect submission that gets feedback
is better than running out of time with no submissions.**
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
        max_subtasks=5,
        max_attempts_per_subtask=2,
        episode_timeout_s=900,
        per_turn_timeout_s=180,
        task_description="A PostgreSQL wire-compatible adapter written in Zig that translates PG protocol to SQLite",
        task_domain="systems programming",
        scoring_context="L1 runs pg_compat_test.sh (72 graded SQL tests across 9 tiers)",
    )


def pg_demo_config() -> TaskConfig:
    instruction_path = (
        Path(__file__).parent.parent.parent
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
        max_subtasks=8,
        max_attempts_per_subtask=3,
        episode_timeout_s=5400,
        per_turn_timeout_s=600,
        task_description="A PostgreSQL wire-compatible adapter written in Zig that translates PG protocol to SQLite",
        task_domain="systems programming",
        scoring_context="L1 runs pg_compat_test.sh (72 graded SQL tests across 9 tiers)",
    )
