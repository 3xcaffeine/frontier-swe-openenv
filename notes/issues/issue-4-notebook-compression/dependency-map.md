# Dependency Map (Pre-Execution)

This map defines strict ordering and prerequisites before code changes.

## Dependency Graph

1. Freeze upstream references
- Depends on: none
- Output: pinned upstream commit URL and task folder links

2. Capture contract requirements
- Depends on: (1)
- Inputs: upstream `instruction.md`, `task.toml`, verifier scripts
- Output: local requirements summary and risk list

3. Scaffold local task assets
- Depends on: (2)
- Inputs: local postgres task layout + upstream notebook task layout
- Output: `tasks/notebook-compression/` structure in this repo

4. Wire task config and registry
- Depends on: (3)
- Inputs: `frontier_swe_env/task_config.py`, `frontier_swe_env/tasks/__init__.py`, new task config module
- Output: task selectable by name and mode

5. Build image wiring
- Depends on: (3), (4)
- Inputs: `docker/Dockerfile.base`, task Dockerfile patterns
- Output: notebook task image and runtime entrypoint compatibility

6. Verifier and L1 scoring integration
- Depends on: (3), (4), (5)
- Inputs: `tests/test.sh`, `compute_reward.py`, gate/test command settings
- Output: deterministic scoring path works in environment

7. End-to-end execution validation
- Depends on: (6)
- Inputs: baseline runner and local container
- Output: successful reset/step cycle plus at least one subtask score path

8. Hardening and docs closure
- Depends on: (7)
- Inputs: run logs and observed failures
- Output: resolved blockers, updated docs, DoD signoff

## Critical Path

(1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8)

## Parallelizable Work

Can run in parallel after (2):
- drafting task instruction adaptation notes
- preparing docs and progress tracker updates
- collecting optional oracle/baseline scripts

Cannot run in parallel:
- verifier scoring integration before task assets exist
- end-to-end validation before image/config wiring

## Risk-Driven Dependencies

- If upstream verifier assumptions are not mirrored, local L1 parsing will be invalid.
- If task config is wired before command contract is stable, score submissions may fail silently.
- If image does not include required compression libs/tooling, agent progress will stall early.
