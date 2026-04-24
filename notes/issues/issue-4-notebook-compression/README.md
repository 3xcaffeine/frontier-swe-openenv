# Issue #4 Working Docs: Notebook Compression

This folder is the execution workspace for issue #4:
- Issue: implement notebook-compression from FrontierSWE
- Goal: onboard a second task environment (after postgres-sqlite) using the same task-agnostic OpenEnv core

## Source Of Truth Links

- Task page: https://www.frontierswe.com/notebook-compression
- Upstream task folder (main): https://github.com/Proximal-Labs/frontier-swe/tree/main/tasks/notebook-compression
- Upstream task folder (pinned snapshot): https://github.com/Proximal-Labs/frontier-swe/tree/55d103355bf0bfffb6b47781733e817f9dc65bb3/tasks/notebook-compression
- Pin captured on: 2026-04-24

## Local Reference Baseline (Already Implemented)

Use postgres-sqlite as the implementation template:
- `tasks/postgres-sqlite-wire-adapter/`
- `frontier_swe_env/tasks/pg.py`
- `docker/Dockerfile.pg`
- `scripts/pg_gate_checks.sh`
- `frontier_swe_env/server/frontier_swe_env_environment.py`

Core rule:
- Keep `frontier_swe_env/server/*`, core rubrics, and client transport task-agnostic.
- Add notebook-compression mostly as task assets + task config + image wiring.

## Working Files In This Folder

- `research-notes.md`: extracted requirements and constraints from upstream task.
- `dependency-map.md`: step-by-step dependency graph before implementation.
- `implementation-plan.md`: ordered execution plan and milestones.
- `dod.md`: definition of done with verifiable checks.
- `progress-tracker.md`: itemized status tracker for execution.
- `decision-log.md`: key decisions and rationale as work progresses.

## Current Status

- Research captured.
- Upstream links pinned.
- Pre-execution dependency mapping drafted.
- Ready to begin implementation once plan approval is confirmed.
