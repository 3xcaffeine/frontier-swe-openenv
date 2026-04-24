# Implementation Plan (Issue #4)

## Phase 0: Pre-Execution Planning (Done)

1. Gather task requirements and constraints.
2. Pin upstream task reference commit.
3. Draft dependency map and DoD.

## Phase 1: Task Skeleton Import

1. Create `tasks/notebook-compression/` in this repo.
2. Bring in required task assets:
- instruction
- task metadata (`task.toml`, optional `job.yaml` references)
- environment workspace scaffold (`run`, `entrypoint.sh`, `timer.sh`)
- verifier scripts (`test.sh`, `compute_reward.py`, shared scoring helpers)
3. Keep file paths consistent with existing task conventions in this repo.

Milestone:
- task folder exists and can be inspected end-to-end locally.

## Phase 2: Environment And Config Wiring

1. Add notebook task config module under `frontier_swe_env/tasks/`.
2. Register task in task registry (`get_task_config` path).
3. Define scoring command, output parser pattern, thresholds, and timing values.
4. Ensure training/demo modes are defined.

Milestone:
- environment can resolve notebook task config without touching core state-machine logic.

## Phase 3: Docker Runtime Wiring

1. Add notebook-specific Dockerfile in `docker/`.
2. Ensure required compression/system dependencies are installed.
3. Wire openenv entrypoint/runtime behavior and timer.
4. Ensure hidden test bundle path and verifier artifacts are copied correctly.

Milestone:
- image builds and starts OpenEnv app for notebook task.

## Phase 4: Scoring And Flow Validation

1. Validate verifier script execution in container.
2. Validate L1 parsing path from test output.
3. Run a short baseline episode to verify:
- plan submission
- subtask submission
- score response payload
- phase transition behavior

Milestone:
- at least one end-to-end episode reaches scored subtask path.

## Phase 5: Hardening And Closeout

1. Fix observed blockers.
2. Update docs and progress tracker with evidence.
3. Verify against DoD checklist.
4. Prepare final summary of implementation deltas.

Milestone:
- DoD fully checked.

## Notes On Reuse From Postgres Task

Directly reuse patterns from postgres implementation for:
- task folder layout
- docker layering strategy
- verifier + compute_reward shape
- baseline and scoring integration flow

Do not reuse task-specific assumptions:
- SQL/gate semantics
- PostgreSQL protocol checks
- postgres-specific acceptance criteria
