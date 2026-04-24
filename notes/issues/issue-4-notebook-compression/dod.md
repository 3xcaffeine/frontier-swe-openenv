# Definition Of Done (DoD): Issue #4 Notebook Compression

Signed off 2026-04-25. Evidence inline per item.

## A. Task Assets And Contracts

- [x] `tasks/notebook-compression/` exists with required instruction, environment, and tests. — vendored verbatim at commit `55d1033` (D-001, D-010).
- [x] `/app/run` contract supports `fit`, `compress`, and `decompress` stages exactly. — upstream stub baked at `/app/run`; agent replaces it per instruction.md.
- [x] Hidden holdout verifier scripts are present and executable in task image. — `/opt/verifier/{test.sh,compute_reward.py,scoring_core.py,hidden_test_set_bundle.zip}` verified in-container smoke.

## B. Configuration And Registry

- [x] Notebook task is registered in `frontier_swe_env/tasks/` and selectable by name. — `notebook` and `notebook-compression` aliases; env-var selection wired in D-011.
- [x] Training and demo configs are defined with valid values. — `frontier_swe_env/tasks/notebook_compression.py` (training: 3 subtasks × 2 attempts, 3600s episode, 1800s L1; demo: 5 × 3, 7200s, 3000s).
- [x] L1 scoring command and output parsing are configured for this task. — `l1_score_mode="reward_json"` reads `/logs/verifier/reward.json`, normalizes `geom_mean_ratio` to [0,1] (D-006 revised).

## C. Image And Runtime

- [x] Notebook task Docker image builds successfully. — `openenv-base:latest` 1.45 GB, `frontier-swe-notebook:latest` 1.96 GB.
- [x] Container starts and serves OpenEnv app. — `/health` responds in ~3 s; uvicorn + FastMCP mounted.
- [x] Required runtime dependencies for compression workload are installed. — zstd/brotli/lz4 system libs + numpy/pandas/scipy/pyarrow/nbformat/zstandard/brotli/lz4 Python libs verified via gate 3/3.

## D. Environment Behavior

- [x] `reset()` initializes notebook task episode without core changes. — `time_remaining_s=3600` confirms notebook config active; phase=PLANNING.
- [x] Agent can call `submit_plan`, `submit_subtask`, `get_status`, `advance`. — container log shows pi invoked all four successfully.
- [x] At least one `submit_subtask` returns a non-error scoring payload. — S1 blended=0.85, S2 blended=0.83; both with full `l1_extras` enrichment (status, geom_mean_ratio, stage timings).

## E. Verification And Evidence

- [x] Visible/local verifier path can run to completion. — trivial zstd wrapper → `status=ok, geom_mean_ratio=0.326335, round-trip OK`.
- [x] Reward output files are generated (`reward.json`, `reward.txt`) in expected location. — `/logs/verifier/reward.json` confirmed, parsed by rubric.
- [x] One baseline execution artifact/log is saved and referenced in docs. — `artifacts/issue-4/notebook-baseline-container.log` (170 lines, gitignored per project policy).

## F. Documentation And Tracking

- [x] This issue folder is updated with decisions and learnings. — `decision-log.md` D-001 through D-013; `research-notes.md`, `dependency-map.md` unchanged.
- [x] `progress-tracker.md` reflects final status for each planned item. — P0–P12 DONE with evidence; P13 is this DoD.
- [x] Any deviations from upstream task contract are documented with rationale. — D-009 (visible-corpus synthesis), D-010 (vendor-verbatim-prune-at-runtime), D-007 (scaled-down timeouts).

## Signoff

- [x] Technical implementation complete.
- [x] DoD validated with evidence links/paths.
- [x] Ready for issue close.
