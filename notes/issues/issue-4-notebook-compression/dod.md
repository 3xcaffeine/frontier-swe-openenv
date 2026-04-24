# Definition Of Done (DoD): Issue #4 Notebook Compression

All items must be true to close this issue.

## A. Task Assets And Contracts

- [ ] `tasks/notebook-compression/` exists with required instruction, environment, and tests.
- [ ] `/app/run` contract supports `fit`, `compress`, and `decompress` stages exactly.
- [ ] Hidden holdout verifier scripts are present and executable in task image.

## B. Configuration And Registry

- [ ] Notebook task is registered in `frontier_swe_env/tasks/` and selectable by name.
- [ ] Training and demo configs are defined with valid values.
- [ ] L1 scoring command and output parsing are configured for this task.

## C. Image And Runtime

- [ ] Notebook task Docker image builds successfully.
- [ ] Container starts and serves OpenEnv app.
- [ ] Required runtime dependencies for compression workload are installed.

## D. Environment Behavior

- [ ] `reset()` initializes notebook task episode without core changes.
- [ ] Agent can call `submit_plan`, `submit_subtask`, `get_status`, `advance`.
- [ ] At least one `submit_subtask` returns a non-error scoring payload.

## E. Verification And Evidence

- [ ] Visible/local verifier path can run to completion.
- [ ] Reward output files are generated (`reward.json`, `reward.txt`) in expected location.
- [ ] One baseline execution artifact/log is saved and referenced in docs.

## F. Documentation And Tracking

- [ ] This issue folder is updated with decisions and learnings.
- [ ] `progress-tracker.md` reflects final status for each planned item.
- [ ] Any deviations from upstream task contract are documented with rationale.

## Signoff

- [ ] Technical implementation complete.
- [ ] DoD validated with evidence links/paths.
- [ ] Ready for issue close.
