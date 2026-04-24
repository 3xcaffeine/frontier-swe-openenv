# Decision Log: Issue #4 Notebook Compression

## D-001: Pin External Reference

- Date: 2026-04-24
- Decision: use pinned upstream snapshot for planning reference.
- Reference: https://github.com/Proximal-Labs/frontier-swe/tree/55d103355bf0bfffb6b47781733e817f9dc65bb3/tasks/notebook-compression
- Why: prevents drift while implementation is in progress.

## D-002: Reuse Task Pattern, Not Task Semantics

- Date: 2026-04-24
- Decision: reuse postgres task implementation structure (layout, wiring, verifier pattern), but do not reuse postgres-specific assumptions.
- Why: architecture is task-agnostic by design after PR #10.

## D-003 (revised): Core Env May Evolve For Task-Agnostic Generalizations

- Date: 2026-04-24
- Decision: supersede original "avoid core changes" stance. Core env may change when the change is a task-agnostic generalization (not a task-specific shim). For issue #4 this covers: adding `TaskConfig.l1_timeout_s`, adding `TestOutputRubric` score_mode `reward_json`, and enriching the `submit_subtask_payload` feedback.
- Why: rigid "no core changes" would force us to regex-parse a structured JSON result via stdout, throwing away hard-fail distinction and per-notebook metadata that the LLM judge and training signal benefit from.

## D-004: Dependency-First Execution

- Date: 2026-04-24
- Decision: execute in strict dependency order from `dependency-map.md`.
- Why: avoids premature wiring and repeated rework.

## D-005: Vendor Hidden Bundle In Repo

- Date: 2026-04-24
- Decision: include upstream `tests/hidden_test_set_bundle.zip` (69 MB, 80 `.ipynb`) in local task scaffold.
- Why: ensures local verifier parity from day one; upstream intended the bundle to be available at verifier runtime.
- Caveat: bundle is effectively public via this repo, so the "hidden holdout" is not genuinely hidden from an agent that reads the repo. Acceptable for training/demo; flag to issue #9 (reward hacking).

## D-006 (revised): L1 Scoring Via Structured reward.json, Not Stdout Regex

- Date: 2026-04-24
- Decision: add a `"reward_json"` score mode to `TestOutputRubric` that reads `/logs/verifier/reward.json` after the verifier runs. Hard-fail (`status != "ok"`) returns 0.0. Otherwise `geom_mean_ratio` is normalized to [0,1] via anchored clamp (`R_max=1.0 → 0`, `R_min=0.15 → 1.0`).
- Why: upstream `compute_reward.py` already produces a rich structured result (status, hard-fail reason, per-notebook metadata, stage timings). Stdout regex throws that richness away, hurting both the LLM judge summary and in-context feedback the agent needs to iterate.
- Supersedes: original D-006 ("Total: N/M passed" stdout ratio).

## D-007 (revised): Scaled-Down Timeouts For Small-Scale Training

- Date: 2026-04-24
- Decision: use episode_timeout_s=3600, per_turn_timeout_s=600, l1_timeout_s=1800; verifier stage limits NOTEBOOK_FIT/COMPRESS=600, NOTEBOOK_DECOMPRESS=300 (half of upstream).
- Why: upstream sizing (8h episode, 14400s verifier) targets a frontier-capable agent. We train a 36B model across many episode iterations and cannot afford upstream latencies. Scoring signal is still informative because the verifier timing subscores reflect stage duration directly.

## D-008: Task Selection Via Environment Variables

- Date: 2026-04-24
- Decision: support `FSWE_TASK_NAME` and `FSWE_TASK_MODE` in environment initialization.
- Why: allows task-specific images to select configs without changing app wiring.

## D-009: Visible Corpus Synthesized From Hidden Bundle At Image-Build Time

- Date: 2026-04-24
- Decision: there is no standalone visible-corpus dataset in the upstream repo — upstream's `$DATA_ROOT/visible/` is supplied by Harbor at job-launch time via bind mount. For our standalone image we split ~75% of the hidden bundle deterministically into `/mnt/notebook-data/visible/` at image build time, and leave the full 80-file bundle in place for verifier scoring.
- Why: the agent needs a visible corpus to `fit` against; no other source is available without pulling from external datasets (blocked by `allow_internet=false`).
- Caveat: visible ⊂ hidden, so the agent can in principle memorize the bundle and get an artificially low `geom_mean_ratio`. Mitigation: verifier enforces byte-exact round-trip and one-to-one file attribution, so memorization alone does not produce a valid submission — the agent must still implement a real lossless codec. Log to issue #9.

## D-010: Vendor Upstream Folder Verbatim; Prune At Runtime Only

- Date: 2026-04-24
- Decision: copy the entire upstream `tasks/notebook-compression/` folder as-is (scripts, sources, oracle.yaml, job.yaml, etc.) into local `tasks/notebook-compression/`. The Dockerfile / gate script / rubric consume only a subset — see runtime-exclusion list in `implementation-plan.md`.
- Why: faithful snapshot aids future onboarding and cross-reference; no ambiguity about what upstream actually shipped.
- Cost: ~70 MB in git history (dominated by the hidden bundle, which we would vendor anyway).

## D-011: Wire `FSWE_TASK_NAME` / `FSWE_TASK_MODE` In Core Env `__init__`

- Date: 2026-04-25
- Decision: read `FSWE_TASK_NAME` and `FSWE_TASK_MODE` from environment in `FrontierSweEnvironment.__init__`, falling back to the explicit `task_name`/`mode` constructor args. Per-image `ENV FSWE_TASK_NAME=notebook` now actually selects the notebook config; previously the env defaulted to PG regardless.
- Why: the Dockerfile.notebook set `ENV FSWE_TASK_NAME=notebook` per D-008's intent, but the env class never consumed the variable. Caught during OpenEnv server smoke (`time_remaining_s=900` instead of 3600 revealed PG config was active). Three-line fix in `__init__`.

## D-012: Defer Full pi Episode

- Date: 2026-04-25
- Decision: implementation stops at container + OpenEnv smoke + MCP startup. Full pi-driven episode (plan/submit/advance with real LLM judge) requires real `FSWE_AGENT_*` and `FSWE_GRADER_*` credentials and is deferred to a separate validation step.
- Why: the work in scope for issue #4 is onboarding the task; episode validation depends on external credentials and compute that aren't part of this branch. Dummy-credential smoke already confirms the path is wired correctly (reset→PLANNING, pi process starts, MCP config written, notebook config in effect).

## D-013: Reward Anchors Validated By Smoke

- Date: 2026-04-25
- Decision: keep initial `R_max=1.0, R_min=0.15` anchors. Trivial zstd-19 wrapper produced `geom_mean_ratio=0.326335` → normalized `(1.0-0.326)/(1.0-0.15)=0.793`, which is reasonable headroom for S2/S3 codec improvements. Tune after observing real agent runs in training.

## Open Decisions

- Whether `scripts/run_baseline.py` is task-agnostic enough for notebook-compression, or needs a second variant. Resolve when P12 runs.
