# Implementation Plan (Issue #4)

Updated 2026-04-24 after full upstream inspection. Supersedes prior draft.

## Phase 0: Pre-Execution Planning (Done)

1. Gathered task requirements from upstream and task website.
2. Pinned upstream reference commit `55d1033`.
3. Inspected hidden bundle, upstream Dockerfile, verifier scripts, and local core env.
4. Produced design spec at `docs/superpowers/specs/2026-04-24-notebook-compression-design.md`.

## Phase 1: Task Asset Vendoring (Done)

1. Copied upstream `tasks/notebook-compression/` verbatim into local `tasks/notebook-compression/`.
2. No pruning at file-system level; runtime prunes (see Phase 3/4 — image build only consumes a subset).

Milestone:
- local `tasks/notebook-compression/` is a faithful snapshot of upstream at the pinned commit.

## Phase 2: Visible-Corpus Split Utility

1. Add `scripts/split_visible_corpus.py`:
- unpacks `tasks/notebook-compression/tests/hidden_test_set_bundle.zip`
- deterministically splits ~75% of notebooks into a `visible/` directory
- writes `manifest.json` alongside
- leaves the original zip untouched for verifier use
2. Invoked at image-build time, not at container runtime.

Milestone:
- running the script locally produces a well-formed `/mnt/notebook-data/visible/` tree and manifest.

## Phase 3: Task Config And Rubric Generalization

1. Add `l1_timeout_s: float = 300.0` field to `TaskConfig`.
2. Add `"reward_json"` score mode to `TestOutputRubric`:
- `reward_json_path` field
- after running the test command, read the JSON file
- hard-fail (`status != "ok"`) → 0.0
- otherwise normalize `geom_mean_ratio` via anchored clamp
3. Thread `l1_timeout_s` through `FrontierSweEnvironment.__init__`.
4. Enrich `submit_subtask_payload` feedback when score mode is `reward_json`: include `geom_mean_ratio`, `compression_score`, `status`, `reason`, and stage timings in the response and the L2 summary string.
5. Add `frontier_swe_env/tasks/notebook_compression.py` with `notebook_training_config()` and `notebook_demo_config()` factories.
6. Register `notebook` and `notebook-compression` aliases in `frontier_swe_env/tasks/__init__.py`.

Milestone:
- `get_task_config("notebook", "training")` resolves; rubric unit-testable against a canned `reward.json`.

## Phase 4: Docker Image Wiring

1. Write `docker/Dockerfile.notebook` (FROM `openenv-base:latest`):
- install system compression libs (zstd, brotli, lz4, zlib1g-dev, liblzma-dev, libbz2-dev)
- install Python scientific + compression deps via uv (numpy, pandas, scipy, pyarrow, joblib, tqdm, nbformat, jsonschema, datasketch, zstandard, brotli, lz4)
- copy workspace stub (upstream `environment/workspace/run`) to `/app/run` with exec bit
- copy verifier scripts (`tests/compute_reward.py`, `tests/scoring_core.py`, `tests/test.sh`, `tests/hidden_test_set_bundle.zip`) to `/opt/verifier/`
- run `scripts/split_visible_corpus.py` to populate `/mnt/notebook-data/visible/` + `manifest.json`
- copy core env code + gate script
- git-init `/app` for L2 diff tracking
- keep our `openenv_entrypoint.sh` as ENTRYPOINT (not upstream's)
2. Add `scripts/notebook_gate_checks.sh`:
- Gate 1: `/app/run` exists and is executable
- Gate 2: `/mnt/notebook-data/visible/` exists and is non-empty
- Gate 3: `python3 -c "import zstandard"` succeeds
- emits `GATE_SCORE=N/3`
3. Build chain: `openenv-base:latest` must exist before `frontier-swe-notebook:latest`.

Milestone:
- `podman build -f docker/Dockerfile.base -t openenv-base:latest .` succeeds.
- `podman build -f docker/Dockerfile.notebook -t frontier-swe-notebook:latest .` succeeds.
- `podman run -p 8000:8000 frontier-swe-notebook:latest` serves OpenEnv on port 8000.

## Phase 5: End-To-End Validation

1. Smoke-run the verifier against upstream `run` stub → expect `status=fail, reason="starter scaffold only"`.
2. Replace `/app/run` with a trivial zstd round-trip script → expect `status=ok, geom_mean_ratio > 0.2`.
3. Run a full episode with pi agent (via `scripts/run_baseline.py` if task-agnostic, otherwise add a notebook variant):
- plan submission works
- at least one `submit_subtask` returns `score > 0` with enriched feedback
- `advance` transitions state correctly
- episode terminates and produces a reward.
4. Capture logs under `artifacts/issue-4/` and reference from `progress-tracker.md`.

Milestone:
- full reset/step/submit/advance loop validated end-to-end in podman.

## Phase 6: Docs And Closeout

1. Update `progress-tracker.md` with evidence per item.
2. Update `decision-log.md` with post-implementation deviations (if any).
3. Verify every item in `dod.md` is true with a referenced artifact.
4. Short summary of deltas vs upstream for PR description.

Milestone:
- DoD signed off; ready for PR review.

## Runtime-Exclusion List (See Decision Log D-010)

These upstream files are present in `tasks/notebook-compression/` for reference and reproducibility but are NOT consumed by the image build or the episode runtime:

- `scripts/build_scoring_anchors.py`
- `scripts/build_splits.py`
- `scripts/canonicalize.py`
- `scripts/check_corpus_acceptance.py`
- `scripts/check_source_manifest.py`
- `scripts/collect_pilot.py`
- `scripts/generic_baseline_run.py`
- `scripts/notebook_aware_baseline_core.py`
- `scripts/notebook_aware_baseline_png.py`
- `scripts/notebook_aware_baseline_run.py`
- `scripts/profile_corpus.py`
- `scripts/rebuild_test_bundle.py`
- `scripts/run_baseline_suite.py`
- `scripts/select_diverse_subset.py`
- `scripts/stage_agent_volume.py`
- `sources/license_manifest.json`
- `sources/public_sources.json`
- `oracle.yaml`
- `job.yaml`
- `environment/Dockerfile` (reference only; our `docker/Dockerfile.notebook` is authoritative)
- `environment/workspace/entrypoint.sh` (conflicts with our openenv entrypoint)
- `environment/workspace/timer.sh` (our base image has its own)

If any of these turn out to be required later (e.g., a baseline we want to reproduce), pull them into the image deliberately and document the reason.
