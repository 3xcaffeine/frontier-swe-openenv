# Notebook Compression Task — Design Spec

- Issue: #4
- Branch: `feat/issue-4-notebook-compression`
- Upstream pin: `Proximal-Labs/frontier-swe@55d1033` (tasks/notebook-compression)
- Date: 2026-04-24
- Status: design approved; ready for implementation plan

## Summary

Onboard `notebook-compression` as a second task on the task-agnostic OpenEnv core (PR #10). The task requires the agent to build a lossless codec for `.ipynb` files exposed as an `/app/run {fit,compress,decompress}` executable. Scoring is deterministic via upstream's `compute_reward.py` (geometric-mean per-notebook ratio, hard-fail on round-trip mismatch). Integration is predominantly task assets + task config + image wiring, plus two small task-agnostic generalizations to the core (see Core Changes).

## Goals

1. Register `notebook` / `notebook-compression` in the task registry.
2. Produce a podman-buildable image `frontier-swe-notebook:latest` that serves the OpenEnv app and the MCP tool surface.
3. Run an end-to-end episode via pi with plan → submit → advance, producing a non-zero training signal.
4. Keep the scoring signal structured and actionable (geom_mean_ratio, hard-fail reason, stage timings) so that both the LLM judge and the RL training loop can consume it meaningfully.

## Non-Goals

- Full parity with upstream Harbor orchestration (bind-mounted datasets, multi-worker parallel verifiers, Modal deployment).
- Matching upstream timeouts or budgets (we scale down for small-scale 36B training).
- HF Spaces deployment (tracked separately in issue #6).
- Reward-hacking hardening (tracked in issue #9).

## Architecture

### File layout after implementation

```
tasks/notebook-compression/                 (vendored upstream, verbatim)
├── instruction.md
├── task.toml
├── job.yaml                                (not consumed at runtime)
├── oracle.yaml                             (not consumed at runtime)
├── environment/
│   ├── Dockerfile                          (reference only)
│   └── workspace/
│       ├── entrypoint.sh                   (not used; conflicts with our entrypoint)
│       ├── timer.sh                        (not used; base image has its own)
│       └── run                             (COPIED to /app/run at image build)
├── scripts/                                (not consumed at runtime; see D-010)
├── sources/                                (not consumed at runtime)
└── tests/
    ├── test.sh                             (COPIED to /opt/verifier/)
    ├── compute_reward.py                   (COPIED to /opt/verifier/)
    ├── scoring_core.py                     (COPIED to /opt/verifier/)
    ├── generate_test_bundle.py             (not consumed at runtime)
    └── hidden_test_set_bundle.zip          (COPIED to /opt/verifier/; 69 MB, 80 .ipynb)

frontier_swe_env/
├── task_config.py                          (MODIFIED: add l1_timeout_s)
├── rubrics/l1_tests.py                     (MODIFIED: add "reward_json" score_mode)
├── server/frontier_swe_env_environment.py  (MODIFIED: thread l1_timeout_s; enrich feedback)
└── tasks/
    ├── __init__.py                         (MODIFIED: register notebook task)
    └── notebook_compression.py             (NEW)

docker/Dockerfile.notebook                  (NEW)
scripts/
├── notebook_gate_checks.sh                 (NEW)
└── split_visible_corpus.py                 (NEW; build-time only)
```

### Runtime data flow

```
  podman run frontier-swe-notebook:latest
            │
            ▼
  openenv_entrypoint.sh
    ├─ generates /root/.pi/agent/models.json from FSWE_AGENT_* env vars
    ├─ starts /app/timer.sh (countdown)
    └─ exec uvicorn frontier_swe_env.server.app:app
            │
            ▼
  client calls POST /reset
    └─ FrontierSweEnvironment.reset()
         ├─ resets /app git state (stub /app/run restored)
         ├─ starts pi subprocess with MCP tools pointing at /tools/mcp
         └─ returns Observation(phase=PLANNING)

  client calls POST /step ("go")
    └─ pi runs its agent loop, calls MCP tools
         ├─ submit_plan  → phase=EXECUTING
         ├─ [agent edits /app/run]
         ├─ submit_subtask(S1)
         │    └─ gate_rubric.forward  → GATE_SCORE=N/3
         │        (if N/3 >= gate_threshold)
         │    └─ test_rubric.forward  → runs bash /opt/verifier/test.sh
         │                              (timeout l1_timeout_s = 1800s)
         │                              → writes /logs/verifier/reward.json
         │                              → reads reward.json, maps to [0,1]
         │    └─ l2_rubric.grade      → LLM judge over git diff + l1_summary
         │    └─ returns {score, l1_score, l2_score, feedback, geom_mean_ratio, ...}
         ├─ [agent iterates on /app/run]
         ├─ submit_subtask(S1)  (2nd attempt)
         ├─ advance
         ├─ submit_subtask(S2) ...
         └─ advance → phase=DONE → episode_rubric.compute → reward in [0,1]
```

### MCP contract

Unchanged. `submit_plan`, `submit_subtask`, `get_status`, `advance` on `/tools/mcp` — task-agnostic at this layer.

## Core Changes

Two additions to the core env, both task-agnostic generalizations (see decision-log D-003 revised). A third task benefiting from structured JSON verifiers (e.g. any Harbor-style task) inherits these for free.

### 1. `TaskConfig.l1_timeout_s`

```python
# frontier_swe_env/task_config.py
class TaskConfig(BaseModel):
    ...
    l1_timeout_s: float = 300.0          # NEW; threaded into TestOutputRubric
    reward_json_path: str = "/logs/verifier/reward.json"  # NEW; used when l1_score_mode == "reward_json"
```

PG config keeps the defaults. Notebook config sets `l1_timeout_s=1800` and keeps `reward_json_path` at default.

### 2. `TestOutputRubric` gains `score_mode="reward_json"`

```python
# frontier_swe_env/rubrics/l1_tests.py (conceptual)
class TestOutputRubric(Rubric):
    def __init__(self,
                 test_command: str,
                 output_pattern: str = r"Total:\s*(\d+)/(\d+)\s*passed",
                 score_mode: str = "ratio",
                 reward_json_path: str = "/logs/verifier/reward.json",
                 timeout_s: int = 300,
                 ...): ...

    def forward(self, action, observation) -> float:
        # run the test command (subprocess)
        # if score_mode == "reward_json":
        #     read reward_json_path
        #     if status != "ok": return 0.0
        #     return _normalize_ratio(geom_mean_ratio)
        # else: existing regex path

    @staticmethod
    def _normalize_ratio(r, r_max=1.0, r_min=0.15):
        # anchor: r=1.0 (no compression) → 0.0; r=0.15 (strong) → 1.0
        return max(0.0, min(1.0, (r_max - r) / (r_max - r_min)))
```

The rubric also exposes the parsed `reward.json` payload to the environment via a last-result cache attribute, so `submit_subtask_payload` can enrich feedback without re-parsing.

### 3. Enriched `submit_subtask_payload` feedback (`reward_json` mode)

When `task_config.l1_score_mode == "reward_json"`, after `test_rubric.forward` runs:

```python
reward = json.loads(Path(self.task_config.reward_json_path).read_text())
l1_extras = {
    "geom_mean_ratio": reward.get("geom_mean_ratio"),
    "compression_score": reward.get("compression_score"),
    "status": reward.get("status"),
    "reason": reward.get("reason"),
    "stage_timings": {
        "fit_elapsed_sec": reward.get("fit_elapsed_sec"),
        "compress_elapsed_sec": reward.get("compress_elapsed_sec"),
        "decompress_elapsed_sec": reward.get("decompress_elapsed_sec"),
    },
}
l1_summary = (f"Status: {reward['status']} | "
              f"geom_mean_ratio: {reward.get('geom_mean_ratio')} | "
              f"reason: {reward.get('reason')}")
```

- `l1_extras` is merged into the dict returned to the agent.
- `l1_summary` replaces the PG-style "Gate: x, Compat tests: y" string in the L2 prompt context.

## Task Config

```python
# frontier_swe_env/tasks/notebook_compression.py (shape)

def notebook_training_config() -> TaskConfig:
    return TaskConfig(
        task_name="notebook-compression",
        docker_image="frontier-swe-notebook:latest",
        instruction=NOTEBOOK_TRAINING_INSTRUCTION,
        workspace_dir="/app",
        build_command=":",                                  # no build; agent edits /app/run
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="bash /opt/verifier/test.sh",
        visible_test_total=80,                              # informational
        # l1_output_pattern left at default; unused in reward_json mode
        l1_score_mode="reward_json",
        l1_timeout_s=1800,
        gate_threshold=0.67,                                # 2 of 3 gates
        max_subtasks=3,
        max_attempts_per_subtask=2,
        episode_timeout_s=3600,
        per_turn_timeout_s=600,
        task_description="Build a lossless codec for Jupyter notebooks with a fit/compress/decompress contract.",
        task_domain="systems / compression",
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh which executes the upstream verifier. "
            "reward.json is parsed: hard-fail (non-ok status) scores 0.0; otherwise "
            "geom_mean_ratio (lower is better) is normalized to [0,1] via "
            "(1.0 - r) / (1.0 - 0.15)."
        ),
        cpus=16,
        memory_mb=32768,
    )

def notebook_demo_config() -> TaskConfig:
    # same shape, max_subtasks=5, max_attempts=3, episode_timeout_s=7200, l1_timeout_s=3000
```

Registered with:

```python
# frontier_swe_env/tasks/__init__.py (addition)
from .notebook_compression import notebook_demo_config, notebook_training_config
register_task("notebook", notebook_training_config, notebook_demo_config)
register_task("notebook-compression", notebook_training_config, notebook_demo_config)
```

## Workspace And Verifier

- `/app/run` starts as the upstream stub (raises on any command) → first submit hard-fails with `starter scaffold only`.
- `/app` is git-init'd at image build time for L2 diff tracking.
- `/opt/verifier/test.sh` is the upstream script with one adjustment: `VERIFIER_DIR` defaults to `/logs/verifier` (already the case in upstream — no diff needed), and `SCRIPT_DIR` looks for the hidden bundle in its own directory (works because we copy the zip alongside it).
- `/mnt/notebook-data/visible/` is populated at image-build time by `scripts/split_visible_corpus.py` (deterministic 75/25 split of the hidden bundle; seeded).

### Gate script (`scripts/notebook_gate_checks.sh`)

```bash
#!/usr/bin/env bash
set -uo pipefail
GATE=0; TOTAL=3
[ -x /app/run ] && { GATE=$((GATE+1)); echo "GATE 1 PASS: /app/run exec"; } || echo "GATE 1 FAIL"
[ -d /mnt/notebook-data/visible ] && [ -n "$(ls -A /mnt/notebook-data/visible 2>/dev/null)" ] \
    && { GATE=$((GATE+1)); echo "GATE 2 PASS: visible corpus present"; } \
    || echo "GATE 2 FAIL: visible corpus missing"
python3 -c 'import zstandard, nbformat' 2>/dev/null \
    && { GATE=$((GATE+1)); echo "GATE 3 PASS: py deps importable"; } \
    || echo "GATE 3 FAIL: py deps missing"
echo "GATE_SCORE=${GATE}/${TOTAL}"
```

Cheap, always-run, catches obvious submission shape errors before spending a 30-min verifier run.

## Dockerfile.notebook Shape

```dockerfile
FROM openenv-base:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV DATA_ROOT=/mnt/notebook-data
ENV TASK_BUDGET_SECS=3600
ENV FSWE_TASK_NAME=notebook
ENV FSWE_TASK_MODE=training

RUN apt-get update && apt-get install -y --no-install-recommends \
      zstd brotli lz4 zlib1g-dev liblzma-dev libbz2-dev \
      unzip jq \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system \
      numpy pandas scipy pyarrow joblib tqdm \
      nbformat jsonschema datasketch \
      zstandard brotli lz4

# Workspace stub
COPY tasks/notebook-compression/environment/workspace/run /app/run
RUN chmod +x /app/run

# Verifier
RUN mkdir -p /opt/verifier /logs/verifier /mnt/notebook-data
COPY tasks/notebook-compression/tests/compute_reward.py /opt/verifier/
COPY tasks/notebook-compression/tests/scoring_core.py /opt/verifier/
COPY tasks/notebook-compression/tests/test.sh /opt/verifier/
COPY tasks/notebook-compression/tests/hidden_test_set_bundle.zip /opt/verifier/
RUN chmod +x /opt/verifier/test.sh

# Visible corpus split (build-time)
COPY scripts/split_visible_corpus.py /tmp/split_visible_corpus.py
RUN python3 /tmp/split_visible_corpus.py \
      --bundle /opt/verifier/hidden_test_set_bundle.zip \
      --out /mnt/notebook-data/visible \
      --manifest /mnt/notebook-data/manifest.json \
      --ratio 0.75 --seed 17

# Gate checks
COPY scripts/notebook_gate_checks.sh /app/gate_checks.sh
RUN chmod +x /app/gate_checks.sh

# OpenEnv core
COPY frontier_swe_env/ /opt/openenv/frontier_swe_env/
COPY pyproject.toml /opt/openenv/pyproject.toml
ENV PYTHONPATH=/opt/openenv

# Git baseline for L2 diff
RUN cd /app \
    && git config --global user.email "agent@frontier-swe-openenv" \
    && git config --global user.name "agent" \
    && git init && git add -A && git commit -m "initial stub"

# Re-copy entrypoint (matches Dockerfile.pg pattern for consistency)
COPY docker/openenv_entrypoint.sh /app/openenv_entrypoint.sh
RUN chmod +x /app/openenv_entrypoint.sh

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

## Subtask Shape (In Instruction)

The instruction guides pi toward 3 incremental codec iterations. Each `submit_subtask` re-runs the full verifier (~5–15 min per run), so we cap attempts tight.

- **S1** — round-trip baseline: wrap `zstd -19` (with `.zst` suffix per input); `fit` writes a marker file; `decompress` reverses the wrapper. Must pass hard-fail. Expected `geom_mean_ratio` ≈ 0.35-0.45.
- **S2** — trained dictionary: `fit` trains a zstd dictionary from the visible corpus and writes it to `artifact_dir`; compress/decompress use the dict. Expected ~0.25-0.30.
- **S3** — notebook-aware preprocessing: canonicalize JSON (strip outputs/metadata into a sidecar bundled in `artifact_dir`; re-inject on decompress); then dict-compress. Expected ~0.18-0.25.

Instruction must be explicit that only byte-exact round-trip is acceptable — lossy simplification fails the verifier.

## Reward Normalization Anchors

- `R_max = 1.0` (no compression → score 0.0)
- `R_min = 0.15` (strong baseline → score 1.0)
- Clamp to [0, 1]
- Values are initial guesses; revisit after first real episodes (tracked in decision-log open decisions).

## Testing Strategy

1. **Unit** — `TestOutputRubric` with `score_mode="reward_json"`:
   - Synthetic `reward.json` with `status=ok, geom_mean_ratio=0.25` → score ≈ 0.88
   - `status=fail` → score = 0.0
   - Missing file → score = 0.0
   - Malformed JSON → score = 0.0
2. **Integration (in container)**:
   - Stub `/app/run` → verifier emits `status=fail` → rubric → 0.0
   - Trivial zstd wrapper `/app/run` → verifier emits `status=ok, geom_mean_ratio ∈ (0.2, 0.5)` → rubric → positive score, round-trip OK
3. **End-to-end**:
   - Podman-run the image, hit `/reset` and `/step`, let pi run an episode, observe at least one `submit_subtask` returning a non-zero `score` with `geom_mean_ratio` surfaced in `feedback`.

## Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Visible ⊂ hidden bundle — agent could memorize | Byte-exact round-trip + one-to-one attribution still force a real codec; logged to issue #9 |
| R2 | Image size ~2.5 GB | Acceptable for podman; flag for HF Spaces (issue #6) |
| R3 | 6 verifier runs per episode × 10 min = 60 min worst case | `episode_timeout_s=3600`, low max_attempts |
| R4 | `R_min/R_max` anchors are guesses — score may saturate or flat-line | Observe empirical distribution after first runs; tune anchors |
| R5 | Upstream `test.sh` uses source-code scan that rejects any file referencing `compute_reward`, `hidden_test_set_bundle`, etc. | We don't ship such references in `/app`; keep task assets clean of verifier strings |
| R6 | `fit` stage only reads `$DATA_ROOT/visible/`; read-only bake is sufficient | Verified in `compute_reward.py::find_fit_input_dir` |

## Open Questions

- `scripts/run_baseline.py` task-agnostic enough? Resolve during P11 (validation); if not, add a second variant.
- Anchors `R_min=0.15`, `R_max=1.0` — tune after first real episodes.

## Implementation Plan

See `notes/issues/issue-4-notebook-compression/implementation-plan.md` (Phases 2–6) for the ordered work plan and `progress-tracker.md` for execution status.
