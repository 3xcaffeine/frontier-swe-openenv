"""Notebook-compression task configuration.

Agent builds a lossless codec for Jupyter .ipynb files exposed as
    ./run fit      <visible_dir> <artifact_dir>
    ./run compress <artifact_dir> <input_dir> <compressed_dir>
    ./run decompress <artifact_dir> <compressed_dir> <recovered_dir>

L1 scoring reads a structured reward.json from the upstream verifier
(see tasks/notebook-compression/tests/compute_reward.py).
"""

from __future__ import annotations

from pathlib import Path

from ..task_config import TaskConfig


NOTEBOOK_TRAINING_INSTRUCTION = """
# Notebook Compression — Lossless Codec

Your workspace is `/app`. The entrypoint is `/app/run` (currently a stub that
fails). You must implement a lossless compressor for Jupyter `.ipynb` files.

## Contract

`/app/run` must support exactly these three subcommands:

```
./run fit        <visible_dir> <artifact_dir>
./run compress   <artifact_dir> <input_dir> <compressed_dir>
./run decompress <artifact_dir> <compressed_dir> <recovered_dir>
```

- `fit` reads the visible corpus at `$DATA_ROOT/visible/` and writes any
  artifacts (dictionary, model, code) to `<artifact_dir>`. The visible
  corpus is NOT available at compress/decompress time.
- `compress` reads each regular file in `<input_dir>` and writes one
  compressed output per input at the same relative path (suffixes allowed).
- `decompress` must recover the original bytes EXACTLY (byte-for-byte,
  same relative paths). Any round-trip mismatch is a hard fail.

## Scoring

L1 runs `bash /opt/verifier/test.sh` which executes the upstream verifier.
The verifier writes `/logs/verifier/reward.json`. The primary metric is
`geom_mean_ratio` (lower is better). Hard failures (`status != "ok"`)
score 0.0. Valid runs are normalized so that `r=1.0` → 0.0 and
`r=0.15` → 1.0.

## Useful commands

- Check timer: `cat /app/.timer/remaining_secs`
- Inspect visible corpus: `ls /mnt/notebook-data/visible/ | head`
- System tools available: `zstd`, `brotli`, `lz4`
- Python compression bindings: `zstandard`, `brotli`, `lz4`, `nbformat`

## Episode workflow

You MUST follow this workflow — your code is only scored when you use these tools.

**IMPORTANT: Each `submit_subtask` runs the full verifier (up to ~30 min).
You have 3 subtasks, 2 attempts each. Budget accordingly — don't waste a
submission on code you know doesn't round-trip.**

1. **Plan first.** Call `submit_plan` with 3 small incremental subtasks:
   ```
   submit_plan({"subtasks": [
     {"id": "S1", "description": "baseline zstd wrapper with byte-exact round-trip",
      "acceptance_criteria": "verifier status=ok, any geom_mean_ratio"},
     {"id": "S2", "description": "trained zstd dictionary from visible corpus",
      "acceptance_criteria": "geom_mean_ratio < S1 ratio"},
     {"id": "S3", "description": "notebook-aware preprocessing (canonicalize JSON)",
      "acceptance_criteria": "geom_mean_ratio < S2 ratio"}
   ]})
   ```

2. **Code the current subtask.** Edit `/app/run` directly. Test locally:
   ```
   mkdir -p /tmp/a /tmp/c /tmp/r
   /app/run fit /mnt/notebook-data/visible /tmp/a
   /app/run compress /tmp/a /mnt/notebook-data/visible /tmp/c
   /app/run decompress /tmp/a /tmp/c /tmp/r
   diff -r /mnt/notebook-data/visible /tmp/r && echo ROUND_TRIP_OK
   ```

3. **Submit for scoring.** Call `submit_subtask` — this invokes the real
   verifier on the HIDDEN set. The response includes `score`,
   `l1_extras.geom_mean_ratio`, `l1_extras.reason`, and `feedback`.

4. **Advance** when satisfied or out of attempts.

5. **Check progress:** `get_status`.

**Remember: byte-exact round-trip is a HARD requirement. Any mismatch scores 0.**
""".strip()


def _load_upstream_instruction() -> str:
    """Return the upstream instruction.md if present, else the training text."""
    upstream = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "notebook-compression"
        / "instruction.md"
    )
    if upstream.is_file():
        return upstream.read_text()
    return NOTEBOOK_TRAINING_INSTRUCTION


def notebook_training_config() -> TaskConfig:
    return TaskConfig(
        task_name="notebook-compression",
        docker_image="frontier-swe-notebook:latest",
        instruction=NOTEBOOK_TRAINING_INSTRUCTION,
        workspace_dir="/app",
        build_command=":",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="bash /opt/verifier/test.sh",
        visible_test_total=80,
        l1_score_mode="reward_json",
        l1_timeout_s=1800.0,
        reward_json_path="/logs/verifier/reward.json",
        gate_threshold=0.67,
        max_subtasks=3,
        max_attempts_per_subtask=2,
        episode_timeout_s=3600.0,
        per_turn_timeout_s=600.0,
        task_description=(
            "Build a lossless codec for Jupyter notebooks exposed as "
            "fit/compress/decompress stages. Scored by geom_mean_ratio "
            "with byte-exact round-trip as a hard gate."
        ),
        task_domain="systems / compression",
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh. Reads /logs/verifier/reward.json. "
            "status=fail scores 0.0. Otherwise geom_mean_ratio is normalized to "
            "[0,1] via (1.0 - r) / (1.0 - 0.15)."
        ),
        cpus=16,
        memory_mb=32768,
    )


def notebook_demo_config() -> TaskConfig:
    return TaskConfig(
        task_name="notebook-compression",
        docker_image="frontier-swe-notebook:latest",
        instruction=_load_upstream_instruction(),
        workspace_dir="/app",
        build_command=":",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="bash /opt/verifier/test.sh",
        visible_test_total=80,
        l1_score_mode="reward_json",
        l1_timeout_s=3000.0,
        reward_json_path="/logs/verifier/reward.json",
        gate_threshold=0.67,
        max_subtasks=5,
        max_attempts_per_subtask=3,
        episode_timeout_s=7200.0,
        per_turn_timeout_s=900.0,
        task_description=(
            "Build a lossless codec for Jupyter notebooks (demo mode: "
            "longer budgets and more subtasks)."
        ),
        task_domain="systems / compression",
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh. Reads /logs/verifier/reward.json. "
            "status=fail scores 0.0. Otherwise geom_mean_ratio is normalized."
        ),
        cpus=16,
        memory_mb=32768,
    )
