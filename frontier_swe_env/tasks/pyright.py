from __future__ import annotations

from pathlib import Path

from ..task_config import TaskConfig


PYRIGHT_TRAINING_INSTRUCTION = """
# Pyright Type Checking Optimization

Pyright is a fast, full-featured Python type checker written in TypeScript
(version 1.1.400, pinned). Your goal is to make it faster without breaking
correctness.

## Quick Start

```bash
# Check baseline performance on a benchmark
/app/baseline/pyright --stats /app/benchmarks/unions/

# Make changes to the TypeScript source, then rebuild
cd /app/pyright-src/packages/pyright && npm run build

# Run the convenience script (rebuild + parity check + benchmark)
/app/run_dev_bench.sh

# Check remaining time
cat /app/.timer/remaining_secs
```
"""


def _load_upstream_instruction() -> str:
    """Return the upstream instruction.md if present, else the training text."""
    upstream = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "pyright-type-checking-optimization"
        / "instruction.md"
    )
    if upstream.is_file():
        return upstream.read_text()
    return PYRIGHT_TRAINING_INSTRUCTION


def pyright_training_config() -> TaskConfig:
    return TaskConfig(
        task_name="pyright-type-checking-optimization",
        docker_image="frontier-swe-pyright:latest",
        instruction=_load_upstream_instruction(),
        workspace_dir="/app",
        build_command=":",
        gate_script_path="/app/run_dev_bench.sh",
        visible_test_command="bash /app/run_dev_bench.sh",
        visible_test_total=10,
        l1_score_mode="reward_json",
        l1_timeout_s=3600.0,
        reward_json_path="/logs/verifier/reward.json",
        gate_threshold=1.0,
        max_subtasks=5,
        max_attempts_per_subtask=3,
        episode_timeout_s=72000.0,
        per_turn_timeout_s=1200.0,
        task_description=(
            "Optimize Pyright's type checking engine for performance without "
            "breaking correctness. Scored by geometric mean of speedup ratios."
        ),
        task_domain="systems / performance-optimization",
        scoring_context=(
            "L1 runs the verifier test suite. Reads /logs/verifier/reward.json. "
            "status=fail scores 0.0. Otherwise reward is geometric mean of speedups."
        ),
        cpus=8,
        memory_mb=32768,
    )


def pyright_demo_config() -> TaskConfig:
    return TaskConfig(
        task_name="pyright-type-checking-optimization",
        docker_image="frontier-swe-pyright:latest",
        instruction=_load_upstream_instruction(),
        workspace_dir="/app",
        build_command=":",
        gate_script_path="/app/run_dev_bench.sh",
        visible_test_command="bash /app/run_dev_bench.sh",
        visible_test_total=10,
        l1_score_mode="reward_json",
        l1_timeout_s=3600.0,
        reward_json_path="/logs/verifier/reward.json",
        gate_threshold=1.0,
        max_subtasks=10,
        max_attempts_per_subtask=5,
        episode_timeout_s=86400.0,
        per_turn_timeout_s=2400.0,
        task_description=(
            "Optimize Pyright's type checking engine for performance (demo mode: "
            "longer budgets and more subtasks)."
        ),
        task_domain="systems / performance-optimization",
        scoring_context=(
            "L1 runs the verifier test suite. Reads /logs/verifier/reward.json. "
            "status=fail scores 0.0. Otherwise reward is geometric mean of speedups."
        ),
        cpus=8,
        memory_mb=32768,
    )
