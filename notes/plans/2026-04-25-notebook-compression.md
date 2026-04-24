# Notebook Compression Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Onboard `notebook-compression` as a second task on the task-agnostic OpenEnv core, so a pi-driven agent can run a plan/submit/advance episode producing a deterministic `geom_mean_ratio`-based training signal.

**Architecture:** Task assets vendored verbatim under `tasks/notebook-compression/` (already done); image consumes only `tests/*` + workspace `run`; two task-agnostic core generalizations (`TaskConfig.l1_timeout_s`, `TaskConfig.reward_json_path`, `TestOutputRubric` gains `score_mode="reward_json"` reading the upstream verifier's structured `reward.json`); new `Dockerfile.notebook` extends `openenv-base:latest` with scientific-Python + compression libs and a build-time visible-corpus splitter.

**Tech Stack:** Python 3.11 (in container), Python 3.13 (repo), FastAPI, FastMCP, pydantic, pytest, zstd/brotli/lz4, nbformat, podman.

**Spec:** `notes/specs/2026-04-24-notebook-compression-design.md`

**Note on tracked vs untracked paths:** The repo's `.gitignore` blanket-ignores `tests/`, `docs/`, `artifacts/`, `*.json`, `*.txt`. Per user decision, tests and local validation artifacts are NOT committed — they exist only on the implementer's disk for verification. **Do not `git add` tests/ or artifacts/.** Commit steps below stage only source files that live outside the ignored paths.

---

## Pre-flight

- [ ] **Step 0.1: Confirm branch and clean tree**

Run:
```bash
git status
git branch --show-current
```
Expected: on `feat/issue-4-notebook-compression`, clean or only plan/notes changes staged.

- [ ] **Step 0.2: Confirm vendored task folder is present**

Run:
```bash
ls tasks/notebook-compression/ && du -sh tasks/notebook-compression/
```
Expected: lists `environment instruction.md job.yaml oracle.yaml scripts sources task.toml tests`; size around 70M (dominated by `tests/hidden_test_set_bundle.zip`).

- [ ] **Step 0.3: Create tests/ directory + conftest**

Files:
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import sys
from pathlib import Path

# Allow tests to import frontier_swe_env without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 0.4: Install test deps**

Run:
```bash
uv pip install --group test 2>/dev/null || uv pip install pytest pytest-asyncio
uv pip install pydantic
```
Expected: pytest resolves. (The repo's python is 3.13 per `pyproject.toml`.)

- [ ] **Step 0.5: (No commit — `tests/` is gitignored; files exist locally only)**

---

## Task 1: Add `l1_timeout_s` and `reward_json_path` to `TaskConfig`

**Files:**
- Modify: `frontier_swe_env/task_config.py`
- Test: `tests/test_task_config.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_task_config.py`:

```python
from frontier_swe_env.task_config import TaskConfig


def _minimal_kwargs():
    return dict(
        task_name="demo",
        docker_image="demo:latest",
        instruction="noop",
        workspace_dir="/app",
        build_command=":",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="bash /opt/verifier/test.sh",
        visible_test_total=1,
        max_subtasks=1,
        max_attempts_per_subtask=1,
        episode_timeout_s=60,
    )


def test_task_config_has_l1_timeout_default():
    cfg = TaskConfig(**_minimal_kwargs())
    assert cfg.l1_timeout_s == 300.0


def test_task_config_has_reward_json_path_default():
    cfg = TaskConfig(**_minimal_kwargs())
    assert cfg.reward_json_path == "/logs/verifier/reward.json"


def test_task_config_accepts_overrides():
    cfg = TaskConfig(**_minimal_kwargs(), l1_timeout_s=1800, reward_json_path="/tmp/r.json")
    assert cfg.l1_timeout_s == 1800
    assert cfg.reward_json_path == "/tmp/r.json"
```

- [ ] **Step 1.2: Run test — expect FAIL**

Run:
```bash
pytest tests/test_task_config.py -v
```
Expected: three failures — `AttributeError: 'TaskConfig' object has no attribute 'l1_timeout_s'` (and same for `reward_json_path`), or pydantic validation error on unknown field.

- [ ] **Step 1.3: Add fields to TaskConfig**

Modify `frontier_swe_env/task_config.py` — in the `TaskConfig(BaseModel)` body, add these fields just after `per_turn_timeout_s`:

```python
    # L1 test-command timeout (seconds). Some verifiers (e.g. notebook
    # compression) run fit/compress/decompress stages and need more than
    # the default 300s.
    l1_timeout_s: float = 300.0
    # Path to the structured reward.json written by the test command when
    # l1_score_mode == "reward_json".
    reward_json_path: str = "/logs/verifier/reward.json"
```

- [ ] **Step 1.4: Run test — expect PASS**

Run:
```bash
pytest tests/test_task_config.py -v
```
Expected: all three tests PASS.

- [ ] **Step 1.5: Commit (source file only; tests are untracked)**

```bash
git add frontier_swe_env/task_config.py
git commit -m "feat(task-config): add l1_timeout_s and reward_json_path fields"
```

---

## Task 2: Add `score_mode="reward_json"` to `TestOutputRubric`

**Files:**
- Modify: `frontier_swe_env/rubrics/l1_tests.py`
- Test: `tests/test_l1_tests.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_l1_tests.py`:

```python
import json
from pathlib import Path

import pytest

from frontier_swe_env.rubrics.l1_tests import TestOutputRubric


def _write_reward(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "reward.json"
    p.write_text(json.dumps(payload))
    return p


def test_reward_json_ok_maps_low_ratio_to_high_score(tmp_path):
    reward_path = _write_reward(tmp_path, {"status": "ok", "geom_mean_ratio": 0.15})
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    score = rubric.forward(None, None)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_reward_json_ok_maps_high_ratio_to_low_score(tmp_path):
    reward_path = _write_reward(tmp_path, {"status": "ok", "geom_mean_ratio": 1.0})
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    assert rubric.forward(None, None) == pytest.approx(0.0, abs=1e-6)


def test_reward_json_ok_maps_mid_ratio_to_mid_score(tmp_path):
    reward_path = _write_reward(tmp_path, {"status": "ok", "geom_mean_ratio": 0.575})
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    # (1.0 - 0.575) / (1.0 - 0.15) = 0.5
    assert rubric.forward(None, None) == pytest.approx(0.5, abs=1e-3)


def test_reward_json_fail_status_scores_zero(tmp_path):
    reward_path = _write_reward(
        tmp_path,
        {"status": "fail", "reason": "round-trip mismatch", "geom_mean_ratio": None},
    )
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    assert rubric.forward(None, None) == 0.0


def test_reward_json_missing_file_scores_zero(tmp_path):
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(tmp_path / "does_not_exist.json"),
    )
    assert rubric.forward(None, None) == 0.0


def test_reward_json_malformed_scores_zero(tmp_path):
    p = tmp_path / "reward.json"
    p.write_text("not json{{{")
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(p),
    )
    assert rubric.forward(None, None) == 0.0


def test_reward_json_ratio_above_rmax_clamps_to_zero(tmp_path):
    reward_path = _write_reward(tmp_path, {"status": "ok", "geom_mean_ratio": 5.0})
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    assert rubric.forward(None, None) == 0.0


def test_reward_json_ratio_below_rmin_clamps_to_one(tmp_path):
    reward_path = _write_reward(tmp_path, {"status": "ok", "geom_mean_ratio": 0.01})
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    assert rubric.forward(None, None) == pytest.approx(1.0, abs=1e-6)


def test_last_reward_cache_populated_on_reward_json_mode(tmp_path):
    payload = {"status": "ok", "geom_mean_ratio": 0.4, "reason": "ok"}
    reward_path = _write_reward(tmp_path, payload)
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(reward_path),
    )
    rubric.forward(None, None)
    assert rubric.last_reward == payload


def test_last_reward_cache_none_when_missing(tmp_path):
    rubric = TestOutputRubric(
        test_command="true",
        score_mode="reward_json",
        reward_json_path=str(tmp_path / "nope.json"),
    )
    rubric.forward(None, None)
    assert rubric.last_reward is None


def test_ratio_mode_still_works(tmp_path):
    # Existing behavior must not regress — regex "Total: N/M passed" still works
    rubric = TestOutputRubric(
        test_command="echo 'Total: 5/10 passed'",
        output_pattern=r"Total:\s*(\d+)/(\d+)\s*passed",
        score_mode="ratio",
    )
    assert rubric.forward(None, None) == pytest.approx(0.5, abs=1e-6)
```

- [ ] **Step 2.2: Run tests — expect FAIL**

Run:
```bash
pytest tests/test_l1_tests.py -v
```
Expected: the `reward_json` tests fail (unknown score_mode falls through to `_parse_ratio` which returns 0.0 — most tests will still appear to "pass by coincidence" returning 0.0 when the right answer is non-zero). Specifically these three will FAIL: `test_reward_json_ok_maps_low_ratio_to_high_score`, `test_reward_json_ok_maps_mid_ratio_to_mid_score`, `test_reward_json_ratio_below_rmin_clamps_to_one`, `test_last_reward_cache_populated_on_reward_json_mode`. If `reward_json_path` isn't accepted by `__init__`, all of them fail with TypeError — that's fine.

- [ ] **Step 2.3: Implement the reward_json mode**

Replace the body of `frontier_swe_env/rubrics/l1_tests.py` with:

```python
"""L1b: Test output rubric — runs a test command and parses the score.

Supports multiple score modes:
- "ratio":       parse numerator/denominator (e.g. "Total: 6/72 passed")
- "speedup":     parse speedup multiplier (e.g. "Speedup: 1.45x")
- "compression": parse compression ratio from stdout (e.g. "Ratio: 0.312")
- "reward_json": read a structured reward.json (status + geom_mean_ratio)
                 produced by a Harbor-style verifier (notebook-compression).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from openenv.core.rubrics.base import Rubric


class TestOutputRubric(Rubric):
    """Run a test command and derive a score in [0, 1].

    In ``reward_json`` mode the test command is run for its side-effect of
    writing ``reward_json_path``; scoring comes from parsing that JSON.
    The last parsed payload is cached on ``self.last_reward`` so callers
    can surface per-notebook metadata in feedback.
    """

    # reward_json normalization anchors: ratio at or above R_MAX → 0.0,
    # ratio at or below R_MIN → 1.0, linear in between. See spec
    # "Reward Normalization Anchors" for rationale.
    R_MAX = 1.0
    R_MIN = 0.15

    def __init__(
        self,
        test_command: str = "bash /app/test.sh",
        output_pattern: str = r"Total:\s*(\d+)/(\d+)\s*passed",
        score_mode: str = "ratio",
        reward_json_path: str = "/logs/verifier/reward.json",
        port: int = 0,
        host: str = "127.0.0.1",
        timeout_s: int = 300,
    ):
        super().__init__()
        self.test_command = test_command
        self.output_pattern = output_pattern
        self.score_mode = score_mode
        self.reward_json_path = reward_json_path
        self.port = port
        self.host = host
        self.timeout_s = timeout_s
        # Cache of the last parsed reward.json payload (or None if missing/bad)
        self.last_reward: Optional[dict] = None

    def forward(self, action: Any, observation: Any) -> float:
        env = {**os.environ, "PG_PORT": str(self.port), "PG_HOST": self.host}
        try:
            result = subprocess.run(
                ["bash", "-c", self.test_command],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            if self.score_mode == "reward_json":
                self.last_reward = None
            return 0.0

        if self.score_mode == "reward_json":
            return self._parse_reward_json()
        return self._parse_stdout(result.stdout)

    # ---- reward_json mode ----

    def _parse_reward_json(self) -> float:
        path = Path(self.reward_json_path)
        if not path.is_file():
            self.last_reward = None
            return 0.0
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            self.last_reward = None
            return 0.0

        self.last_reward = payload

        if payload.get("status") != "ok":
            return 0.0

        ratio = payload.get("geom_mean_ratio")
        if ratio is None:
            return 0.0
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            return 0.0

        span = self.R_MAX - self.R_MIN
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.R_MAX - r) / span))

    # ---- stdout-regex modes (unchanged semantics) ----

    def _parse_stdout(self, stdout: str) -> float:
        match = re.search(self.output_pattern, stdout)
        if not match:
            return 0.0
        if self.score_mode == "ratio":
            return self._parse_ratio(match)
        if self.score_mode == "speedup":
            return self._parse_speedup(match)
        if self.score_mode == "compression":
            return self._parse_compression(match)
        return self._parse_ratio(match)

    @staticmethod
    def _parse_ratio(match: re.Match) -> float:
        try:
            passed = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                return passed / total
        except (IndexError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _parse_speedup(match: re.Match) -> float:
        try:
            speedup = float(match.group(1))
            return max(0.0, min((speedup - 1.0) * 5.0, 1.0))
        except (IndexError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _parse_compression(match: re.Match) -> float:
        try:
            ratio = float(match.group(1))
            return max(0.0, min((0.5 - ratio) / 0.5, 1.0))
        except (IndexError, ValueError):
            pass
        return 0.0


# Backward-compatible alias
PGCompatTestRubric = TestOutputRubric
```

- [ ] **Step 2.4: Run tests — expect PASS**

Run:
```bash
pytest tests/test_l1_tests.py -v
```
Expected: all tests pass.

- [ ] **Step 2.5: Commit (source file only; tests are untracked)**

```bash
git add frontier_swe_env/rubrics/l1_tests.py
git commit -m "feat(rubric): add score_mode='reward_json' to TestOutputRubric"
```

---

## Task 3: Thread `l1_timeout_s` + `reward_json_path` through the environment, enrich subtask feedback

**Files:**
- Modify: `frontier_swe_env/server/frontier_swe_env_environment.py`

No new tests — this is pure wiring, covered later by the integration smoke (Task 9).

- [ ] **Step 3.1: Update `TestOutputRubric` construction**

In `frontier_swe_env/server/frontier_swe_env_environment.py`, find the `TestOutputRubric(...)` construction (currently around line 75) and replace with:

```python
        self.test_rubric = TestOutputRubric(
            test_command=self.task_config.visible_test_command,
            output_pattern=self.task_config.l1_output_pattern,
            score_mode=self.task_config.l1_score_mode,
            reward_json_path=self.task_config.reward_json_path,
            timeout_s=int(self.task_config.l1_timeout_s),
        )
```

- [ ] **Step 3.2: Enrich `submit_subtask_payload` when `reward_json` mode is active**

In `frontier_swe_env/server/frontier_swe_env_environment.py`, locate the `submit_subtask_payload` method. Find the block that starts with `l1_summary = (` and runs through the method's final `return {...}`. Replace that entire block (from `l1_summary = (` through the closing `}` of the return) with:

```python
        l1_extras: dict = {}
        if self.task_config.l1_score_mode == "reward_json":
            reward = getattr(self.test_rubric, "last_reward", None)
            if reward is not None:
                l1_extras = {
                    "status": reward.get("status"),
                    "reason": reward.get("reason"),
                    "geom_mean_ratio": reward.get("geom_mean_ratio"),
                    "compression_score": reward.get("compression_score"),
                    "stage_timings": {
                        "fit_elapsed_sec": reward.get("fit_elapsed_sec"),
                        "compress_elapsed_sec": reward.get("compress_elapsed_sec"),
                        "decompress_elapsed_sec": reward.get("decompress_elapsed_sec"),
                    },
                }
                l1_summary = (
                    f"Gate: {gate_score:.2f} | "
                    f"Verifier: status={reward.get('status')}, "
                    f"geom_mean_ratio={reward.get('geom_mean_ratio')}, "
                    f"reason={reward.get('reason')} | "
                    f"L1 blended: {l1_score:.2f}"
                )
            else:
                l1_summary = (
                    f"Gate: {gate_score:.2f} | Verifier: no reward.json produced | "
                    f"L1 blended: {l1_score:.2f}"
                )
        else:
            l1_summary = (
                f"Gate: {gate_score:.2f}, "
                f"Compat tests: {l1_test_score:.2f}, "
                f"L1 blended: {l1_score:.2f}"
            )

        # L2 scoring (async LLM judge)
        l2_result = await self.l2_rubric.grade(
            subtask_description=subtask.get("description", ""),
            acceptance_criteria=subtask.get("acceptance_criteria", ""),
            l1_summary=l1_summary,
        )
        l2_score = l2_result.normalized

        # Blend L1 and L2
        blended = (
            1.0 - self.task_config.l2_weight
        ) * l1_score + self.task_config.l2_weight * l2_score

        # Track best score
        prev_best = self.episode_state.frozen_scores.get(subtask_id, 0.0)
        self.episode_state.frozen_scores[subtask_id] = max(prev_best, blended)

        attempts_remaining = (
            self.episode_state.max_attempts_per_subtask
            - self.episode_state.attempts[subtask_id]
        )

        logger.info(
            "Subtask %s attempt %d: gate=%.2f l1_test=%.2f l1=%.2f l2=%.2f blended=%.2f (best=%.2f)",
            subtask_id,
            self.episode_state.attempts[subtask_id],
            gate_score,
            l1_test_score,
            l1_score,
            l2_score,
            blended,
            self.episode_state.frozen_scores[subtask_id],
        )

        response = {
            "score": round(blended, 4),
            "l1_score": round(l1_score, 4),
            "l2_score": round(l2_score, 4),
            "gate_score": round(gate_score, 4),
            "test_score": round(l1_test_score, 4),
            "best_score": round(self.episode_state.frozen_scores[subtask_id], 4),
            "feedback": l2_result.feedback,
            "attempts_remaining": attempts_remaining,
        }
        if l1_extras:
            response["l1_extras"] = l1_extras
        return response
```

Delete the now-superseded `l1_summary = ...` assignment that appeared before this block.

- [ ] **Step 3.3: Verify no existing tests regressed**

Run:
```bash
pytest tests/ -v
```
Expected: all tests still pass (6 config tests + 10 rubric tests — no env-level tests yet).

- [ ] **Step 3.4: Syntax-check the modified file**

Run:
```bash
python3 -c "import ast; ast.parse(open('frontier_swe_env/server/frontier_swe_env_environment.py').read())"
```
Expected: no output (clean parse).

- [ ] **Step 3.5: Commit**

```bash
git add frontier_swe_env/server/frontier_swe_env_environment.py
git commit -m "feat(env): thread l1_timeout_s + reward_json_path; enrich subtask feedback"
```

---

## Task 4: Visible-corpus splitter

**Files:**
- Create: `scripts/split_visible_corpus.py`

- [ ] **Step 4.1: Write the splitter**

Create `scripts/split_visible_corpus.py`:

```python
#!/usr/bin/env python3
"""Deterministically carve a 'visible' corpus out of the hidden test bundle.

The upstream Harbor orchestrator bind-mounts a visible corpus at
$DATA_ROOT/visible/ from a separate dataset volume. Our standalone
image has no such orchestrator, so we synthesize the visible corpus
at image-build time by taking a seeded random subset of the hidden
bundle's notebook files.

See decision-log D-009 for the rationale (and the reward-hacking
caveat that visible ⊂ hidden).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Path to hidden_test_set_bundle.zip")
    parser.add_argument("--out", required=True, help="Output directory for visible corpus")
    parser.add_argument("--manifest", required=True, help="Output path for manifest.json")
    parser.add_argument("--ratio", type=float, default=0.75, help="Fraction of files in the visible split")
    parser.add_argument("--seed", type=int, default=17, help="Deterministic shuffle seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bundle = Path(args.bundle)
    if not bundle.is_file():
        print(f"ERROR: bundle not found: {bundle}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    manifest_path = Path(args.manifest)

    # Clean target
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nbc_split_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(tmp)

        files_root = tmp / "hidden_test_set_bundle" / "files"
        if not files_root.is_dir():
            print(f"ERROR: bundle is missing hidden_test_set_bundle/files/: {files_root}", file=sys.stderr)
            return 2

        all_files = sorted(p for p in files_root.iterdir() if p.is_file())
        if not all_files:
            print("ERROR: no files in bundle", file=sys.stderr)
            return 2

        rng = random.Random(args.seed)
        shuffled = list(all_files)
        rng.shuffle(shuffled)
        n_visible = max(1, int(round(len(shuffled) * args.ratio)))
        visible = shuffled[:n_visible]

        for src in visible:
            shutil.copy2(src, out_dir / src.name)

    manifest = {
        "corpus": "notebook-compression-visible",
        "source_bundle": bundle.name,
        "ratio": args.ratio,
        "seed": args.seed,
        "count": n_visible,
        "files": sorted(p.name for p in visible),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {n_visible} files to {out_dir} and manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.2: Smoke-run the splitter locally**

Run:
```bash
python3 scripts/split_visible_corpus.py \
  --bundle tasks/notebook-compression/tests/hidden_test_set_bundle.zip \
  --out /tmp/nbc_visible \
  --manifest /tmp/nbc_manifest.json \
  --ratio 0.75 --seed 17
ls /tmp/nbc_visible/ | wc -l
head -30 /tmp/nbc_manifest.json
rm -rf /tmp/nbc_visible /tmp/nbc_manifest.json
```
Expected: roughly 60 files (75% of 80); manifest includes `corpus`, `count`, `files` array.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/split_visible_corpus.py
git commit -m "feat(scripts): add visible-corpus splitter for notebook task build"
```

---

## Task 5: Notebook gate-check script

**Files:**
- Create: `scripts/notebook_gate_checks.sh`

- [ ] **Step 5.1: Write the gate script**

Create `scripts/notebook_gate_checks.sh`:

```bash
#!/usr/bin/env bash
# Gate checks for the notebook-compression task.
# Outputs GATE_SCORE=N/3 on the last line. Cheap, always-run — catches
# obviously-broken submissions before spending a multi-minute verifier run.
set -uo pipefail

GATE=0
TOTAL=3
DATA_ROOT="${DATA_ROOT:-/mnt/notebook-data}"

# ---------- Gate 1: /app/run exists and is executable ----------
if [ -x /app/run ]; then
    GATE=$((GATE + 1))
    echo "GATE 1 PASS: /app/run exists and is executable"
else
    echo "GATE 1 FAIL: /app/run missing or not executable"
fi

# ---------- Gate 2: visible corpus is populated ----------
if [ -d "${DATA_ROOT}/visible" ] && [ -n "$(ls -A "${DATA_ROOT}/visible" 2>/dev/null)" ]; then
    GATE=$((GATE + 1))
    echo "GATE 2 PASS: visible corpus present at ${DATA_ROOT}/visible"
else
    echo "GATE 2 FAIL: visible corpus missing at ${DATA_ROOT}/visible"
fi

# ---------- Gate 3: python3 + zstandard + nbformat importable ----------
if python3 -c 'import zstandard, nbformat' 2>/dev/null; then
    GATE=$((GATE + 1))
    echo "GATE 3 PASS: python3 zstandard/nbformat available"
else
    echo "GATE 3 FAIL: python3 imports failed"
fi

echo "GATE_SCORE=${GATE}/${TOTAL}"
```

- [ ] **Step 5.2: Mark executable**

Run:
```bash
chmod +x scripts/notebook_gate_checks.sh
```

- [ ] **Step 5.3: Smoke-run locally (expect partial pass on host)**

Run:
```bash
bash scripts/notebook_gate_checks.sh | tail -5
```
Expected: `GATE_SCORE=0/3` or `1/3` on host (depends on host python), since `/app/run` and `/mnt/notebook-data` don't exist on the host. We only need to confirm the script runs and emits the `GATE_SCORE=` line.

- [ ] **Step 5.4: Commit**

```bash
git add scripts/notebook_gate_checks.sh
git commit -m "feat(scripts): add notebook-compression gate checks (3 gates)"
```

---

## Task 6: Notebook task config module + registry registration

**Files:**
- Create: `frontier_swe_env/tasks/notebook_compression.py`
- Modify: `frontier_swe_env/tasks/__init__.py`
- Test: `tests/test_notebook_task_config.py`

- [ ] **Step 6.1: Write the failing test**

Create `tests/test_notebook_task_config.py`:

```python
import pytest

from frontier_swe_env.tasks import get_task_config, list_tasks


def test_notebook_registered():
    tasks = list_tasks()
    assert "notebook" in tasks
    assert "notebook-compression" in tasks


@pytest.mark.parametrize("name", ["notebook", "notebook-compression"])
def test_notebook_training_config_shape(name):
    cfg = get_task_config(name, "training")
    assert cfg.task_name == "notebook-compression"
    assert cfg.docker_image == "frontier-swe-notebook:latest"
    assert cfg.workspace_dir == "/app"
    assert cfg.gate_script_path == "/app/gate_checks.sh"
    assert cfg.visible_test_command == "bash /opt/verifier/test.sh"
    assert cfg.l1_score_mode == "reward_json"
    assert cfg.l1_timeout_s >= 1200
    assert cfg.episode_timeout_s <= 7200
    assert cfg.max_subtasks <= 5
    assert cfg.max_attempts_per_subtask <= 3
    assert "notebook" in cfg.task_description.lower() or "compress" in cfg.task_description.lower()


def test_notebook_demo_config_shape():
    training = get_task_config("notebook", "training")
    demo = get_task_config("notebook", "demo")
    assert demo.l1_score_mode == "reward_json"
    # demo mode should be at least as generous as training mode
    assert demo.episode_timeout_s >= training.episode_timeout_s
    assert demo.l1_timeout_s >= 1800
    assert demo.max_subtasks >= training.max_subtasks
```

- [ ] **Step 6.2: Run test — expect FAIL**

Run:
```bash
pytest tests/test_notebook_task_config.py -v
```
Expected: `ValueError: Unknown task 'notebook'`.

- [ ] **Step 6.3: Write the config module**

Create `frontier_swe_env/tasks/notebook_compression.py`:

```python
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
```

- [ ] **Step 6.4: Register the task**

Modify `frontier_swe_env/tasks/__init__.py` — append after the existing `register_task("pg", ...)` line:

```python
from .notebook_compression import notebook_demo_config, notebook_training_config  # noqa: E402

register_task("notebook", notebook_training_config, notebook_demo_config)
register_task("notebook-compression", notebook_training_config, notebook_demo_config)
```

- [ ] **Step 6.5: Run test — expect PASS**

Run:
```bash
pytest tests/test_notebook_task_config.py -v
```
Expected: all pass.

- [ ] **Step 6.6: Re-run full test suite to check for regressions**

Run:
```bash
pytest tests/ -v
```
Expected: all previous tests still pass.

- [ ] **Step 6.7: Commit (source files only; tests are untracked)**

```bash
git add frontier_swe_env/tasks/notebook_compression.py frontier_swe_env/tasks/__init__.py
git commit -m "feat(tasks): register notebook-compression training + demo configs"
```

---

## Task 7: Build the openenv base image (prerequisite)

**Files:**
- None modified — just validates the base image exists.

- [ ] **Step 7.1: Check if base image already exists**

Run:
```bash
podman images openenv-base:latest --format '{{.Repository}}:{{.Tag}}'
```
Expected: either `openenv-base:latest` (already built — skip 7.2) or empty (proceed to 7.2).

- [ ] **Step 7.2: Build the base image (if missing)**

Run:
```bash
podman build -f docker/Dockerfile.base -t openenv-base:latest .
```
Expected: build completes successfully. This is slow (~5-10 min first time). The base installs Node.js, pi, OpenEnv, etc.

- [ ] **Step 7.3: Verify base image works**

Run:
```bash
podman run --rm openenv-base:latest python3 -c "import openenv; print('openenv imports ok')"
```
Expected: `openenv imports ok`.

- [ ] **Step 7.4: No commit**

(No repo changes.)

---

## Task 8: Notebook Dockerfile

**Files:**
- Create: `docker/Dockerfile.notebook`

- [ ] **Step 8.1: Write the Dockerfile**

Create `docker/Dockerfile.notebook`:

```dockerfile
# Notebook Compression — Task Image
#
# Extends openenv-base with compression tooling, scientific Python deps,
# the vendored upstream verifier, and a build-time-synthesized visible
# corpus.
#
# Build (must build base first):
#   podman build -f docker/Dockerfile.base     -t openenv-base:latest .
#   podman build -f docker/Dockerfile.notebook -t frontier-swe-notebook:latest .
#
# Run:
#   podman run -p 8000:8000 frontier-swe-notebook:latest

FROM openenv-base:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV DATA_ROOT=/mnt/notebook-data
ENV TASK_BUDGET_SECS=3600
ENV FSWE_TASK_NAME=notebook
ENV FSWE_TASK_MODE=training

# System compression tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    zstd \
    brotli \
    lz4 \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    unzip \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Scientific Python + compression bindings
RUN uv pip install --system \
    numpy \
    pandas \
    scipy \
    pyarrow \
    joblib \
    tqdm \
    nbformat \
    jsonschema \
    datasketch \
    zstandard \
    brotli \
    lz4

# Workspace stub (upstream run script — starts as a failing scaffold)
COPY tasks/notebook-compression/environment/workspace/run /app/run
RUN chmod +x /app/run

# Verifier scripts + hidden bundle
RUN mkdir -p /opt/verifier /logs/verifier /mnt/notebook-data
COPY tasks/notebook-compression/tests/compute_reward.py /opt/verifier/
COPY tasks/notebook-compression/tests/scoring_core.py /opt/verifier/
COPY tasks/notebook-compression/tests/test.sh /opt/verifier/
COPY tasks/notebook-compression/tests/hidden_test_set_bundle.zip /opt/verifier/
RUN chmod +x /opt/verifier/test.sh

# Build-time visible-corpus synthesis (see decision-log D-009)
COPY scripts/split_visible_corpus.py /tmp/split_visible_corpus.py
RUN python3 /tmp/split_visible_corpus.py \
    --bundle /opt/verifier/hidden_test_set_bundle.zip \
    --out /mnt/notebook-data/visible \
    --manifest /mnt/notebook-data/manifest.json \
    --ratio 0.75 --seed 17 \
    && rm /tmp/split_visible_corpus.py

# Gate checks
COPY scripts/notebook_gate_checks.sh /app/gate_checks.sh
RUN chmod +x /app/gate_checks.sh

# OpenEnv core code
COPY frontier_swe_env/ /opt/openenv/frontier_swe_env/
COPY pyproject.toml /opt/openenv/pyproject.toml
ENV PYTHONPATH="/opt/openenv"

# Git baseline for L2 diff tracking
RUN cd /app \
    && git config --global user.email "agent@frontier-swe-openenv" \
    && git config --global user.name "agent" \
    && git init && git add -A && git commit -m "initial stub"

# Re-copy entrypoint (matches Dockerfile.pg pattern for explicitness)
COPY docker/openenv_entrypoint.sh /app/openenv_entrypoint.sh
RUN chmod +x /app/openenv_entrypoint.sh

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

- [ ] **Step 8.2: Build the image**

Run:
```bash
podman build -f docker/Dockerfile.notebook -t frontier-swe-notebook:latest .
```
Expected: build completes (~5-10 min). Final image around 2.5 GB.

- [ ] **Step 8.3: Verify key paths exist in the image**

Run:
```bash
podman run --rm frontier-swe-notebook:latest bash -c '
  test -x /app/run &&
  test -x /app/gate_checks.sh &&
  test -x /opt/verifier/test.sh &&
  test -f /opt/verifier/compute_reward.py &&
  test -f /opt/verifier/scoring_core.py &&
  test -f /opt/verifier/hidden_test_set_bundle.zip &&
  test -d /mnt/notebook-data/visible &&
  test -f /mnt/notebook-data/manifest.json &&
  ls /mnt/notebook-data/visible | wc -l &&
  python3 -c "import zstandard, brotli, lz4, nbformat; print(\"py deps ok\")"
'
```
Expected: no failures; count of visible files prints ~60; "py deps ok".

- [ ] **Step 8.4: Commit**

```bash
git add docker/Dockerfile.notebook
git commit -m "feat(docker): add Dockerfile.notebook extending openenv-base"
```

---

## Task 9: Verifier smoke tests inside the container

**Files:**
- None — validation only.

- [ ] **Step 9.1: Smoke-test gate script in container**

Run:
```bash
podman run --rm frontier-swe-notebook:latest bash /app/gate_checks.sh
```
Expected: `GATE_SCORE=3/3` — all three gates pass in the freshly-built image (the stub `/app/run` is still executable even though it exits 1 when invoked).

- [ ] **Step 9.2: Run verifier against stub `/app/run` — expect fail**

Run:
```bash
podman run --rm frontier-swe-notebook:latest bash -c '
  bash /opt/verifier/test.sh 2>&1 | tail -30
  echo "---reward.json:"
  cat /logs/verifier/reward.json
'
```
Expected: `test.sh` runs to completion; `reward.json` exists with `"status": "fail"` and a `reason` mentioning stub/fit-failure. `geom_mean_ratio` is null.

- [ ] **Step 9.3: Run verifier against a trivial zstd wrapper — expect ok**

Create a throwaway test script (not committed):

Run:
```bash
cat > /tmp/trivial_run <<'EOF'
#!/usr/bin/env python3
"""Trivial zstd wrapper for verifier smoke-testing. Not committed."""
import sys
import shutil
import subprocess
from pathlib import Path


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def fit(visible, artifact):
    Path(artifact).mkdir(parents=True, exist_ok=True)
    (Path(artifact) / "marker").write_text("trivial-zstd-v1\n")


def compress(artifact, in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(in_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(in_dir)
        dst = out_dir / (str(rel) + ".zst")
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["zstd", "-19", "-q", "-f", "-o", str(dst), str(src)])


def decompress(artifact, in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(in_dir.rglob("*")):
        if not src.is_file() or src.suffix != ".zst":
            continue
        rel = src.relative_to(in_dir).with_suffix("")
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["zstd", "-d", "-q", "-f", "-o", str(dst), str(src)])


def main():
    if len(sys.argv) < 2:
        die("usage: run {fit|compress|decompress} ...")
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "fit" and len(args) == 2:
        fit(*args)
    elif cmd == "compress" and len(args) == 3:
        compress(*args)
    elif cmd == "decompress" and len(args) == 3:
        decompress(*args)
    else:
        die(f"bad command/args: {cmd} {args}")


if __name__ == "__main__":
    main()
EOF
chmod +x /tmp/trivial_run
```

Then run:
```bash
podman run --rm -v /tmp/trivial_run:/app/run:ro frontier-swe-notebook:latest bash -c '
  bash /opt/verifier/test.sh 2>&1 | tail -40
  echo "---reward.json status + ratio:"
  python3 -c "import json; d=json.load(open(\"/logs/verifier/reward.json\")); print(d[\"status\"], d.get(\"geom_mean_ratio\"))"
'
```
Expected: status=`ok`, `geom_mean_ratio` in roughly (0.2, 0.6). Run takes ~3-10 minutes.

- [ ] **Step 9.4: Clean up test artifact**

Run:
```bash
rm /tmp/trivial_run
```

- [ ] **Step 9.5: No commit** (validation only).

---

## Task 10: End-to-end episode with OpenEnv server

**Files:**
- None modified — validation only.

- [ ] **Step 10.1: Start the container in the background**

Run:
```bash
podman run -d --name fswe-notebook-smoke -p 8000:8000 \
  -e FSWE_TASK_NAME=notebook \
  -e FSWE_TASK_MODE=training \
  frontier-swe-notebook:latest
```
Expected: container id printed.

- [ ] **Step 10.2: Wait for /health to respond**

Run:
```bash
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "ready after ${i}s"; break
  fi
  sleep 1
done
curl -s http://localhost:8000/health
```
Expected: "ready after Ns" for some N < 30, then health JSON.

- [ ] **Step 10.3: Hit reset endpoint**

Run:
```bash
curl -s -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{}' | head -c 500
```
Expected: JSON observation with `phase=PLANNING` and a non-zero `time_remaining_s`.

- [ ] **Step 10.4: Confirm MCP transport is mounted**

Run:
```bash
curl -sI http://localhost:8000/tools/mcp | head -5
```
Expected: 2xx or 4xx header (confirming the path is served); any 404 means the mount failed.

- [ ] **Step 10.5: Capture container logs (local-only; artifacts/ is gitignored)**

Run:
```bash
mkdir -p artifacts/issue-4
podman logs fswe-notebook-smoke > artifacts/issue-4/smoke-container.log 2>&1
head -40 artifacts/issue-4/smoke-container.log
```
Expected: uvicorn startup messages, `Grader LLM config:` line, no tracebacks. Paste the interesting portion (first 40 lines + any error) into the subagent report so the controller can record it in the decision log.

- [ ] **Step 10.6: Stop the container**

Run:
```bash
podman rm -f fswe-notebook-smoke
```

- [ ] **Step 10.7: (No commit — `artifacts/` is gitignored.)**

---

## Task 11: Update progress tracker + decision-log evidence

**Files:**
- Modify: `notes/issues/issue-4-notebook-compression/progress-tracker.md`
- Modify: `notes/issues/issue-4-notebook-compression/decision-log.md`

- [ ] **Step 11.1: Update tracker statuses**

In `notes/issues/issue-4-notebook-compression/progress-tracker.md`, change the statuses of P5–P10 to `DONE` with evidence pointers, e.g.:

```markdown
| P5 | Visible-corpus split utility | P3 | DONE | `scripts/split_visible_corpus.py` + local smoke run |
| P6 | Core generalizations: l1_timeout_s + reward_json | P4 | DONE | tests/test_task_config.py, tests/test_l1_tests.py |
| P7 | Task config module + registry | P6 | DONE | `frontier_swe_env/tasks/notebook_compression.py` + tests |
| P8 | Dockerfile.notebook + gate checks | P3,P5,P6,P7 | DONE | image built; gates pass 3/3 in container |
| P9 | Verifier smoke (stub → fail) | P8 | DONE | reward.json status=fail as expected |
| P10 | Verifier smoke (trivial codec → ok) | P8 | DONE | geom_mean_ratio ≈ 0.X (recorded in container run) |
```

Leave P11–P12 as TODO (full pi episode + DoD closeout are out of scope for this plan — pending agent config + GPU/CPU pool).

Add to the Update Log section:

```markdown
- 2026-04-25: Completed P5–P10. End-to-end container smoke works; pi-driven full episode deferred pending agent credentials.
```

- [ ] **Step 11.2: Add post-implementation notes to decision log**

Append to `notes/issues/issue-4-notebook-compression/decision-log.md`:

```markdown
## D-011: Final Reward Anchors

- Date: 2026-04-25
- Decision: initial `R_max=1.0, R_min=0.15` anchors kept; trivial-zstd smoke produced geom_mean_ratio ≈ 0.3-0.5, mapping to ~0.6-0.8 normalized. Anchors tune-able in TestOutputRubric class constants without task-config churn.
- Why: placeholder values from the spec held up; no reason to change before observing real agent runs.

## D-012: Defer Full pi Episode

- Date: 2026-04-25
- Decision: implementation plan stops at container + OpenEnv smoke. Full pi-driven episode (plan/submit/advance with real LLM judge) requires FSWE_AGENT_* and FSWE_GRADER_* credentials and is deferred to a separate validation step.
- Why: the work in scope for issue #4 is onboarding the task; episode validation depends on external credentials and compute that aren't part of this branch.
```

- [ ] **Step 11.3: Commit**

```bash
git add notes/issues/issue-4-notebook-compression/progress-tracker.md notes/issues/issue-4-notebook-compression/decision-log.md
git commit -m "docs(issue-4): update progress tracker and decision log with implementation evidence"
```

---

## Final self-review checklist (runs at the end)

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `podman images frontier-swe-notebook:latest` — image exists
- [ ] `podman run --rm frontier-swe-notebook:latest bash /app/gate_checks.sh | tail -1` — `GATE_SCORE=3/3`
- [ ] `git log --oneline feat/issue-4-notebook-compression ^main` — commits are cleanly scoped per task
- [ ] DoD items A–F reviewed; mark each one DONE/TODO in `dod.md`
