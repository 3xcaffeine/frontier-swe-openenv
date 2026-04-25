"""Dependent-type-checker task configuration.

Agent ships a Rust binary at /app/type-checker/target/release/type-checker
that takes one or more S-expression files and exits 0 iff every top-level
command type-checks under a Martin-Löf-style dependently-typed language
with Pi/Sigma (eta), inductive families with parameters/indices,
auto-generated recursors, strict positivity, and bidirectional checking.

L1 scoring reads a structured reward.json with the upstream verifier's
``score`` field (= geometric mean of median speedups vs reference impl,
gated on accept_rate >= 0.99 and reject_rate >= 0.95).
"""

from __future__ import annotations

from pathlib import Path

from ..task_config import TaskConfig


DTC_TRAINING_INSTRUCTION = """
# Dependent Type Checker

Your workspace is `/app/type-checker/`. A scaffolded Cargo project is already
in place — `Cargo.toml` and a stub `src/main.rs`. You must implement a Rust
binary that type-checks files in a dependently-typed language.

## Contract

The verifier builds your project with:

```
cd /app/type-checker && cargo build --release
```

It then invokes the binary `target/release/type-checker` (or `type_checker`
or `dependent-type-checker`) with one or more `.sexp` file paths as args.

- **Exit 0** if every top-level command in every file type-checks.
- **Exit 1** if any command fails.
- Diagnostics may go to **stderr**. **Stdout must be empty.** Only exit codes
  are scored.

The language supports cumulative universes `Type i`, dependent functions
`(Pi (x : A) B)` / `(lam x e)` / `(app f a)` with eta, dependent pairs
`(Sigma ...)` / `(pair ...)` / `(fst/snd)` with eta, `(let (x : A) v body)`
with delta, type annotations `(ann e A)`, and general inductive types with
parameters/indices, auto-generated recursors with iota, and strict-positivity
checking. See `/app/instruction.md` for the full operational spec.

## Examples

`/app/examples/` contains four `.sexp` files (identity, nat_arith, vectors,
invalid_example) you can use to sanity-check your implementation locally:

```
/app/type-checker/target/release/type-checker /app/examples/identity.sexp
echo $?   # expect 0
/app/type-checker/target/release/type-checker /app/examples/invalid_example.sexp
echo $?   # expect 1
```

## Scoring

The verifier (`bash /opt/verifier/test.sh`) runs in three stages:

1. **Source scan** — your `src/` is regex-checked for references to verifier
   internals (`/tests/|compute_reward|reference_impl|/logs/verifier|...`).
   Any match → hard-fail with `reason="source_scan_match"`.
2. **Reference-copy check** — sha256 of your `src/main.rs` vs the reference
   implementation. Match → hard-fail with `reason="reference_copy"`.
3. **Build** — `cargo build --release` must succeed.
4. **Correctness** — `accept_rate >= 0.99` AND `reject_rate >= 0.95` over
   174 accept-corpus + 81 reject-corpus files. Either fails → hard-fail
   with `reason="correctness_gate_failed"`, score = 0.0.
5. **Speed** — geometric mean of `ref_latency / cand_latency` across 3
   workloads (heavy_norm, inductive_elim, small_lemmas), measured via
   3 warmup + 15 measurement ABBA pairs per workload, capped at 100x per
   workload.

Final `reward.json` carries `score` (the speedup-based number) plus
`subscores` (`accept_rate`, `reject_rate`, `throughput_speedup`) and
`additional_data.reason` if hard-failed.

## Workflow

1. **Read** the full spec at `/app/instruction.md`.
2. **Plan** — call `submit_plan` with one subtask covering the whole
   implementation (correctness first, speed after).
3. **Implement** — edit `/app/type-checker/src/main.rs` and any helper
   modules under `/app/type-checker/src/`. You may add dependencies to
   `Cargo.toml` but the build runs with `--offline`-style sandboxing
   (no internet at verifier time).
4. **Submit** — call `submit_subtask` with the subtask id. The verifier
   runs the full pipeline and returns a normalized reward.

**Remember:** correctness is gated. If the gate fails, score is 0 regardless
of speed. Aim for correctness first, then optimize.
""".strip()


def _load_upstream_instruction() -> str:
    """Return the upstream instruction.md if present, else the training text."""
    upstream = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "dependent-type-checker"
        / "instruction.md"
    )
    if upstream.is_file():
        return upstream.read_text()
    return DTC_TRAINING_INSTRUCTION


def dtc_training_config() -> TaskConfig:
    return TaskConfig(
        task_name="dependent-type-checker",
        docker_image="frontier-swe-dependent-type-checker:latest",
        instruction=DTC_TRAINING_INSTRUCTION,
        workspace_dir="/app/type-checker",
        build_command="cd /app/type-checker && cargo build --release",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="bash /opt/verifier/test.sh",
        # No regex-style total; the verifier writes reward.json.
        visible_test_total=1,
        l1_score_mode="reward_json_score",
        l1_timeout_s=600.0,
        reward_json_path="/logs/verifier/reward.json",
        reward_json_score_field="score",
        # score=1.0 means matches reference impl. Anchor at (0.0, 2.0) so
        # 1x = 0.5 normalized, 2x = 1.0. Tune after observing real agent runs.
        reward_json_score_anchors=(0.0, 2.0),
        reward_json_score_higher_is_better=True,
        gate_threshold=0.67,
        max_subtasks=1,
        max_attempts_per_subtask=3,
        episode_timeout_s=3600.0,
        per_turn_timeout_s=600.0,
        task_description=(
            "Implement a Rust binary that type-checks files in a Martin-Löf-"
            "style dependently-typed language. Scored by geometric mean of "
            "speedup vs the reference implementation, gated on correctness."
        ),
        task_domain="programming languages / type theory",
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh. Reads /logs/verifier/reward.json. "
            "additional_data.reason set scores 0.0. Otherwise score is normalized "
            "via score / 2.0 clamped to [0, 1]. Subscores include accept_rate, "
            "reject_rate, throughput_speedup."
        ),
        cpus=8,
        memory_mb=32768,
    )


def dtc_demo_config() -> TaskConfig:
    return TaskConfig(
        task_name="dependent-type-checker",
        docker_image="frontier-swe-dependent-type-checker:latest",
        instruction=_load_upstream_instruction(),
        workspace_dir="/app/type-checker",
        build_command="cd /app/type-checker && cargo build --release",
        gate_script_path="/app/gate_checks.sh",
        visible_test_command="bash /opt/verifier/test.sh",
        visible_test_total=1,
        l1_score_mode="reward_json_score",
        l1_timeout_s=1200.0,
        reward_json_path="/logs/verifier/reward.json",
        reward_json_score_field="score",
        reward_json_score_anchors=(0.0, 2.0),
        reward_json_score_higher_is_better=True,
        gate_threshold=0.67,
        max_subtasks=1,
        max_attempts_per_subtask=5,
        episode_timeout_s=7200.0,
        per_turn_timeout_s=900.0,
        task_description=(
            "Implement a fast and correct dependent type checker (demo mode: "
            "longer budgets and more attempts)."
        ),
        task_domain="programming languages / type theory",
        scoring_context=(
            "L1 runs bash /opt/verifier/test.sh. Reads /logs/verifier/reward.json. "
            "additional_data.reason set scores 0.0. Otherwise score / 2.0 normalized."
        ),
        cpus=8,
        memory_mb=32768,
    )
