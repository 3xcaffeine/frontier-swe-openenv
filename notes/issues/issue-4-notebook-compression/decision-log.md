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

## D-003: Keep Core Environment Stable

- Date: 2026-04-24
- Decision: avoid modifying core episode state machine unless notebook task contract requires it.
- Why: reduces regression risk across existing task(s).

## D-004: Dependency-First Execution

- Date: 2026-04-24
- Decision: execute in strict dependency order from `dependency-map.md`.
- Why: avoids premature wiring and repeated rework.

## D-005: Vendor Hidden Bundle In Repo

- Date: 2026-04-24
- Decision: include upstream `tests/hidden_test_set_bundle.zip` in local task scaffold.
- Why: ensures local verifier parity from day one.

## D-006: L1 Starts With Lighter Ratio Mode

- Date: 2026-04-24
- Decision: use a lightweight visible ratio-based L1 (`Total: N/M passed`) initially, then iterate toward stricter parity.
- Why: faster integration path while environment/task wiring is stabilized.

## D-007: Notebook Training Timeouts

- Date: 2026-04-24
- Decision: set notebook training episode timeout to 3600s and per-turn timeout to 600s.
- Why: task has longer fit/compress/decompress stages than postgres and needs longer command windows.

## D-008: Task Selection Via Environment Variables

- Date: 2026-04-24
- Decision: support `FSWE_TASK_NAME` and `FSWE_TASK_MODE` in environment initialization.
- Why: allows task-specific images to select configs without changing app wiring.

## Open Decisions

- Do we need a separate baseline runner script for notebook task, or can `scripts/run_baseline.py` be generalized?
