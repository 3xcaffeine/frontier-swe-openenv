# Progress Tracker: Issue #4 Notebook Compression

Reset 2026-04-24 after discovering the prior tracker described aspirational state. Execution starts fresh on this branch; nothing claimed DONE below without on-branch evidence.

## Status Legend

- `DONE`: finished with evidence
- `IN_PROGRESS`: currently active
- `BLOCKED`: waiting on dependency
- `TODO`: not started

## Tracker

| ID | Work Item | Dependency | Status | Evidence / Notes |
|---|---|---|---|---|
| P0 | Gather task requirements from website and upstream | none | DONE | `research-notes.md`; upstream tree inspected at pinned commit |
| P1 | Pin upstream notebook-compression reference | P0 | DONE | commit `55d103355bf0bfffb6b47781733e817f9dc65bb3` |
| P2 | Draft dependency map, DoD, implementation plan | P0,P1 | DONE | `dependency-map.md`, `implementation-plan.md`, `dod.md` |
| P3 | Vendor upstream `tasks/notebook-compression/` verbatim | P2 | DONE | `tasks/notebook-compression/` (70 MB incl. hidden bundle) |
| P4 | Write design spec (brainstorming output) | P2 | DONE | `docs/superpowers/specs/2026-04-24-notebook-compression-design.md` |
| P5 | Visible-corpus split utility (`scripts/split_visible_corpus.py`) | P3 | TODO | |
| P6 | Core generalizations: `TaskConfig.l1_timeout_s`; `score_mode="reward_json"` | P4 | TODO | touches `task_config.py`, `rubrics/l1_tests.py`, `server/frontier_swe_env_environment.py` |
| P7 | Task config module + registry registration | P6 | TODO | `frontier_swe_env/tasks/notebook_compression.py` |
| P8 | Dockerfile.notebook + gate checks script | P3,P5,P6,P7 | TODO | `docker/Dockerfile.notebook`, `scripts/notebook_gate_checks.sh` |
| P9 | Verifier smoke test in container (stub → fail) | P8 | TODO | expect `status=fail` |
| P10 | Verifier smoke test with trivial zstd codec (round-trip ok) | P8 | TODO | expect `status=ok` with non-zero score |
| P11 | End-to-end episode (plan/submit/advance) with pi | P10 | TODO | log + reward artifacts under `artifacts/issue-4/` |
| P12 | DoD verification and docs closeout | P11 | TODO | |

## Current Focus

Next execution target: P5 (visible-corpus splitter) + P6 (core rubric generalizations) in parallel since they are independent.

## Update Log

- 2026-04-24: Reset tracker after discovering prior entries did not match branch state. Vendored upstream task folder verbatim (P3 DONE). Spec written (P4 DONE).
- 2026-04-24 (earlier, superseded): Prior tracker claimed P3–P5 complete; no evidence on branch. Entries reset per user confirmation.
