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
| P4 | Write design spec (brainstorming output) | P2 | DONE | `notes/specs/2026-04-24-notebook-compression-design.md` |
| P5 | Visible-corpus split utility (`scripts/split_visible_corpus.py`) | P3 | DONE | smoke run produced 60/80 files + manifest |
| P6 | Core generalizations: `TaskConfig.l1_timeout_s`; `score_mode="reward_json"` | P4 | DONE | commits `7a4a1b8` + `c201377` |
| P7 | Task config module + registry registration | P6 | DONE | `notebook` and `notebook-compression` resolve via `get_task_config` |
| P8 | Dockerfile.notebook + gate checks script | P3,P5,P6,P7 | DONE | `openenv-base:latest` (1.45GB) + `frontier-swe-notebook:latest` (1.96GB) built; gate 3/3 in-container |
| P9 | Verifier smoke test in container (stub → fail) | P8 | DONE | upstream `run` stub → `status=fail, reason="fit stage failed: exit code 1"` |
| P10 | Verifier smoke test with trivial zstd codec (round-trip ok) | P8 | DONE | zstd -19 wrapper → `status=ok, geom_mean_ratio=0.326335, compression_score=0.559899`, round-trip OK on 80 notebooks (fit 0.017s, compress 27.5s, decompress 0.32s) |
| P11 | End-to-end OpenEnv server + notebook-task selection | P10 | DONE | server boots 3s; `/health` OK; `FSWE_TASK_NAME=notebook` resolves correctly (`time_remaining_s=3600` confirms notebook config, not PG fallback); `/reset` returns `PLANNING`; pi harness starts with correct agent/grader config |
| P12 | Full pi episode with real agent/grader (plan/submit/advance) | P11 | DEFERRED | needs real `FSWE_AGENT_API_KEY`/`FSWE_GRADER_API_KEY` + endpoints; see D-012 |
| P13 | DoD verification and docs closeout | P12 | TODO | |

## Current Focus

Next execution target: P12 when agent/grader credentials are available; then P13 DoD signoff.

## Update Log

- 2026-04-24: Reset tracker after discovering prior entries did not match branch state. Vendored upstream task folder verbatim (P3 DONE). Spec written (P4 DONE).
- 2026-04-25: Completed P5–P7; narrowed `.gitignore` (`/tests/`, `/docs/`, with negations for nested task/notes assets) so vendored verifier artifacts actually track. Spec + plan moved from `docs/superpowers/` to `notes/specs/` and `notes/plans/` per project convention. Built both images (base 1.45GB, notebook 1.96GB). Verifier smoke confirms hard-fail path (stub → `status=fail`) and success path (trivial zstd → `geom_mean_ratio=0.326335, round-trip OK`). OpenEnv server smoke confirms task-env-var selection picks up notebook config (`time_remaining_s=3600`), pi harness starts correctly. P8–P11 DONE. P12 requires external credentials and is deferred.
- 2026-04-24 (earlier, superseded): Prior tracker claimed P3–P5 complete; no evidence on branch. Entries reset per user confirmation.
