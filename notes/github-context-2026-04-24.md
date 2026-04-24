# GitHub Context Snapshot (2026-04-24)

Repository: `3xcaffeine/frontier-swe-openenv`
Default branch: `main`

## Open Issues (Observed)

- #1 megathread for env
  - Contains minimum submission expectations (OpenEnv usage, training evidence, HF Space, README quality, linked artifacts).
  - Priority guidance comment indicates ordering: #8 > #7 > #6 (note: #8 is now closed).
- #4 implement notebook-compression from fswe
- #5 impl pyright-type-checking-optimization task
- #6 envs need to be deployed on hf spaces
- #7 training (needs research/discussion)
- #9 reward hacking concerns

## Recently Closed Issues

- #8 decouple the env from the task
- #3 implement postgres-sqlite-wire-adapter and run one episode E2E
- #2 verify harness and pi adapter in real dockerized env

## Pull Request Snapshot

Open PRs: none at snapshot time.

Recently merged:
- PR #10 refactor: task-agnostic env architecture
  - Introduced task registry model.
  - Parameterized L2/L3 prompting and L1 scoring parser behavior.
  - Clarified dual MCP transport pattern.
  - Reduced PG-specific coupling in core env code.

## Interpretation for Planning

- Core technical foundation is in place and recently refactored for extensibility.
- The next layer of work appears to be:
  1. Additional tasks beyond PG.
  2. Training workflow maturation.
  3. Deployment polish (HF Spaces + documentation).
  4. Reward-hacking hardening.
