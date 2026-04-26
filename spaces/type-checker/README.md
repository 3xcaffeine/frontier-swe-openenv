---
title: Frontier SWE — Dependent Type Checker
emoji: 🧮
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Frontier SWE — Dependent Type Checker

OpenEnv-shaped **FastAPI** service for the **dependent-type-checker** task: implement a Martin-Löf-style dependently typed language **type checker** in **Rust** (`cargo build --release`), scored on correctness gates and speedup versus a reference implementation via `/logs/verifier/reward.json`.

## The task in depth

The agent edits **`/app/type-checker/`** (Cargo project) and must produce a release binary that type-checks `.sexp` programs for a language with dependent functions, inductive families, cumulativity, and related features spelled out in **`instruction.md`**. The verifier (**`bash /opt/verifier/test.sh`**) enforces anti-cheat rules, checks accept/reject corpus rates, then measures speedups vs a reference implementation on fixed workloads. It writes **`/logs/verifier/reward.json`** with a numeric **`score`** and optional **`additional_data.reason`** on hard fail. Python config uses **`l1_score_mode="reward_json_score"`** with anchors **`(0.0, 2.0)`** so the server normalises that scalar into the shared \([0,1]\) L1 channel.

## How this maps to the monorepo

- **`tasks/dependent-type-checker/`** — Full formal spec, corpora, reference implementation pieces, and verifier scripts under **`tests/`**.
- **`frontier_swe_env/tasks/dependent_type_checker.py`** — Registers **`TaskConfig`** (`dependent-type-checker` / alias `type-checker`), build command, verifier timeout, JSON field names, and training vs demo instruction loading (demo can pull [`instruction.md`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/tasks/dependent-type-checker/instruction.md) from the repo when present on the host).
- **`spaces/type-checker/`** — This Space; GHCR image name uses **`frontier-swe-dependent-type-checker`**.

Architecture overview: [**Task assets and runtime configuration**](https://github.com/3xcaffeine/frontier-swe-openenv#task-assets-and-runtime-configuration).

## Features

- **Rust workspace**: `/app/type-checker` with release binary expected by the verifier.
- **Structured L1**: Score from `reward.json` (normalised with configured anchors, hard-fail signals documented in manifest).
- **Gate checks**: Workspace, `Cargo.toml`, toolchain, and successful release build.
- **MCP tools**: `submit_plan`, `submit_subtask`, `get_status`, `advance`.

## HTTP API

| Endpoint | Notes |
| --- | --- |
| `GET /health` | Liveness. |
| `POST /reset`, `POST /step`, `GET /state` | OpenEnv Gym-style control. |
| `POST /mcp` | OpenEnv JSON-RPC MCP. |
| `/tools/mcp` | FastMCP Streamable HTTP. |

## Quick start (Docker)

The GHCR image name uses `dependent-type-checker` (the workflow task id), while this Hugging Face Space repo id uses `type-checker`.

```bash
docker run --rm -p 8000:8000 \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-dependent-type-checker:latest
```

With grader API:

```bash
docker run --rm -p 8000:8000 \
  -e FSWE_GRADER_MODEL=... \
  -e FSWE_GRADER_API_URL=... \
  -e FSWE_GRADER_API_KEY=... \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-dependent-type-checker:latest
```

## Python client (host)

```python
import asyncio
from frontier_swe_env.client import FrontierSweEnv
from frontier_swe_env.models import FrontierSweAction


async def main():
    client = FrontierSweEnv(base_url="http://localhost:8000")
    await client.connect()
    try:
        await client.reset()
        await client.step(FrontierSweAction(message="Work on the type checker."))
    finally:
        await client.close()


asyncio.run(main())
```

## Task manifest

[`openenv.yaml`](openenv.yaml) — build command, L1 timeouts, reward anchors, rubric. Task sources: `tasks/dependent-type-checker/`.

## Deployment

- **Image**: `ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-dependent-type-checker:latest`
- **Source**: [3xcaffeine/frontier-swe-openenv](https://github.com/3xcaffeine/frontier-swe-openenv)
- **Sync**: Deployed from `main` via the repository HF Spaces workflow.

Benchmark context: [FrontierSWE — Dependent type checker](https://www.frontierswe.com/dependent-type-checker).
