---
title: Frontier SWE — Notebook Compression
emoji: 📓
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Frontier SWE — Notebook Compression

OpenEnv-shaped **FastAPI** service for the **notebook-compression** task: build a fit / compress / decompress pipeline for Jupyter notebooks inside a Linux workspace, with multi-layer rubric scoring and a structured `reward.json` written by the verifier.

## The task in depth

The agent needs to ship an executable **`/app/run`** with three subcommands: **`fit`** (train or build artifacts from a **visible** corpus only), **`compress`**, and **`decompress`**. At scoring time the agent does not see the hidden corpus: the verifier checks **byte-for-byte** recovery of every notebook file. Compression quality is summarised as a geometric mean of size ratios; hard failures (round-trip mismatch, crashes, invalid `reward.json` status) collapse the L1 signal to zero. That logic lives in the repo under [`tasks/notebook-compression/tests/`](https://github.com/3xcaffeine/frontier-swe-openenv/tree/main/tasks/notebook-compression/tests) (shell driver plus [`compute_reward.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/tasks/notebook-compression/tests/compute_reward.py)), which writes **`/logs/verifier/reward.json`** for the server to read.

## How this maps to the monorepo

- **`tasks/notebook-compression/`** — Authoritative instructions, verifier, and reward computation; copied into the image (for example **`/opt/verifier/test.sh`** and data mounts).
- **`frontier_swe_env/tasks/notebook_compression.py`** — Registers **`TaskConfig`** with `l1_score_mode="reward_json"`, the container test command, long L1 timeouts, gate path, and prose for L2/L3 judges. The running server selects it when `FSWE_TASK_NAME` is `notebook` or `notebook-compression` (see [`__init__.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/frontier_swe_env/tasks/__init__.py)).
- **`spaces/notebook/`** — This Space: thin Dockerfile, this README, and **`openenv.yaml`** describing the same episode for Hugging Face and external tooling.

For the full picture of how task directories and Python configs interact, see the root README section [**Task assets and runtime configuration**](https://github.com/3xcaffeine/frontier-swe-openenv#task-assets-and-runtime-configuration).

## Features

- **Long-horizon SWE**: Plan subtasks, edit code under the configured workspace, submit for scoring.
- **Composite rubric**: Shell gate checks → structured L1 from `/logs/verifier/reward.json` → optional LLM code review (L2) and plan review (L3) → weighted episode reward.
- **MCP tools**: `submit_plan`, `submit_subtask`, `get_status`, `advance` (same contract as other Frontier SWE Spaces).
- **Dual MCP transports**: OpenEnv `POST /mcp` and Streamable HTTP `/tools/mcp` for adapters.

## HTTP API

| Endpoint | Notes |
| --- | --- |
| `GET /health` | Liveness for orchestration and HF health checks. |
| `POST /reset`, `POST /step`, `GET /state` | OpenEnv Gym-style control. |
| `POST /mcp` | OpenEnv JSON-RPC MCP. |
| `/tools/mcp` | FastMCP Streamable HTTP (POST + GET/SSE). |

## Quick start (Docker)

```bash
docker run --rm -p 8000:8000 \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-notebook:latest
```

Optional grader configuration for LLM rubric layers:

```bash
docker run --rm -p 8000:8000 \
  -e FSWE_GRADER_MODEL=... \
  -e FSWE_GRADER_API_URL=... \
  -e FSWE_GRADER_API_KEY=... \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-notebook:latest
```

## Python client (host)

From the [source repository](https://github.com/3xcaffeine/frontier-swe-openenv), with dependencies installed:

```python
import asyncio
from frontier_swe_env.client import FrontierSweEnv
from frontier_swe_env.models import FrontierSweAction


async def main():
    client = FrontierSweEnv(base_url="http://localhost:8000")
    await client.connect()
    try:
        await client.reset()
        await client.step(FrontierSweAction(message="Continue the task."))
    finally:
        await client.close()


asyncio.run(main())
```

## Task manifest

OpenEnv metadata for judges and tooling: [`openenv.yaml`](openenv.yaml) in this Space (mirrors `spaces/notebook/openenv.yaml` in the GitHub repo). Task sources: `tasks/notebook-compression/`.

## Deployment

- **Image**: `ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-notebook:latest`
- **Source**: [3xcaffeine/frontier-swe-openenv](https://github.com/3xcaffeine/frontier-swe-openenv)
- **Sync**: Pushed from `main` by the repository’s HF Spaces sync workflow after GHCR builds succeed.

Benchmark context: [FrontierSWE — Notebook compression](https://www.frontierswe.com/notebook-compression).
