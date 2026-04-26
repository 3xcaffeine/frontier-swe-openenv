---
title: Frontier SWE — Postgres SQLite Wire Adapter
emoji: 🐘
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
---

# Frontier SWE — Postgres / SQLite Wire Adapter

OpenEnv-shaped **FastAPI** service for the **postgres-sqlite-wire-adapter** task: implement a PostgreSQL wire-protocol-compatible server in **Zig** backed by **SQLite**, with gate checks, a graded test runner, and composite rubric scoring.

## The task in depth

The workspace is **`/app/postgres-sqlite`**. The agent grows a Zig project that mimics enough **`postgres` / `pg_ctl` / `initdb`** behaviour and the **Frontend/Backend protocol** so that real PostgreSQL clients can connect and run a large scripted compatibility matrix. **L1** is driven by a visible test script whose stdout looks like **`Total: N/M passed`**; the shared rubric parses that as a pass ratio (see `l1_score_mode="ratio"`). Hidden or stronger checks can live alongside the same task pack under [`tasks/postgres-sqlite-wire-adapter/tests/`](https://github.com/3xcaffeine/frontier-swe-openenv/tree/main/tasks/postgres-sqlite-wire-adapter/tests). Unlike the JSON-heavy tasks, there is no requirement for `reward.json` unless you extend the verifier that way.

## How this maps to the monorepo

- **`tasks/postgres-sqlite-wire-adapter/`** — Stubs, instructions, **`pg_compat_test.sh`**, smoke tests, and hidden verifier assets copied into the image.
- **`frontier_swe_env/tasks/pg.py`** — **`TaskConfig`** for this task: Zig workspace path, **`bash /app/gate_checks.sh`**, **`PG_PORT=55432 bash /app/pg_compat_test.sh`** as the L1 command, regex pattern for totals, timeouts, and judge-facing descriptions.
- **`spaces/postgres/`** — Space wrapper and **`openenv.yaml`** aligned with the same episode.

More detail: [**Task assets and runtime configuration**](https://github.com/3xcaffeine/frontier-swe-openenv#task-assets-and-runtime-configuration) in the root README.

## Features

- **Systems programming focus**: Zig workspace under `/app/postgres-sqlite`, verifier and hidden tests shipped in the image.
- **L1 scoring**: Regex ratio over test runner output (`Total: N/M passed`) plus gate script.
- **LLM-assisted layers**: L2 code review and L3 plan review when grader env vars are set.
- **MCP tools**: `submit_plan`, `submit_subtask`, `get_status`, `advance`.

## HTTP API

| Endpoint | Notes |
| --- | --- |
| `GET /health` | Liveness. |
| `POST /reset`, `POST /step`, `GET /state` | OpenEnv Gym-style control. |
| `POST /mcp` | OpenEnv JSON-RPC MCP. |
| `/tools/mcp` | FastMCP Streamable HTTP. |

## Quick start (Docker)

```bash
docker run --rm -p 8000:8000 \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-postgres:latest
```

With grader API for full rubric:

```bash
docker run --rm -p 8000:8000 \
  -e FSWE_GRADER_MODEL=... \
  -e FSWE_GRADER_API_URL=... \
  -e FSWE_GRADER_API_KEY=... \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-postgres:latest
```

## Baseline script

The repo ships [`scripts/run_baseline.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/run_baseline.py) for a full WebSocket episode against a running container (defaults to `http://localhost:8000`).

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
        await client.step(FrontierSweAction(message="Implement the next milestone."))
    finally:
        await client.close()


asyncio.run(main())
```

## Task manifest

[`openenv.yaml`](openenv.yaml) — workspace, timeouts, rubric layers, and metrics. Task sources: `tasks/postgres-sqlite-wire-adapter/`.

## Deployment

- **Image**: `ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-postgres:latest`
- **Source**: [3xcaffeine/frontier-swe-openenv](https://github.com/3xcaffeine/frontier-swe-openenv)
- **Sync**: HF Space payload is assembled from this directory on `main` after GHCR builds.

Benchmark context: [FrontierSWE — PostgreSQL on SQLite](https://www.frontierswe.com/postgres-sqlite-wire-adapter).
