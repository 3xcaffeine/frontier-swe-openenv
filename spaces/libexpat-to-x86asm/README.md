---
title: Frontier SWE — libexpat to x86-64 Assembly
emoji: 🔧
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# Frontier SWE — libexpat to x86-64 Assembly

OpenEnv-shaped **FastAPI** service for the **libexpat-to-x86asm** task: reimplement **libexpat 2.6.4** in **x86-64 assembly**, producing `/app/asm-port/libexpat.so` with the **expat C ABI**. The verifier compares against reference C libexpat, runs upstream tests and benchmarks, and writes `/logs/verifier/reward.json` (correctness and performance blend; hard fail to `0.0` on anti-cheat or missing `.so`).

## The task in depth

The agent’s deliverable is a **shared library** built from **`.s` / `.asm`** sources under **`/app/asm-port/`**, exporting symbols such as **`XML_ParserCreate`** so the upstream **expat** test suite can link against it. There is **no C compiler** in the agent environment; the verifier may compile reference C code for comparison. Scoring combines **weighted test pass rates** with **benchmark timing ratios** (reference time vs agent time) into a single **`score`** in **`reward.json`**, with explicit anti-cheat checks (no `dlopen` of system libexpat, no smuggled C core files, etc.). The server treats that file in **`reward_json_score`** mode with anchors **`(0.0, 1.0)`**.

## How this maps to the monorepo

- **`tasks/libexpat-to-x86asm/`** — Instructions, encrypted or staged toolchain bundles as designed, **`tests/`** with **`test.sh`**, **`compute_reward.py`**, and benchmark XML generators.
- **`frontier_swe_env/tasks/libexpat_to_x86asm.py`** — **`TaskConfig`**: workspace **`/app/asm-port`**, gate script, verifier command, JSON path and anchors, CPU/memory hints, and judge context strings.
- **`spaces/libexpat-to-x86asm/`** — This Space and manifest.

See [**Task assets and runtime configuration**](https://github.com/3xcaffeine/frontier-swe-openenv#task-assets-and-runtime-configuration) in the root README.

## Features

- **Assembly port workspace**: `/app/asm-port` with staged toolchain and bundles (see gate checks in manifest).
- **Structured L1**: Normalised score from `reward.json`; gates for writable workspace, headers, `nasm` / `as` / `ld`, and staged artifacts.
- **LLM rubric layers**: L2 code review and L3 plan review when grader env vars are set.
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
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-libexpat-to-x86asm:latest
```

This task is CPU- and memory-sensitive; the manifest requests **4 CPUs** and **8192 MiB** where the platform allows.

```bash
docker run --rm -p 8000:8000 \
  -e FSWE_GRADER_MODEL=... \
  -e FSWE_GRADER_API_URL=... \
  -e FSWE_GRADER_API_KEY=... \
  ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-libexpat-to-x86asm:latest
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
        await client.step(FrontierSweAction(message="Continue the assembly port."))
    finally:
        await client.close()


asyncio.run(main())
```

## Task manifest

[`openenv.yaml`](openenv.yaml) — episode timeout, L1 timeout, reward field anchors, rubric layers, metrics. Task sources: `tasks/libexpat-to-x86asm/`.

## Deployment

- **Image**: `ghcr.io/3xcaffeine/frontier-swe-openenv/frontier-swe-libexpat-to-x86asm:latest`
- **Source**: [3xcaffeine/frontier-swe-openenv](https://github.com/3xcaffeine/frontier-swe-openenv)
- **Sync**: HF Space updated from `main` after successful GHCR build.

Benchmark context: [FrontierSWE — libexpat to x86-64 assembly](https://www.frontierswe.com/libexpat-to-x86asm).
