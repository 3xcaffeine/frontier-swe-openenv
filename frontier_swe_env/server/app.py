# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Frontier Swe Env Environment.

Serves two things on the same port:
1. OpenEnv Gym-style API at /, /reset, /step, /ws, /mcp (POST-only JSON-RPC)
2. FastMCP native Streamable HTTP at /tools/mcp (POST + GET/SSE)

Pi-mcp-adapter connects to (2) because it requires Streamable HTTP transport
(the POST-only /mcp from OpenEnv returns 405 on the GET SSE probe).
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import FrontierSweAction, FrontierSweObservation
    from .frontier_swe_env_environment import FrontierSweEnvironment
except ImportError:
    from models import FrontierSweAction, FrontierSweObservation
    from server.frontier_swe_env_environment import FrontierSweEnvironment

from fastmcp import FastMCP

# Shared MCP server for pi-mcp-adapter (Streamable HTTP transport)
# This FastMCP instance is mounted at /tools so pi can connect via
# Streamable HTTP at http://localhost:8000/tools/mcp.
#
# The tools delegate to a mutable _active_env reference that is set
# by FrontierSweEnvironment on reset().  Since max_concurrent_envs=1,
# there is exactly one active environment at a time.

_active_env = None  # set by the environment on reset()

pi_mcp = FastMCP("frontier-swe-tools")


@pi_mcp.tool
async def submit_plan(subtasks: list[dict]) -> dict:
    """Propose a subtask plan for the episode."""
    if _active_env is None:
        return {"error": "Environment not initialised. Call reset() first."}
    return await _active_env.submit_plan_payload(subtasks)


@pi_mcp.tool
async def submit_subtask(subtask_id: str) -> dict:
    """Submit the current subtask for L1+L2 scoring."""
    if _active_env is None:
        return {"error": "Environment not initialised. Call reset() first."}
    return await _active_env.submit_subtask_payload(subtask_id)


@pi_mcp.tool
def get_status() -> dict:
    """Get current episode status snapshot."""
    if _active_env is None:
        return {"error": "Environment not initialised. Call reset() first."}
    return _active_env.get_status_payload()


@pi_mcp.tool
def advance() -> dict:
    """Freeze current subtask score and move to the next subtask."""
    if _active_env is None:
        return {"error": "Environment not initialised. Call reset() first."}
    return _active_env.advance_payload()


def set_active_env(env):
    """Called by FrontierSweEnvironment.reset() to register itself."""
    global _active_env
    _active_env = env


# OpenEnv app
app = create_app(
    FrontierSweEnvironment,
    FrontierSweAction,
    FrontierSweObservation,
    env_name="frontier_swe_env",
    max_concurrent_envs=1,
)

# Mount FastMCP's native Streamable HTTP app at /tools
# This gives us POST + GET (SSE) at /tools/mcp — which pi-mcp-adapter needs.
# We must wire the lifespan so FastMCP's session manager initialises.
_mcp_http_app = pi_mcp.http_app()

from contextlib import asynccontextmanager  # noqa: E402

_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _combined_lifespan(a):
    async with _mcp_http_app.router.lifespan_context(_mcp_http_app):
        if _original_lifespan is not None:
            async with _original_lifespan(a):
                yield
        else:
            yield


app.router.lifespan_context = _combined_lifespan
app.mount("/tools", _mcp_http_app)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
