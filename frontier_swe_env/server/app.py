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

import logging

# Configure application logging so our loggers output alongside uvicorn.
# uvicorn only configures its own loggers; without this, all logger.info()
# calls in frontier_swe_env.* go nowhere.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

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
    logger.info("MCP submit_plan called with %d subtasks", len(subtasks) if subtasks else 0)
    if _active_env is None:
        logger.error("submit_plan: _active_env is None!")
        return {"error": "Environment not initialised. Call reset() first."}
    try:
        result = await _active_env.submit_plan_payload(subtasks)
        logger.info("submit_plan result: phase=%s score=%s", result.get("phase"), result.get("plan_score"))
        return result
    except Exception:
        logger.exception("submit_plan EXCEPTION")
        return {"error": "Internal error in submit_plan. Check server logs."}


@pi_mcp.tool
async def submit_subtask(subtask_id: str) -> dict:
    """Submit the current subtask for L1+L2 scoring."""
    logger.info("MCP submit_subtask called: %s", subtask_id)
    if _active_env is None:
        logger.error("submit_subtask: _active_env is None!")
        return {"error": "Environment not initialised. Call reset() first."}
    try:
        result = await _active_env.submit_subtask_payload(subtask_id)
        logger.info("submit_subtask result: score=%s best=%s remaining=%s",
                    result.get("score"), result.get("best_score"), result.get("attempts_remaining"))
        return result
    except Exception:
        logger.exception("submit_subtask EXCEPTION")
        return {"error": "Internal error in submit_subtask. Check server logs."}


@pi_mcp.tool
def get_status() -> dict:
    """Get current episode status snapshot."""
    if _active_env is None:
        return {"error": "Environment not initialised. Call reset() first."}
    return _active_env.get_status_payload()


@pi_mcp.tool
def advance() -> dict:
    """Freeze current subtask score and move to the next subtask."""
    logger.info("MCP advance called")
    if _active_env is None:
        logger.error("advance: _active_env is None!")
        return {"error": "Environment not initialised. Call reset() first."}
    try:
        result = _active_env.advance_payload()
        logger.info("advance result: next=%s done=%s", result.get("next_subtask_id"), result.get("episode_done"))
        return result
    except Exception:
        logger.exception("advance EXCEPTION")
        return {"error": "Internal error in advance. Check server logs."}


def set_active_env(env):
    """Called by FrontierSweEnvironment.reset() to register itself."""
    global _active_env
    _active_env = env
    logger.info("set_active_env: registered %s (phase=%s)", type(env).__name__, getattr(env, 'episode_state', {}))


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
