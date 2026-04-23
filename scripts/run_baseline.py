#!/usr/bin/env python3
"""
Run a single baseline episode of the FrontierSWE PostgreSQL task.

This script runs on the HOST and connects to the environment container
over WebSocket. The container must already be running.

Usage:
    # 1. Start the container
    docker run -d --name fswe-baseline -p 8000:8000 \\
      -e FSWE_AGENT_MODEL=qwen-3.5-27b \\
      -e FSWE_AGENT_PROVIDER=openai \\
      -e FSWE_AGENT_API_URL=https://api.siemens.com/llm/v1 \\
      -e FSWE_AGENT_API_KEY=... \\
      -e FSWE_GRADER_MODEL=glm-5 \\
      -e FSWE_GRADER_API_URL=https://api.siemens.com/llm/v1 \\
      -e FSWE_GRADER_API_KEY=... \\
      frontier-swe-pg:latest

    # 2. Run the baseline
    python scripts/run_baseline.py

    # 3. Cleanup
    docker rm -f fswe-baseline

Options:
    --url URL           Server URL (default: http://localhost:8000)
    --max-turns N       Max step() calls (default: 100)
    --timeout SECS      WebSocket message timeout (default: 600)
    --output PATH       Write result JSON to file (default: baseline_result.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from frontier_swe_env.client import FrontierSweEnv
from frontier_swe_env.models import FrontierSweAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline")

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    base_url: str = "http://localhost:8000",
    max_turns: int = 100,
    message_timeout_s: float = 600.0,
    output_path: str = "baseline_result.json",
) -> dict:
    """Connect to the container and run one full episode."""

    logger.info("=" * 60)
    logger.info("FrontierSWE Baseline — PostgreSQL Wire Adapter")
    logger.info("=" * 60)
    logger.info("Server:     %s", base_url)
    logger.info("Max turns:  %d", max_turns)
    logger.info("Msg timeout:%ds", message_timeout_s)
    logger.info("=" * 60)

    client = FrontierSweEnv(
        base_url=base_url,
        message_timeout_s=message_timeout_s,
    )

    t0 = time.time()

    try:
        # Connect WebSocket
        logger.info("Connecting to %s ...", base_url)
        await client.connect()
        logger.info("Connected.")

        # Reset — starts pi inside the container (fast, ~3 seconds).
        # The task instruction is NOT sent yet; it will be prepended to
        # the first step() message automatically.
        logger.info("Calling reset()...")
        result = await client.reset()
        obs = result.observation

        logger.info("Phase: %s", obs.phase)
        logger.info("Reset returned: %s", obs.response)

        # Step loop — the first step carries the instruction to pi.
        turn = 0
        while turn < max_turns:
            turn += 1
            elapsed = time.time() - t0

            # Check episode timeout (server-side is 900s)
            if elapsed > 890:
                logger.info("Approaching episode timeout, stopping.")
                break

            logger.info(
                "--- Turn %d | phase=%s | elapsed=%.0fs | remaining=%.0fs ---",
                turn, obs.phase, elapsed, obs.time_remaining_s,
            )

            # First turn: send a kickoff message; subsequent turns: smart continue
            if turn == 1:
                msg = (
                    "Please begin. Read the workspace, plan your approach, "
                    "then call submit_plan with your subtasks."
                )
            else:
                # Option C: Smart continue messages that nudge the agent
                # toward using the episode protocol.
                current_subtask = obs.current_subtask or "?"
                remaining = obs.time_remaining_s
                if obs.phase == "PLANNING":
                    msg = (
                        f"TURN TIMEOUT. You have {remaining:.0f}s remaining. "
                        f"You MUST call submit_plan NOW with your subtasks "
                        f"to enter the EXECUTING phase."
                    )
                elif obs.phase == "EXECUTING":
                    # Check if auto-submit feedback was provided
                    if obs.subtask_feedback and "score" in obs.subtask_feedback:
                        score = obs.subtask_feedback.get("score", 0)
                        best = obs.subtask_feedback.get("best_score", 0)
                        attempts_left = obs.subtask_feedback.get(
                            "attempts_remaining", 0
                        )
                        feedback = obs.subtask_feedback.get("feedback", "")
                        if attempts_left > 0 and score < 0.7:
                            msg = (
                                f"TURN TIMEOUT. Auto-submitted subtask "
                                f"{current_subtask}: score={score:.2f} "
                                f"(best={best:.2f}). "
                                f"Feedback: {feedback[:300]}\n\n"
                                f"You have {attempts_left} attempt(s) left "
                                f"and {remaining:.0f}s remaining. "
                                f"Fix the issues and call "
                                f"submit_subtask('{current_subtask}') again, "
                                f"then advance."
                            )
                        else:
                            msg = (
                                f"TURN TIMEOUT. Auto-submitted subtask "
                                f"{current_subtask}: score={score:.2f} "
                                f"(best={best:.2f}). "
                                f"Call advance() to move to the next subtask. "
                                f"You have {remaining:.0f}s remaining."
                            )
                    else:
                        msg = (
                            f"TURN TIMEOUT. You have {remaining:.0f}s remaining. "
                            f"You are working on subtask {current_subtask}. "
                            f"Call submit_subtask('{current_subtask}') NOW "
                            f"to get your score, then call advance() to proceed."
                        )
                else:
                    msg = "continue"

            result = await client.step(FrontierSweAction(message=msg))
            obs = result.observation

            snippet = (obs.response or "")[:300].replace("\n", " ")
            logger.info(
                "Response (%d chars): %s",
                len(obs.response or ""), snippet,
            )

            if obs.frozen_scores:
                logger.info("Scores: %s", obs.frozen_scores)

            if obs.subtask_feedback:
                logger.info(
                    "Auto-submit feedback: score=%.4f best=%.4f attempts_left=%d",
                    obs.subtask_feedback.get("score", 0),
                    obs.subtask_feedback.get("best_score", 0),
                    obs.subtask_feedback.get("attempts_remaining", 0),
                )

            if obs.episode_reward is not None:
                logger.info("Episode reward: %s", obs.episode_reward)

            # Stop when the episode is actually DONE
            if obs.phase == "DONE":
                logger.info("Episode reached DONE.")
                break

        # Final state
        state = await client.state()
        elapsed = time.time() - t0

        episode_result = {
            "turns": turn,
            "elapsed_s": round(elapsed, 1),
            "phase": obs.phase,
            "plan_score": getattr(state, "plan_score", None),
            "frozen_scores": dict(getattr(state, "frozen_scores", {})),
            "episode_reward": getattr(state, "episode_reward", obs.episode_reward),
            "tool_call_count": getattr(state, "tool_call_count", None),
            "plan": getattr(state, "plan", None),
            "done": result.done,
        }

    except Exception:
        elapsed = time.time() - t0
        logger.exception("Episode failed after %.1fs", elapsed)
        episode_result = {
            "error": True,
            "elapsed_s": round(elapsed, 1),
            "turns": turn if "turn" in dir() else 0, # pyright: ignore[reportPossiblyUnboundVariable]
        }
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    # Print summary
    logger.info("=" * 60)
    logger.info("EPISODE COMPLETE")
    logger.info("=" * 60)
    for k, v in episode_result.items():
        logger.info("  %-18s %s", k + ":", v)
    logger.info("=" * 60)

    # Write result
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episode_result, indent=2))
    logger.info("Result written to %s", out)

    # Dump container logs (captures server-side event logging)
    _dump_container_logs(output_path)

    return episode_result


def _dump_container_logs(output_path: str) -> None:
    """Dump docker logs and pi session log from the container."""
    import subprocess

    out_dir = Path(output_path).parent

    # Docker logs (server-side: tool calls, MCP interactions, rubric scores)
    try:
        result = subprocess.run(
            ["docker", "logs", "fswe-baseline"],
            capture_output=True, text=True, timeout=10,
        )
        log_path = out_dir / "container_logs.txt"
        log_path.write_text(result.stdout + result.stderr)
        logger.info("Container logs written to %s (%d lines)",
                    log_path, log_path.read_text().count("\n"))
    except Exception as e:
        logger.warning("Failed to dump container logs: %s", e)

    # Pi session log (complete agent trajectory: every tool call, LLM response, etc.)
    try:
        result = subprocess.run(
            ["docker", "exec", "fswe-baseline", "bash", "-c",
             "find /root/.pi/agent/sessions -name '*.jsonl' -type f 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=5,
        )
        session_file = result.stdout.strip()
        if session_file:
            result = subprocess.run(
                ["docker", "cp", f"fswe-baseline:{session_file}",
                 str(out_dir / "pi_session.jsonl")],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info("Pi session log copied to %s", out_dir / "pi_session.jsonl")
            else:
                logger.warning("Failed to copy pi session log")
        else:
            logger.info("No pi session log found in container")
    except Exception as e:
        logger.warning("Failed to extract pi session log: %s", e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a FrontierSWE baseline episode",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Environment server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=100,
        help="Max step() calls (default: 100)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="WebSocket message timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--output", default="baseline_result.json",
        help="Output file for result JSON (default: baseline_result.json)",
    )
    args = parser.parse_args()

    result = asyncio.run(run_episode(
        base_url=args.url,
        max_turns=args.max_turns,
        message_timeout_s=args.timeout,
        output_path=args.output,
    ))

    if result.get("error"):
        sys.exit(1)
    if result.get("phase") != "DONE":
        logger.warning("Episode did not reach DONE (got %s)", result.get("phase"))


if __name__ == "__main__":
    main()
