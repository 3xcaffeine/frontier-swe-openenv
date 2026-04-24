#!/usr/bin/env python3
"""
Collect DPO trajectories by running N episodes across W parallel workers.

Spins up W Docker containers (one per worker), then round-robins episodes
across them. Each episode produces:
  - result.json          (episode metadata + reward)
  - pi_session.jsonl     (full agent trajectory)
  - container_logs.txt   (server-side scoring logs)

Usage:
    # 20 episodes across 4 parallel workers (default)
    PYTHONPATH=. uv run python scripts/collect_trajectories.py

    # Custom settings
    PYTHONPATH=. uv run python scripts/collect_trajectories.py \
        --episodes 20 --workers 4 --output-dir trajectories/

    # Resume from a previous run (skips existing episodes)
    PYTHONPATH=. uv run python scripts/collect_trajectories.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

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
logger = logging.getLogger("collect")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "frontier-swe-pg:latest"
CONTAINER_PREFIX = "fswe-worker"
BASE_PORT = 8100  # workers use ports 8100, 8101, 8102, ...
ENV_FILE = ".env"
MAX_TURNS = 20
MESSAGE_TIMEOUT_S = 600.0
EPISODE_TIMEOUT_S = 2700  # 45 min (must match task_config)
CONTAINER_STARTUP_WAIT = 10  # seconds to wait after docker run
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 2


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

def container_name(worker_id: int) -> str:
    return f"{CONTAINER_PREFIX}-{worker_id}"


def start_container(worker_id: int) -> bool:
    """Start a Docker container for the given worker. Returns True on success."""
    name = container_name(worker_id)
    port = BASE_PORT + worker_id

    # Remove any existing container with this name
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, timeout=10,
    )

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "-p", f"{port}:8000",
        "--env-file", ENV_FILE,
        DOCKER_IMAGE,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.error("Failed to start container %s: %s", name, result.stderr.strip())
        return False

    logger.info("Started container %s on port %d", name, port)
    return True


def wait_for_healthy(worker_id: int) -> bool:
    """Wait for the container's health endpoint to respond."""
    import urllib.request
    import urllib.error

    port = BASE_PORT + worker_id
    url = f"http://localhost:{port}/health"

    for attempt in range(HEALTH_CHECK_RETRIES):
        try:
            req = urllib.request.urlopen(url, timeout=3)
            if req.status == 200:
                logger.info("Worker %d healthy", worker_id)
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)

    logger.error("Worker %d failed health check after %d attempts",
                 worker_id, HEALTH_CHECK_RETRIES)
    return False


def stop_container(worker_id: int) -> None:
    """Stop and remove a worker container."""
    name = container_name(worker_id)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
    logger.info("Stopped container %s", name)


def reset_container(worker_id: int) -> bool:
    """Stop and restart a container for a fresh episode.

    Pi persists its session across reset() calls within the same container
    because the session file stays on disk. To get a truly independent
    trajectory for each episode, we restart the container.
    """
    name = container_name(worker_id)

    # Remove old container
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
    time.sleep(1)

    # Start fresh
    if not start_container(worker_id):
        return False
    return wait_for_healthy(worker_id)


# ---------------------------------------------------------------------------
# Artifact extraction
# ---------------------------------------------------------------------------

def extract_artifacts(worker_id: int, episode_dir: Path) -> dict:
    """Extract logs and session JSONL from a worker container."""
    name = container_name(worker_id)
    artifacts = {"container_logs": False, "pi_session": False}

    # Container logs
    try:
        result = subprocess.run(
            ["docker", "logs", name],
            capture_output=True, text=True, timeout=15,
        )
        log_path = episode_dir / "container_logs.txt"
        log_path.write_text(result.stdout + result.stderr)
        artifacts["container_logs"] = True
        logger.info("  Container logs: %d lines", log_path.read_text().count("\n"))
    except Exception as e:
        logger.warning("  Failed to dump container logs: %s", e)

    # Pi session JSONL
    try:
        result = subprocess.run(
            ["docker", "exec", name, "bash", "-c",
             "find /root/.pi/agent/sessions -name '*.jsonl' -type f 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=5,
        )
        session_file = result.stdout.strip()

        if not session_file:
            result = subprocess.run(
                ["docker", "exec", name, "bash", "-c",
                 "find /root/.pi -name '*.jsonl' -type f 2>/dev/null | head -1"],
                capture_output=True, text=True, timeout=5,
            )
            session_file = result.stdout.strip()

        if session_file:
            dest = episode_dir / "pi_session.jsonl"
            result = subprocess.run(
                ["docker", "cp", f"{name}:{session_file}", str(dest)],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and dest.exists():
                size_kb = dest.stat().st_size / 1024
                lines = dest.read_text().count("\n")
                artifacts["pi_session"] = True
                logger.info("  Pi session: %.1f KB, %d lines", size_kb, lines)
            else:
                logger.warning("  docker cp failed: %s",
                              result.stderr[:200] if result.stderr else "unknown")
        else:
            logger.warning("  No pi_session.jsonl found in container!")
    except Exception as e:
        logger.warning("  Failed to extract pi session: %s", e)

    return artifacts


# ---------------------------------------------------------------------------
# Single episode runner (adapted from run_baseline.py)
# ---------------------------------------------------------------------------

async def run_single_episode(
    worker_id: int,
    episode_id: int,
    episode_dir: Path,
) -> dict:
    """Run one episode on the given worker. Returns the episode result dict."""

    port = BASE_PORT + worker_id
    base_url = f"http://localhost:{port}"

    logger.info("Episode %d starting on worker %d (port %d)", episode_id, worker_id, port)

    client = FrontierSweEnv(
        base_url=base_url,
        message_timeout_s=MESSAGE_TIMEOUT_S,
    )

    t0 = time.time()
    turn = 0

    try:
        await client.connect()

        result = await client.reset()
        obs = result.observation

        while turn < MAX_TURNS:
            turn += 1
            elapsed = time.time() - t0

            if elapsed > EPISODE_TIMEOUT_S - 10:
                logger.info("  Ep %d: approaching timeout at turn %d", episode_id, turn)
                break

            # Build message
            if turn == 1:
                msg = (
                    "Please begin. Read the workspace, plan your approach, "
                    "then call submit_plan with your subtasks."
                )
            else:
                current_subtask = obs.current_subtask or "?"
                remaining = obs.time_remaining_s

                if obs.phase == "PLANNING":
                    msg = (
                        f"TURN TIMEOUT. You have {remaining:.0f}s remaining. "
                        f"You MUST call submit_plan NOW with your subtasks "
                        f"to enter the EXECUTING phase."
                    )
                elif obs.phase == "EXECUTING":
                    if obs.subtask_feedback and "score" in obs.subtask_feedback:
                        score = obs.subtask_feedback.get("score", 0)
                        best = obs.subtask_feedback.get("best_score", 0)
                        attempts_left = obs.subtask_feedback.get("attempts_remaining", 0)
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

            # Brief per-turn log
            scores_str = " ".join(f"{k}={v:.2f}" for k, v in obs.frozen_scores.items()) if obs.frozen_scores else "none"
            logger.info(
                "  Ep %d turn %d: phase=%s scores=[%s] remaining=%.0fs",
                episode_id, turn, obs.phase, scores_str, obs.time_remaining_s,
            )

            if obs.phase == "DONE":
                logger.info("  Ep %d reached DONE at turn %d", episode_id, turn)
                break

        # Final state
        state = await client.state()
        elapsed = time.time() - t0

        episode_result = {
            "episode_id": episode_id,
            "worker_id": worker_id,
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

    except Exception as e:
        elapsed = time.time() - t0
        logger.exception("  Ep %d failed after %.1fs: %s", episode_id, elapsed, e)
        episode_result = {
            "episode_id": episode_id,
            "worker_id": worker_id,
            "error": str(e),
            "elapsed_s": round(elapsed, 1),
            "turns": turn,
        }
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    # Save result
    episode_dir.mkdir(parents=True, exist_ok=True)
    result_path = episode_dir / "result.json"
    result_path.write_text(json.dumps(episode_result, indent=2))

    # Extract artifacts from container
    artifacts = extract_artifacts(worker_id, episode_dir)
    episode_result["_artifacts"] = artifacts

    return episode_result


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

async def worker_loop(
    worker_id: int,
    episode_queue: asyncio.Queue,
    output_dir: Path,
    results: list,
    skip_episodes: set[int],
) -> None:
    """Worker coroutine: pulls episode IDs from the queue and runs them."""

    while True:
        try:
            episode_id = episode_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if episode_id in skip_episodes:
            logger.info("Skipping episode %d (already completed)", episode_id)
            episode_queue.task_done()
            continue

        episode_dir = output_dir / f"episode_{episode_id:03d}"

        # Restart container for a clean slate
        logger.info("Worker %d: restarting container for episode %d", worker_id, episode_id)
        ok = await asyncio.to_thread(reset_container, worker_id)
        if not ok:
            logger.error("Worker %d: container restart failed, skipping episode %d",
                         worker_id, episode_id)
            results.append({
                "episode_id": episode_id,
                "worker_id": worker_id,
                "error": "container_restart_failed",
            })
            episode_queue.task_done()
            continue

        # Run the episode
        ep_result = await run_single_episode(worker_id, episode_id, episode_dir)
        results.append(ep_result)

        reward = ep_result.get("episode_reward")
        phase = ep_result.get("phase", "?")
        has_jsonl = ep_result.get("_artifacts", {}).get("pi_session", False)
        logger.info(
            "Episode %d complete: reward=%s phase=%s jsonl=%s turns=%d elapsed=%.0fs",
            episode_id, reward, phase, has_jsonl, ep_result.get("turns", 0),
            ep_result.get("elapsed_s", 0),
        )

        episode_queue.task_done()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def collect(
    num_episodes: int = 20,
    num_workers: int = 4,
    output_dir: str = "trajectories",
    resume: bool = False,
) -> None:
    """Collect trajectories across parallel workers."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check which episodes are already done (for --resume)
    skip_episodes: set[int] = set()
    if resume:
        for ep_dir in out.glob("episode_*"):
            result_file = ep_dir / "result.json"
            session_file = ep_dir / "pi_session.jsonl"
            if result_file.exists() and session_file.exists():
                try:
                    data = json.loads(result_file.read_text())
                    if data.get("episode_reward") is not None or data.get("frozen_scores"):
                        ep_id = int(ep_dir.name.split("_")[1])
                        skip_episodes.add(ep_id)
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass
        if skip_episodes:
            logger.info("Resuming: skipping %d completed episodes: %s",
                         len(skip_episodes), sorted(skip_episodes))

    remaining = num_episodes - len(skip_episodes)
    if remaining <= 0:
        logger.info("All %d episodes already completed!", num_episodes)
        return

    logger.info("=" * 70)
    logger.info("Trajectory Collection")
    logger.info("=" * 70)
    logger.info("Episodes:    %d (%d remaining)", num_episodes, remaining)
    logger.info("Workers:     %d", num_workers)
    logger.info("Output:      %s/", out)
    logger.info("Per episode: ~45 min (2700s episode + overhead)")
    logger.info("Estimated:   ~%.0f min total",
                 remaining / num_workers * 50)  # 45 min + 5 min overhead
    logger.info("=" * 70)

    # Verify Docker image exists
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True, timeout=10,
    )
    if result.returncode != 0:
        logger.error("Docker image %s not found. Build it first:\n"
                     "  docker build -f docker/Dockerfile.pg -t %s .",
                     DOCKER_IMAGE, DOCKER_IMAGE)
        sys.exit(1)

    # Verify .env file exists
    if not Path(ENV_FILE).exists():
        logger.error(".env file not found at %s", ENV_FILE)
        sys.exit(1)

    # Build episode queue
    queue: asyncio.Queue[int] = asyncio.Queue()
    for ep_id in range(1, num_episodes + 1):
        queue.put_nowait(ep_id)

    # Start all workers
    results: list[dict] = []
    t0 = time.time()

    logger.info("Starting %d worker containers...", num_workers)
    for w in range(num_workers):
        ok = start_container(w)
        if not ok:
            logger.error("Failed to start worker %d, aborting", w)
            for j in range(w):
                stop_container(j)
            sys.exit(1)

    # Wait for all containers to be healthy
    logger.info("Waiting for containers to be healthy...")
    for w in range(num_workers):
        if not wait_for_healthy(w):
            logger.error("Worker %d not healthy, aborting", w)
            for j in range(num_workers):
                stop_container(j)
            sys.exit(1)

    logger.info("All %d workers healthy. Starting collection...", num_workers)

    # Run worker coroutines concurrently
    tasks = [
        asyncio.create_task(worker_loop(w, queue, out, results, skip_episodes))
        for w in range(num_workers)
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving partial results...")
    finally:
        # Cleanup containers
        logger.info("Stopping worker containers...")
        for w in range(num_workers):
            stop_container(w)

    elapsed = time.time() - t0

    # Write summary
    summary = {
        "total_episodes": len(results),
        "elapsed_s": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 1),
        "episodes": [],
    }

    successful = 0
    rewards = []
    for r in sorted(results, key=lambda x: x.get("episode_id", 0)):
        ep_summary = {
            "episode_id": r.get("episode_id"),
            "reward": r.get("episode_reward"),
            "phase": r.get("phase"),
            "turns": r.get("turns"),
            "elapsed_s": r.get("elapsed_s"),
            "has_jsonl": r.get("_artifacts", {}).get("pi_session", False),
            "error": r.get("error"),
        }
        summary["episodes"].append(ep_summary)
        if r.get("episode_reward") is not None:
            successful += 1
            rewards.append(r["episode_reward"])

    summary["successful_episodes"] = successful
    summary["failed_episodes"] = len(results) - successful

    if rewards:
        rewards.sort()
        summary["reward_stats"] = {
            "min": round(min(rewards), 4),
            "max": round(max(rewards), 4),
            "mean": round(sum(rewards) / len(rewards), 4),
            "median": round(rewards[len(rewards) // 2], 4),
            "top_quartile_min": round(rewards[3 * len(rewards) // 4], 4) if len(rewards) >= 4 else None,
            "bottom_quartile_max": round(rewards[len(rewards) // 4], 4) if len(rewards) >= 4 else None,
        }

    summary_path = out / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print final report
    logger.info("=" * 70)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time:        %.1f min", elapsed / 60)
    logger.info("Episodes run:      %d", len(results))
    logger.info("Successful:        %d", successful)
    logger.info("Failed:            %d", len(results) - successful)
    if rewards:
        logger.info("Reward range:      %.4f - %.4f", min(rewards), max(rewards))
        logger.info("Reward mean:       %.4f", sum(rewards) / len(rewards))
    logger.info("Summary written to %s", summary_path)

    # Check for missing JSONLs
    missing_jsonl = sum(
        1 for r in results
        if not r.get("_artifacts", {}).get("pi_session", False)
        and not r.get("error")
    )
    if missing_jsonl > 0:
        logger.warning(
            "%d episodes completed but have NO pi_session.jsonl! "
            "Check the --no-session fix.", missing_jsonl
        )

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect DPO trajectories across parallel workers",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Total number of episodes to collect (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel Docker containers (default: 4)",
    )
    parser.add_argument(
        "--output-dir", default="trajectories",
        help="Output directory for trajectory data (default: trajectories/)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip episodes that already have result.json + pi_session.jsonl",
    )
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Override MAX_TURNS per episode (default: 20)",
    )
    parser.add_argument(
        "--episode-timeout", type=int, default=None,
        help="Override episode timeout in seconds (default: 2700 = 45 min)",
    )
    args = parser.parse_args()

    if args.max_turns is not None:
        global MAX_TURNS
        MAX_TURNS = args.max_turns
    if args.episode_timeout is not None:
        global EPISODE_TIMEOUT_S
        EPISODE_TIMEOUT_S = args.episode_timeout

    asyncio.run(collect(
        num_episodes=args.episodes,
        num_workers=args.workers,
        output_dir=args.output_dir,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
