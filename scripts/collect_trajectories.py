#!/usr/bin/env python3
"""
Collect DPO trajectories by running N episodes across W parallel workers.

Spins up W containers (one per worker), then round-robins episodes
across them. Each episode produces:
  - result.json          (episode metadata + reward)
  - pi_session.jsonl     (full agent trajectory)
  - container_logs.txt   (server-side scoring logs)

Default behaviour preserves the original postgres-only flow: with no flags
the script is identical to its pre-multi-task form (20 episodes, 4 workers,
600 s WS timeout, ``frontier-swe-pg:latest`` image). Other tasks
(``notebook``, ``type-checker``, ``libexpat-to-x86asm``) are opt-in via
``--task``. Reward shapes (ratio / reward_json / reward_json_score) are
L1-rubric concerns; by the time scores reach ``frozen_scores`` they are
normalised [0, 1] floats, and the ``EpisodeRubric`` weights (plan / subtask
/ completion / tool) come from ``TaskConfig`` defaults that no task
currently overrides — so the offline reward backfill works for every task.

Usage:
    # Original postgres flow (unchanged): 20 episodes, 4 workers, 45-min ep
    PYTHONPATH=. uv run python scripts/collect_trajectories.py

    # Quick reward-shape smoke against another task
    PYTHONPATH=. uv run python scripts/collect_trajectories.py \
        --task type-checker --episodes 2 --workers 1

    # Override the image tag (useful for local :smoke builds)
    PYTHONPATH=. uv run python scripts/collect_trajectories.py \
        --task type-checker \
        --image frontier-swe-dependent-type-checker:smoke

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

from frontier_swe_env.client import FrontierSweEnv  # noqa: E402
from frontier_swe_env.models import FrontierSweAction  # noqa: E402


def _disable_openenv_ws_keepalive() -> None:
    """Disable the websockets keepalive ping for the EnvClient.

    The notebook env can run a single agent turn for >10 minutes (e.g. 96
    bash calls in a row) without yielding back to the WS event loop. The
    `websockets` library's auto-ping defaults (interval=20s, timeout=20s)
    fire during that gap, the env never pongs in time, and the connection
    dies with `ConnectionClosedError: keepalive ping timeout` mid-episode.
    The OpenEnv pin doesn't expose ping_interval/ping_timeout kwargs yet,
    so we rebind the `ws_connect` symbol inside `openenv.core.env_client`
    to a wrapper that injects `ping_interval=None`.

    TODO: remove once openenv-core exposes ping kwargs upstream and we bump
    the pin in pyproject.toml.
    """
    from openenv.core import env_client

    assert hasattr(env_client, "ws_connect"), (
        "openenv-core layout changed: env_client no longer exposes "
        "ws_connect. Remove this monkey-patch or update it."
    )

    _orig_connect = env_client.ws_connect

    def _connect_no_keepalive(*args, **kwargs):
        kwargs.setdefault("ping_interval", None)
        return _orig_connect(*args, **kwargs)

    env_client.ws_connect = _connect_no_keepalive


_disable_openenv_ws_keepalive()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)


# Constants

# Per-task profile. Postgres values match the original hardcoded constants
# verbatim (2700 s episode, 600 s WS message timeout, frontier-swe-pg:latest
# image) so the default flow with no flags reproduces the pre-multi-task
# behaviour bit-for-bit. Other tasks need a longer WS timeout because pi
# explores deeper before submitting (observed up to 900 s on type-checker).
TASK_PROFILES: dict[str, dict] = {
    "postgres": {
        "image": "frontier-swe-pg:latest",
        "episode_timeout_s": 2700,
        "message_timeout_s": 600.0,
        "dockerfile": "docker/Dockerfile.pg",
    },
    "type-checker": {
        "image": "frontier-swe-dependent-type-checker:latest",
        "episode_timeout_s": 1800,
        "message_timeout_s": 900.0,
        "dockerfile": "docker/Dockerfile.dependent-type-checker",
    },
    "notebook": {
        "image": "frontier-swe-notebook:latest",
        "episode_timeout_s": 1800,
        "message_timeout_s": 900.0,
        "dockerfile": "docker/Dockerfile.notebook",
    },
    "libexpat-to-x86asm": {
        "image": "frontier-swe-libexpat-to-x86asm:latest",
        "episode_timeout_s": 1800,
        "message_timeout_s": 900.0,
        "dockerfile": "docker/Dockerfile.libexpat-to-x86asm",
    },
}

# Mutated by main() once CLI args are parsed. Defaults below match the
# original postgres-hardcoded values so an import-side smoke that doesn't
# call main() still sees PG semantics.
DOCKER_IMAGE: str = TASK_PROFILES["postgres"]["image"]
DOCKERFILE: str = TASK_PROFILES["postgres"]["dockerfile"]
EPISODE_TIMEOUT_S: int = TASK_PROFILES["postgres"]["episode_timeout_s"]
MESSAGE_TIMEOUT_S: float = TASK_PROFILES["postgres"]["message_timeout_s"]

# Mutated by main() — see --container-prefix / --base-port. Defaults are
# the original PG-flow values so existing call sites stay byte-identical.
CONTAINER_PREFIX = "fswe-worker"
BASE_PORT = 8100  # workers use ports 8100, 8101, 8102, ...

ENV_FILE = ".env"
MAX_TURNS = 20
CONTAINER_STARTUP_WAIT = 10  # seconds to wait after container run
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 2

# Container runtime — `docker` or `podman`. Set via --runtime / $CONTAINER_RUNTIME
# or auto-detected at startup. The two CLIs share the verbs we use
# (run, rm, exec, logs, cp, image inspect), so the same code path works.
CONTAINER_RUNTIME: str = "docker"


# Offline reward computation


def _compute_reward_offline(result: dict) -> float:
    """Compute episode reward from result.json data.

    Same formula as EpisodeRubric.compute(), applied to the client-side
    state snapshot when the server didn't transition to DONE.
    """
    plan = result.get("plan")
    plan_score = result.get("plan_score", 0.0) or 0.0
    frozen_scores = result.get("frozen_scores", {}) or {}
    tool_call_count = result.get("tool_call_count", 0) or 0

    plan_count = max(len(plan), 1) if plan else 1

    # Weights match EpisodeRubric defaults from TaskConfig. No task in the
    # registry currently overrides these, so the same formula applies to
    # postgres, notebook, type-checker and libexpat-to-x86asm.
    plan_weight = 0.25
    subtask_weight = 0.60
    completion_weight = 0.10
    tool_weight = 0.05

    scores = list(frozen_scores.values())
    while len(scores) < plan_count:
        scores.append(0.0)
    subtask_mean = sum(scores) / max(len(scores), 1)

    scored_count = len(frozen_scores)
    completion = min(scored_count / plan_count, 1.0)

    tool_density = min(tool_call_count / (5 * plan_count), 1.0)

    reward = (
        plan_weight * plan_score
        + subtask_weight * subtask_mean
        + completion_weight * completion
        + tool_weight * tool_density
    )
    return max(0.0, min(1.0, reward))


# Container management


def container_name(worker_id: int) -> str:
    return f"{CONTAINER_PREFIX}-{worker_id}"


def start_container(worker_id: int) -> bool:
    """Start a Docker container for the given worker. Returns True on success."""
    name = container_name(worker_id)
    port = BASE_PORT + worker_id

    # Remove any existing container with this name
    subprocess.run(
        [CONTAINER_RUNTIME, "rm", "-f", name],
        capture_output=True,
        timeout=10,
    )

    cmd = [
        CONTAINER_RUNTIME,
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{port}:8000",
        "--env-file",
        ENV_FILE,
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

    logger.error(
        "Worker %d failed health check after %d attempts",
        worker_id,
        HEALTH_CHECK_RETRIES,
    )
    return False


def stop_container(worker_id: int) -> None:
    """Stop and remove a worker container."""
    name = container_name(worker_id)
    subprocess.run([CONTAINER_RUNTIME, "rm", "-f", name], capture_output=True, timeout=15)
    logger.info("Stopped container %s", name)


def reset_container(worker_id: int) -> bool:
    """Stop and restart a container for a fresh episode.

    Pi persists its session across reset() calls within the same container
    because the session file stays on disk. To get a truly independent
    trajectory for each episode, we restart the container.
    """
    name = container_name(worker_id)

    # Remove old container
    subprocess.run([CONTAINER_RUNTIME, "rm", "-f", name], capture_output=True, timeout=15)
    time.sleep(1)

    # Start fresh
    if not start_container(worker_id):
        return False
    return wait_for_healthy(worker_id)


# Artifact extraction


def extract_artifacts(worker_id: int, episode_dir: Path) -> dict:
    """Extract logs and session JSONL from a worker container."""
    name = container_name(worker_id)
    artifacts = {"container_logs": False, "pi_session": False}

    # Container logs
    try:
        result = subprocess.run(
            [CONTAINER_RUNTIME, "logs", name],
            capture_output=True,
            text=True,
            timeout=15,
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
            [
                CONTAINER_RUNTIME,
                "exec",
                name,
                "bash",
                "-c",
                "find /root/.pi/agent/sessions -name '*.jsonl' -type f 2>/dev/null | head -1",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        session_file = result.stdout.strip()

        if not session_file:
            result = subprocess.run(
                [
                    CONTAINER_RUNTIME,
                    "exec",
                    name,
                    "bash",
                    "-c",
                    "find /root/.pi -name '*.jsonl' -type f 2>/dev/null | head -1",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            session_file = result.stdout.strip()

        if session_file:
            dest = episode_dir / "pi_session.jsonl"
            result = subprocess.run(
                [CONTAINER_RUNTIME, "cp", f"{name}:{session_file}", str(dest)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and dest.exists():
                size_kb = dest.stat().st_size / 1024
                lines = dest.read_text().count("\n")
                artifacts["pi_session"] = True
                logger.info("  Pi session: %.1f KB, %d lines", size_kb, lines)
            else:
                logger.warning(
                    "  docker cp failed: %s",
                    result.stderr[:200] if result.stderr else "unknown",
                )
        else:
            logger.warning("  No pi_session.jsonl found in container!")
    except Exception as e:
        logger.warning("  Failed to extract pi session: %s", e)

    return artifacts


# Single episode runner (adapted from run_baseline.py)


async def run_single_episode(
    worker_id: int,
    episode_id: int,
    episode_dir: Path,
) -> dict:
    """Run one episode on the given worker. Returns the episode result dict."""

    port = BASE_PORT + worker_id
    base_url = f"http://localhost:{port}"

    logger.info(
        "Episode %d starting on worker %d (port %d)", episode_id, worker_id, port
    )

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

            # Brief per-turn log
            scores_str = (
                " ".join(f"{k}={v:.2f}" for k, v in obs.frozen_scores.items())
                if obs.frozen_scores
                else "none"
            )
            logger.info(
                "  Ep %d turn %d: phase=%s scores=[%s] remaining=%.0fs",
                episode_id,
                turn,
                obs.phase,
                scores_str,
                obs.time_remaining_s,
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

        # Backfill reward if the server didn't compute one (episode didn't
        # reach DONE because the client hit max_turns or timeout first).
        if episode_result["episode_reward"] is None:
            episode_result["episode_reward"] = _compute_reward_offline(episode_result)
            episode_result["_reward_backfilled"] = True
            logger.info(
                "  Ep %d: backfilled reward=%.4f",
                episode_id,
                episode_result["episode_reward"],
            )

    except Exception as e:
        elapsed = time.time() - t0
        logger.exception("  Ep %d failed after %.1fs: %s", episode_id, elapsed, e)
        episode_result = {
            "episode_id": episode_id,
            "worker_id": worker_id,
            "error": str(e) or type(e).__name__,
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


# Worker loop


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
        logger.info(
            "Worker %d: restarting container for episode %d", worker_id, episode_id
        )
        ok = await asyncio.to_thread(reset_container, worker_id)
        if not ok:
            logger.error(
                "Worker %d: container restart failed, skipping episode %d",
                worker_id,
                episode_id,
            )
            results.append(
                {
                    "episode_id": episode_id,
                    "worker_id": worker_id,
                    "error": "container_restart_failed",
                }
            )
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
            episode_id,
            reward,
            phase,
            has_jsonl,
            ep_result.get("turns", 0),
            ep_result.get("elapsed_s", 0),
        )

        episode_queue.task_done()


# Main orchestrator


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
                    if data.get("episode_reward") is not None or data.get(
                        "frozen_scores"
                    ):
                        ep_id = int(ep_dir.name.split("_")[1])
                        skip_episodes.add(ep_id)
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass
        if skip_episodes:
            logger.info(
                "Resuming: skipping %d completed episodes: %s",
                len(skip_episodes),
                sorted(skip_episodes),
            )

    remaining = num_episodes - len(skip_episodes)
    if remaining <= 0:
        logger.info("All %d episodes already completed!", num_episodes)
        return

    logger.info("=" * 70)
    logger.info("Trajectory Collection")
    logger.info("=" * 70)
    logger.info("Episodes:    %d (%d remaining)", num_episodes, remaining)
    logger.info("Workers:     %d", num_workers)
    logger.info("Image:       %s", DOCKER_IMAGE)
    logger.info("Output:      %s/", out)
    logger.info(
        "Per-episode soft ceiling: %d s (env timer is authoritative)",
        EPISODE_TIMEOUT_S,
    )
    logger.info("=" * 70)

    # Verify Docker image exists
    result = subprocess.run(
        [CONTAINER_RUNTIME, "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        logger.error(
            "Docker image %s not found. Build it first:\n"
            "  docker build -f %s -t %s .",
            DOCKER_IMAGE,
            DOCKERFILE,
            DOCKER_IMAGE,
        )
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
            "top_quartile_min": round(rewards[3 * len(rewards) // 4], 4)
            if len(rewards) >= 4
            else None,
            "bottom_quartile_max": round(rewards[len(rewards) // 4], 4)
            if len(rewards) >= 4
            else None,
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
        1
        for r in results
        if not r.get("_artifacts", {}).get("pi_session", False) and not r.get("error")
    )
    if missing_jsonl > 0:
        logger.warning(
            "%d episodes completed but have NO pi_session.jsonl! "
            "Check the --no-session fix.",
            missing_jsonl,
        )

    logger.info("=" * 70)


# Entrypoint


def main():
    parser = argparse.ArgumentParser(
        description="Collect DPO trajectories across parallel workers",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_PROFILES.keys()),
        default="postgres",
        help=(
            "Task slug — selects the default image and timeouts. "
            "Default is postgres (preserves the pre-multi-task hardcoded flow)."
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=["docker", "podman"],
        default=None,
        help=(
            "Container runtime CLI to use. Default: $CONTAINER_RUNTIME, "
            "else auto-detect (prefer docker, fall back to podman)."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Override the Docker image tag (useful for local :smoke builds). "
            "Defaults to the image registered for --task."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Total number of episodes to collect (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel containers (default: 4)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8100,
        help=(
            "Host port for worker_id=0 (subsequent workers use base+1, "
            "base+2, ...). Bump this to run multiple invocations of the "
            "script in parallel without colliding (default: 8100)."
        ),
    )
    parser.add_argument(
        "--container-prefix",
        default="fswe-worker",
        help=(
            "Prefix used to name worker containers (full name is "
            "<prefix>-<worker_id>). Use a unique value per parallel "
            "invocation so runs don't fight over container names."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="trajectories",
        help="Output directory for trajectory data (default: trajectories/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes that already have result.json + pi_session.jsonl",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override MAX_TURNS per episode (default: 20)",
    )
    parser.add_argument(
        "--episode-timeout",
        type=int,
        default=None,
        help=(
            "Override episode timeout in seconds. Default is the value "
            "registered for --task in TASK_PROFILES."
        ),
    )
    args = parser.parse_args()

    profile = TASK_PROFILES[args.task]
    global DOCKER_IMAGE, DOCKERFILE, EPISODE_TIMEOUT_S, MESSAGE_TIMEOUT_S
    global CONTAINER_RUNTIME, BASE_PORT, CONTAINER_PREFIX
    DOCKER_IMAGE = args.image or profile["image"]
    DOCKERFILE = profile["dockerfile"]
    EPISODE_TIMEOUT_S = (
        args.episode_timeout
        if args.episode_timeout is not None
        else profile["episode_timeout_s"]
    )
    MESSAGE_TIMEOUT_S = profile["message_timeout_s"]
    BASE_PORT = args.base_port
    CONTAINER_PREFIX = args.container_prefix

    import os
    import shutil

    runtime = args.runtime or os.environ.get("CONTAINER_RUNTIME")
    if not runtime:
        runtime = "docker" if shutil.which("docker") else (
            "podman" if shutil.which("podman") else "docker"
        )
    if not shutil.which(runtime):
        logger.error(
            "Container runtime %r not on PATH. Install it or pass --runtime.",
            runtime,
        )
        sys.exit(1)
    CONTAINER_RUNTIME = runtime
    logger.info("Container runtime: %s", CONTAINER_RUNTIME)

    if args.max_turns is not None:
        global MAX_TURNS
        MAX_TURNS = args.max_turns

    asyncio.run(
        collect(
            num_episodes=args.episodes,
            num_workers=args.workers,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
