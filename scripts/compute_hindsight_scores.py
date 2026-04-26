#!/usr/bin/env python3
"""Compute HCAPO hindsight credit assignment scores for collected trajectories.

For each episode, for each assistant step, this script:
1. Builds a hindsight-augmented prompt (injects final outcome into context)
2. Calls SGLang's native /generate endpoint to get log-probabilities
   of the original action tokens given the hindsight context
3. Computes the hindsight importance ratio rho_t and Q_H values

Based on HCAPO (paper 2603.08754), Eq. 5-7.

Usage:
    uv run python scripts/compute_hindsight_scores.py \\
        --api-base "$FSWE_AGENT_API_URL" \\
        --model "$FSWE_AGENT_MODEL" \\
        --api-key "$FSWE_AGENT_API_KEY"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from build_training_dataset import load_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hindsight_scores")

HINDSIGHT_TEMPLATE = """\
[HINDSIGHT — This is post-hoc scoring context, not visible during generation]
Trajectory outcome:
- Final reward: {reward:.4f}
- Phase reached: {phase}
- Plan score: {plan_score}
- Subtask scores: {scores_summary}
- Subtasks completed: {scored_count}/{plan_count}
- Current subtask: {current_subtask}
- Current subtask score: {current_subtask_score}"""


# ---------------------------------------------------------------------------
# Message normalisation helpers
# ---------------------------------------------------------------------------

def _unwrap_arguments(arguments: Any) -> str:
    """Convert the {"arguments": "json"} wrapper to a plain JSON string."""
    if isinstance(arguments, dict):
        inner = arguments.get("arguments")
        if inner is not None:
            return inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False)
        return json.dumps(arguments, ensure_ascii=False)
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False) if arguments is not None else "{}"


def normalize_message_for_template(msg: dict) -> dict:
    """Make tool_calls/tool messages compatible with Qwen chat templates."""
    msg = dict(msg)
    if msg.get("tool_calls"):
        calls = []
        for tc in msg["tool_calls"]:
            tc = dict(tc)
            fn = dict(tc.get("function", {}))
            fn["arguments"] = _unwrap_arguments(fn.get("arguments"))
            tc["function"] = fn
            calls.append(tc)
        msg["tool_calls"] = calls
    return msg


def normalize_messages(messages: list[dict]) -> list[dict]:
    return [normalize_message_for_template(m) for m in messages]


def _flatten_for_template(messages: list[dict]) -> list[dict]:
    """Fallback: flatten tool_calls and tool messages into plain text."""
    out: list[dict] = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "tool":
            m = {
                "role": "user",
                "content": f"[Tool Result: {m.get('name', 'tool')}]\n{m.get('content', '')}",
            }
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            parts = []
            if m.get("content"):
                parts.append(m["content"])
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                parts.append(f"[Tool Call: {fn.get('name', '?')}]\n{fn.get('arguments', '{}')}")
            m = {"role": "assistant", "content": "\n".join(parts)}
        out.append(m)
    return out


def safe_apply_chat_template(
    tokenizer: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """apply_chat_template with a fallback that flattens tool messages."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        flat = _flatten_for_template(messages)
        return tokenizer.apply_chat_template(
            flat, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


# ---------------------------------------------------------------------------
# Subtask mapping — assigns each assistant step a dense intermediate reward
# ---------------------------------------------------------------------------

def _extract_effective_tool_names(msg: dict) -> list[str]:
    """Extract effective tool names, unwrapping the ``mcp`` wrapper.

    Direct tool calls return the function name as-is.  For ``mcp``
    calls the inner ``tool`` field (e.g. ``openenv_submit_plan``) is
    extracted from the doubly-nested arguments.
    """
    names: list[str] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        name = fn.get("name", "")
        if name == "mcp":
            raw = fn.get("arguments", {})
            if isinstance(raw, dict):
                inner_str = raw.get("arguments", "")
            else:
                inner_str = raw
            if isinstance(inner_str, str):
                try:
                    inner = json.loads(inner_str)
                except (json.JSONDecodeError, TypeError):
                    inner = {}
            else:
                inner = inner_str if isinstance(inner_str, dict) else {}
            inner_name = inner.get("tool", "") if isinstance(inner, dict) else ""
            if inner_name:
                names.append(inner_name)
            else:
                names.append(name)
        else:
            names.append(name)
    return names


def _is_successful_response(content: str) -> bool:
    """Heuristic: a tool response indicates success if it looks like valid
    JSON and does not start with a known failure prefix."""
    c = content.strip()
    if not c:
        return False
    fail_prefixes = ("[tool_error]", "Failed to call tool", "Error:")
    for p in fail_prefixes:
        if c.startswith(p):
            return False
    if c.startswith("{"):
        try:
            obj = json.loads(c)
            return "error" not in obj
        except (json.JSONDecodeError, TypeError):
            return False
    return False


def map_steps_to_subtasks(messages: list[dict], episode: dict) -> list[dict]:
    """Map each assistant step to the subtask it was working on.

    Parses ``submit_plan`` / ``advance`` tool calls **and** their
    responses to detect phase transitions reliably.  Returns one entry
    per assistant message with phase, subtask_id, and the subtask's
    frozen_score as a dense intermediate reward.
    """
    plan = episode.get("plan") or []
    subtask_ids = [s["id"] for s in plan] if plan else []
    frozen_scores = episode.get("frozen_scores", {})
    plan_score = episode.get("plan_score", 0)

    current_phase = "planning"
    current_subtask_idx = -1
    pending_transition: str | None = None

    step_info: list[dict] = []

    for msg in messages:
        role = msg.get("role")

        # --- tool response: check if a pending transition succeeded ---
        if role == "tool" and pending_transition is not None:
            content = msg.get("content", "") or ""
            if _is_successful_response(content):
                if pending_transition == "submit_plan":
                    current_phase = "executing"
                    current_subtask_idx = 0
                elif pending_transition == "advance":
                    try:
                        resp = json.loads(content)
                        nxt = resp.get("next_subtask_id", "")
                        if nxt in subtask_ids:
                            current_subtask_idx = subtask_ids.index(nxt)
                        else:
                            current_subtask_idx = min(
                                current_subtask_idx + 1,
                                max(len(subtask_ids) - 1, 0),
                            )
                    except (json.JSONDecodeError, TypeError):
                        current_subtask_idx = min(
                            current_subtask_idx + 1,
                            max(len(subtask_ids) - 1, 0),
                        )
            pending_transition = None

        if role != "assistant":
            continue

        # --- record current phase for this step ---
        if current_phase == "planning":
            step_info.append({
                "phase": "planning",
                "subtask_id": None,
                "subtask_reward": plan_score,
            })
        else:
            sid = (
                subtask_ids[current_subtask_idx]
                if 0 <= current_subtask_idx < len(subtask_ids)
                else None
            )
            step_info.append({
                "phase": "executing",
                "subtask_id": sid,
                "subtask_reward": frozen_scores.get(sid, 0.0) if sid else 0.0,
            })

        # --- detect phase-transition tool calls ---
        for name in _extract_effective_tool_names(msg):
            canonical = name.replace("openenv_", "")
            if canonical == "submit_plan":
                pending_transition = "submit_plan"
            elif canonical == "advance":
                pending_transition = "advance"

    return step_info


# ---------------------------------------------------------------------------
# Hindsight prompt construction
# ---------------------------------------------------------------------------

def build_hindsight_info(
    episode: dict,
    current_subtask: str = "planning",
    current_subtask_score: float = -1.0,
) -> str:
    frozen = episode.get("frozen_scores", {})
    plan = episode.get("plan") or frozen
    plan_count = max(len(plan), 1)
    scored_count = len(frozen)
    scores_summary = ", ".join(f"{k}={v:.3f}" for k, v in frozen.items()) or "none"
    subtask_score_str = f"{current_subtask_score:.3f}" if current_subtask_score >= 0 else "n/a"
    return HINDSIGHT_TEMPLATE.format(
        reward=episode["reward"],
        phase=episode.get("phase", "?"),
        plan_score=episode.get("plan_score", 0),
        scores_summary=scores_summary,
        scored_count=scored_count,
        plan_count=plan_count,
        current_subtask=current_subtask,
        current_subtask_score=subtask_score_str,
    )


def inject_hindsight(messages: list[dict], hindsight_info: str) -> list[dict]:
    """Clone messages and append hindsight info to the first user/system message."""
    if not messages:
        return messages
    out = list(messages)
    first = dict(out[0])
    first["content"] = first.get("content", "") + "\n\n" + hindsight_info
    out[0] = first
    return out


# ---------------------------------------------------------------------------
# API scoring
# ---------------------------------------------------------------------------

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 5.0


def _build_prompt_pair(
    tokenizer: Any,
    prefix_messages: list[dict],
    action_message: dict,
    hindsight_info: str,
    max_context: int,
) -> tuple[str, int, int] | None:
    """Build the full prompt text and compute prefix/action token spans.

    Returns (prompt_text, prefix_len, action_len) or None if the action
    is empty.  Truncates the prefix to stay within *max_context*.
    """
    hind_prefix = inject_hindsight(
        normalize_messages(prefix_messages), hindsight_info,
    )
    action_msg = normalize_message_for_template(action_message)

    full_text = safe_apply_chat_template(
        tokenizer, hind_prefix + [action_msg], add_generation_prompt=False,
    )
    prefix_text = safe_apply_chat_template(
        tokenizer, hind_prefix, add_generation_prompt=True,
    )

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_len = len(prefix_ids)
    action_len = len(full_ids) - prefix_len

    if action_len <= 0:
        return None

    if len(full_ids) > max_context:
        action_ids = full_ids[prefix_len:]
        max_prefix_tokens = max_context - len(action_ids)
        if max_prefix_tokens <= 0:
            logger.warning(
                "Action too long (%d tokens, limit %d). Keeping only action suffix.",
                len(action_ids), max_context,
            )
            kept_action_ids = action_ids[-max_context:]
            full_text = tokenizer.decode(kept_action_ids)
            return full_text, 0, len(kept_action_ids)

        anchor_text = safe_apply_chat_template(
            tokenizer, hind_prefix[:1], add_generation_prompt=False,
        ) if hind_prefix else ""
        marker_text = (
            "\n\n[... earlier trajectory context truncated; "
            "hindsight outcome preserved above ...]\n\n"
        )
        anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
        marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)

        # Keep the outcome-bearing first message plus the most recent prefix
        # tail.  HCAPO scoring needs the hindsight anchor more than old tool
        # chatter from the middle of a long trajectory.
        tail_budget = max_prefix_tokens - len(anchor_ids) - len(marker_ids)
        if tail_budget > 0:
            tail_ids = prefix_ids[-tail_budget:]
            trimmed_prefix_ids = anchor_ids + marker_ids + tail_ids
        else:
            anchor_budget = max(max_prefix_tokens - len(marker_ids), 0)
            trimmed_prefix_ids = anchor_ids[:anchor_budget] + marker_ids
            trimmed_prefix_ids = trimmed_prefix_ids[:max_prefix_tokens]

        prefix_text = tokenizer.decode(trimmed_prefix_ids)
        action_text = tokenizer.decode(action_ids)
        full_text = prefix_text + action_text
        final_prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        final_full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prefix_len = len(final_prefix_ids)
        action_len = len(final_full_ids) - prefix_len
        tokens_dropped = len(full_ids) - len(final_full_ids)
        logger.warning(
            "Prompt too long (%d tokens, limit %d). "
            "Kept hindsight anchor + recent prefix tail; dropped ~%d tokens.",
            len(full_ids), max_context, tokens_dropped,
        )

    return full_text, prefix_len, action_len


def _is_retryable(status_code: int = 0, error_text: str = "") -> bool:
    if status_code in (500, 502, 503, 504, 204):
        return True
    lower = error_text.lower()
    return any(
        tok in lower
        for tok in ("oom", "out of memory", "overloaded",
                    "resource exhausted", "timeout", "timed out",
                    "connection", "no content")
    )


async def score_step_logprobs(
    http_client: httpx.AsyncClient,
    generate_url: str,
    model: str,
    tokenizer: Any,
    prefix_messages: list[dict],
    action_message: dict,
    hindsight_info: str,
    semaphore: asyncio.Semaphore,
    max_context: int = 32768,
    max_logprob_tokens: int = 2048,
) -> dict[str, Any]:
    """Score one assistant action's log-probabilities with hindsight context.

    Uses SGLang's native ``/generate`` endpoint with ``logprob_start_len``
    so that logits are only materialised for a bounded suffix of the
    action tokens, not the entire prompt/action.  SGLang materialises a
    ``scored_tokens x vocab_size`` logits tensor for returned logprobs,
    so long tool-heavy actions must be sampled instead of scored fully.
    """
    async with semaphore:
        pair = _build_prompt_pair(
            tokenizer, prefix_messages, action_message,
            hindsight_info, max_context,
        )
        if pair is None:
            return {"mean_logprob": 0.0, "action_token_count": 0, "skipped": "empty_action"}

        full_text, prefix_len, action_len = pair
        if max_logprob_tokens > 0:
            scored_action_len = min(action_len, max_logprob_tokens)
        else:
            scored_action_len = action_len
        skipped_action_tokens = action_len - scored_action_len
        logprob_start_len = prefix_len + skipped_action_tokens

        payload = {
            "text": full_text,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
        }

        last_err: str = ""
        data: dict = {}
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await http_client.post(
                    generate_url, json=payload, timeout=180.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    break
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if not _is_retryable(resp.status_code) or attempt == _MAX_RETRIES - 1:
                    return {
                        "mean_logprob": 0.0,
                        "action_token_count": scored_action_len,
                        "total_action_tokens": action_len,
                        "skipped_action_tokens": skipped_action_tokens,
                        "error": last_err,
                    }
            except Exception as exc:
                last_err = str(exc)
                if not _is_retryable(error_text=last_err) or attempt == _MAX_RETRIES - 1:
                    return {
                        "mean_logprob": 0.0,
                        "action_token_count": scored_action_len,
                        "total_action_tokens": action_len,
                        "skipped_action_tokens": skipped_action_tokens,
                        "error": last_err,
                    }
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "  Server error (attempt %d/%d), retrying in %.0fs: %s",
                attempt + 1, _MAX_RETRIES, delay, last_err[:120],
            )
            await asyncio.sleep(delay)
        else:
            return {"mean_logprob": 0.0, "action_token_count": action_len, "error": last_err}

    meta = data.get("meta_info", {})
    input_lps = meta.get("input_token_logprobs", [])

    if not input_lps:
        return {
            "mean_logprob": 0.0,
            "action_token_count": scored_action_len,
            "total_action_tokens": action_len,
            "skipped_action_tokens": skipped_action_tokens,
            "error": "no_logprobs",
        }

    valid: list[float] = []
    for entry in input_lps:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] is not None:
            valid.append(float(entry[0]))
        elif isinstance(entry, (int, float)) and entry is not None:
            valid.append(float(entry))
        elif isinstance(entry, dict):
            lp = entry.get("logprob")
            if lp is not None:
                valid.append(float(lp))

    if not valid:
        return {
            "mean_logprob": 0.0,
            "action_token_count": scored_action_len,
            "total_action_tokens": action_len,
            "skipped_action_tokens": skipped_action_tokens,
            "error": "all_none",
        }

    mean_lp = sum(valid) / len(valid)
    return {
        "mean_logprob": mean_lp,
        "action_token_count": len(valid),
        "total_action_tokens": action_len,
        "skipped_action_tokens": skipped_action_tokens,
        "logprob_start_len": logprob_start_len,
    }


# ---------------------------------------------------------------------------
# Episode-level scoring
# ---------------------------------------------------------------------------

def identify_assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


async def score_episode(
    http_client: httpx.AsyncClient,
    generate_url: str,
    model: str,
    tokenizer: Any,
    episode: dict,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> list[dict]:
    messages = episode["messages"]
    assistant_indices = identify_assistant_indices(messages)
    step_subtask_info = map_steps_to_subtasks(messages, episode)
    total = len(assistant_indices)
    batch_size = getattr(args, "batch_size", 4) or total

    steps: list[dict] = []
    t0 = time.monotonic()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = assistant_indices[batch_start:batch_end]

        coros = []
        for step_idx_offset, msg_idx in enumerate(batch_indices):
            step_idx = batch_start + step_idx_offset
            prefix = messages[:msg_idx]
            action = messages[msg_idx]
            si = step_subtask_info[step_idx] if step_idx < len(step_subtask_info) else {}
            hindsight_info = build_hindsight_info(
                episode,
                current_subtask=si.get("subtask_id") or si.get("phase", "planning"),
                current_subtask_score=si.get("subtask_reward", -1.0),
            )
            coros.append(
                score_step_logprobs(
                    http_client, generate_url, model, tokenizer, prefix, action,
                    hindsight_info, semaphore, max_context=args.max_context,
                    max_logprob_tokens=args.max_logprob_tokens,
                )
            )

        results = await asyncio.gather(*coros, return_exceptions=True)

        for step_idx_offset, (msg_idx, res) in enumerate(zip(batch_indices, results)):
            step_idx = batch_start + step_idx_offset
            si = step_subtask_info[step_idx] if step_idx < len(step_subtask_info) else {}
            if isinstance(res, BaseException):
                logger.warning("Episode %s step %d failed: %s", episode["episode_id"], step_idx, res)
                entry = {"step_index": step_idx, "message_index": msg_idx, "error": str(res), "mean_logprob": 0.0}
            else:
                entry = dict(res)
                entry["step_index"] = step_idx
                entry["message_index"] = msg_idx
            entry["subtask_id"] = si.get("subtask_id")
            entry["subtask_reward"] = si.get("subtask_reward", 0.0)
            entry["phase"] = si.get("phase", "unknown")
            steps.append(entry)

        elapsed = time.monotonic() - t0
        logger.info(
            "  Episode %s: %d/%d steps scored (%.1fs elapsed)",
            episode["episode_id"], len(steps), total, elapsed,
        )

    return steps


# ---------------------------------------------------------------------------
# Post-processing: rho, Q_H, temporal smoothing (Eq. 5-7 + Appendix A)
# ---------------------------------------------------------------------------

def compute_ratios_and_qh(
    steps: list[dict],
    episode_reward: float,
    *,
    t_temp: float = 5.0,
    gamma: float = 0.95,
    c_min: float = 0.8,
    c_max: float = 1.2,
    alpha: float = 0.5,
    smooth: bool = True,
    use_dense_rewards: bool = True,
) -> list[dict]:
    """Compute importance ratios and Q_H values (Eq. 5-7).

    When *use_dense_rewards* is True each step uses its per-subtask
    frozen_score (stored in step["subtask_reward"]) instead of the single
    terminal episode_reward.  This gives the model a denser credit signal
    for long-horizon tasks.
    """
    T = len(steps)
    if T == 0:
        return steps

    # Eq. 6: pi_hind(a_t) = exp(mean_logprob / T_temp)
    for s in steps:
        mlp = s.get("mean_logprob", 0.0)
        s["pi_hind"] = math.exp(mlp / t_temp) if t_temp > 0 else math.exp(mlp)

    # Eq. 7 denominator: intra-trajectory mean
    pi_values = [s["pi_hind"] for s in steps]
    pi_mean = sum(pi_values) / len(pi_values) if pi_values else 1.0
    if pi_mean == 0:
        pi_mean = 1e-12

    # Group steps by subtask so discount is relative to subtask boundaries
    subtask_groups: dict[str, list[int]] = {}
    for t, s in enumerate(steps):
        key = s.get("subtask_id") or s.get("phase", "planning")
        subtask_groups.setdefault(key, []).append(t)

    for t, s in enumerate(steps):
        # Eq. 7: importance ratio
        raw_rho = s["pi_hind"] / pi_mean
        s["rho"] = max(c_min, min(c_max, raw_rho))

        if use_dense_rewards:
            r_t = s.get("subtask_reward", episode_reward)
            key = s.get("subtask_id") or s.get("phase", "planning")
            group = subtask_groups.get(key, [t])
            group_end = max(group)
            discount = gamma ** (group_end - t)
        else:
            r_t = episode_reward
            discount = gamma ** (T - 1 - t)

        s["q_h"] = s["rho"] * discount * r_t

    # Appendix A: temporal smoothing
    if smooth and T > 1:
        for t in range(T - 2, -1, -1):
            steps[t]["q_h_smoothed"] = (
                alpha * steps[t]["q_h"]
                + (1 - alpha) * steps[t + 1].get("q_h_smoothed", steps[t + 1]["q_h"])
            )
        steps[T - 1]["q_h_smoothed"] = steps[T - 1]["q_h"]
    else:
        for s in steps:
            s["q_h_smoothed"] = s["q_h"]

    return steps


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_episode_scores(
    episode_dir: Path,
    episode: dict,
    steps: list[dict],
    hyperparams: dict,
) -> None:
    pi_values = [s.get("pi_hind", 0) for s in steps]
    subtask_rewards = [s.get("subtask_reward", 0) for s in steps]
    unique_subtasks = {s.get("subtask_id") or s.get("phase", "?") for s in steps}
    output = {
        "episode_id": episode["episode_id"],
        "reward": episode["reward"],
        "frozen_scores": episode.get("frozen_scores", {}),
        "dense_rewards_used": True,
        "num_steps": len(steps),
        "num_subtasks_covered": len(unique_subtasks),
        "subtask_reward_range": [min(subtask_rewards), max(subtask_rewards)] if subtask_rewards else [0, 0],
        "steps": steps,
        "pi_hind_mean": sum(pi_values) / len(pi_values) if pi_values else 0,
        "hyperparams": hyperparams,
    }
    out_path = episode_dir / "hindsight_scores.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(
        "  Saved %d step scores → %s (pi_hind range: %.4f–%.4f, subtask_reward range: %.4f–%.4f)",
        len(steps), out_path,
        min(pi_values) if pi_values else 0,
        max(pi_values) if pi_values else 0,
        min(subtask_rewards) if subtask_rewards else 0,
        max(subtask_rewards) if subtask_rewards else 0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute HCAPO hindsight scores via SGLang /generate API",
    )
    parser.add_argument("--input-dir", default="trajectories", help="Trajectories directory")
    parser.add_argument("--api-base", default=os.environ.get("FSWE_AGENT_API_URL", ""), help="OpenAI-compat base URL")
    parser.add_argument("--model", default=os.environ.get("FSWE_AGENT_MODEL", ""), help="Model name for API calls")
    parser.add_argument("--api-key", default=os.environ.get("FSWE_AGENT_API_KEY", "unused"), help="API key")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer name (defaults to --model)")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Skip episodes below this reward")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent API calls (keep low to avoid server OOM)")
    parser.add_argument("--batch-size", type=int, default=4, help="Steps to batch per episode (limits client-side memory)")
    parser.add_argument("--max-context", type=int, default=32768, help="Max tokens per API call (truncates prefix beyond this)")
    parser.add_argument(
        "--max-logprob-tokens",
        type=int,
        default=2048,
        help=(
            "Max action tokens to request logprobs for per step. "
            "Scores the action suffix; use <=0 to score the full action."
        ),
    )

    parser.add_argument("--t-temp", type=float, default=5.0, help="Sharpening temperature T_temp (Eq. 6)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (Eq. 5)")
    parser.add_argument("--c-min", type=float, default=0.8, help="Lower clipping bound for rho (Eq. 7)")
    parser.add_argument("--c-max", type=float, default=1.2, help="Upper clipping bound for rho (Eq. 7)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Temporal smoothing factor (Appendix A)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing")
    parser.add_argument(
        "--no-dense-rewards", action="store_true",
        help="Use single episode reward instead of per-subtask frozen_scores",
    )

    parser.add_argument("--overwrite", action="store_true", help="Re-score episodes that already have scores")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be scored without calling API")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # Load episodes
    episodes: list[tuple[Path, dict]] = []
    for ep_dir in sorted(input_dir.glob("episode_*")):
        ep = load_episode(ep_dir, include_thinking=True, max_tool_result_chars=4000)
        if ep is None:
            continue
        if ep["reward"] < args.min_reward:
            continue
        if not args.overwrite and (ep_dir / "hindsight_scores.json").exists():
            logger.info("  Episode %s: already scored, skipping", ep["episode_id"])
            continue
        episodes.append((ep_dir, ep))

    logger.info("Scoring %d episodes (min_reward=%.2f)", len(episodes), args.min_reward)

    if args.dry_run:
        for ep_dir, ep in episodes:
            n_steps = len(identify_assistant_indices(ep["messages"]))
            subtask_info = map_steps_to_subtasks(ep["messages"], ep)
            subtask_summary = {}
            for si in subtask_info:
                key = si.get("subtask_id") or si.get("phase", "?")
                subtask_summary[key] = subtask_summary.get(key, 0) + 1
            frozen = ep.get("frozen_scores", {})
            logger.info(
                "  [DRY RUN] Episode %s: reward=%.4f, %d steps, subtask_steps=%s, frozen_scores=%s",
                ep["episode_id"], ep["reward"], n_steps,
                dict(subtask_summary),
                {k: f"{v:.3f}" for k, v in frozen.items()} if frozen else "none",
            )
        logger.info("Dry run complete — %d episodes, no API calls made.", len(episodes))
        return

    if not args.api_base or not args.model:
        logger.error("--api-base and --model are required (or set FSWE_AGENT_API_URL / FSWE_AGENT_MODEL)")
        sys.exit(1)

    # Load tokenizer
    tok_name = args.tokenizer or args.model
    logger.info("Loading tokenizer: %s", tok_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

    use_dense = not args.no_dense_rewards
    hyperparams = {
        "t_temp": args.t_temp,
        "gamma": args.gamma,
        "c_min": args.c_min,
        "c_max": args.c_max,
        "alpha": args.alpha,
        "smooth": not args.no_smooth,
        "dense_rewards": use_dense,
        "max_logprob_tokens": args.max_logprob_tokens,
    }

    base = args.api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    generate_url = base + "/generate"
    logger.info("Using SGLang native endpoint: %s", generate_url)

    headers: dict[str, str] = {}
    if args.api_key and args.api_key != "unused":
        headers["Authorization"] = f"Bearer {args.api_key}"

    http_client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(300.0))
    semaphore = asyncio.Semaphore(args.concurrency)

    try:
        for ep_dir, ep in episodes:
            logger.info(
                "Scoring episode %s (reward=%.4f, %d messages)...",
                ep["episode_id"], ep["reward"], len(ep["messages"]),
            )
            raw_steps = await score_episode(
                http_client, generate_url, args.model, tokenizer,
                ep, semaphore, args,
            )

            steps = compute_ratios_and_qh(
                raw_steps,
                episode_reward=ep["reward"],
                t_temp=args.t_temp,
                gamma=args.gamma,
                c_min=args.c_min,
                c_max=args.c_max,
                alpha=args.alpha,
                smooth=not args.no_smooth,
                use_dense_rewards=use_dense,
            )

            save_episode_scores(ep_dir, ep, steps, hyperparams)

        logger.info("Done — scored %d episodes.", len(episodes))
    finally:
        await http_client.aclose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
