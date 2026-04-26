# Building long-horizon SWE environments on Hugging Face: Frontier SWE × OpenEnv

**By the-thing**: we packaged and adapted 4 [FrontierSWE](https://www.frontierswe.com/) tasks as [OpenEnv](https://github.com/rycerzes/OpenEnv)-shaped services, pushed them to **Hugging Face Spaces**, and ran an **offline RL-style** training loop with public **datasets**, **Trackio** metrics, and a trainer Space.

---

## TL;DR

- **Four Dockerized environments** (notebook compression, Postgres wire adapter on SQLite, dependent type checker, libexpat → x86-64 asm) with a **shared Gym-style API** and **MCP** tools for planning and submission.
- **Custom harness adapter** built on top of OpenEnv harness work ([meta-pytorch/OpenEnv PR #389](https://github.com/meta-pytorch/OpenEnv/pull/389) and RFC005), then forked and extended in [`rycerzes/OpenEnv` on `feature/pi-harness-adapter`](https://github.com/rycerzes/OpenEnv/commits/feature/pi-harness-adapter/).
- **Composite rubric**: gates → L1 (tests / `reward.json` / regex ratios) → optional LLM layers → **episode reward** you can log and filter on for training.
- **Offline pipeline**: trajectories on the Hub → hindsight scoring (SGLang) → HCAPO-style dataset → **LoRA fine-tune** on a GPU Space, with **Trackio** curves for loss, LR, and gradient norms.

**Try it:** [frontier-swe-postgres](https://huggingface.co/spaces/rycerzes/frontier-swe-postgres) · [frontier-swe-notebook](https://huggingface.co/spaces/rycerzes/frontier-swe-notebook) · [frontier-swe-type-checker](https://huggingface.co/spaces/rycerzes/frontier-swe-type-checker) · [frontier-swe-libexpat-to-x86asm](https://huggingface.co/spaces/rycerzes/frontier-swe-libexpat-to-x86asm) · [source on GitHub](https://github.com/3xcaffeine/frontier-swe-openenv)

---

## 1. Environment innovation - why this setup is hard (and worth it)

Classic coding benchmarks often score a single patch. **Long-horizon software engineering** is different: the agent has to **plan**, **edit a real workspace**, **call tools**, and **submit** work over many steps-closer to how people ship systems than to a one-shot fix.

**What we built on top of that idea**

We did not reinvent the underlying FrontierSWE task specs; we **re-homed** them inside a **uniform environment contract**:

That includes a **custom harness adapter** layer we built on top of [meta-pytorch/OpenEnv PR #389](https://github.com/meta-pytorch/OpenEnv/pull/389) and RFC005, then maintained and updated in our fork: [`rycerzes/OpenEnv` `feature/pi-harness-adapter`](https://github.com/rycerzes/OpenEnv/tree/feature/pi-harness-adapter/).

| Piece | What it does for the agent |
| --- | --- |
| **HTTP control** | `reset` / `step` / `state` / `health` - same shape every task, so harnesses and demos do not fork per domain. Maintaining the `openenv` specs |
| **MCP tools** | `submit_plan`, `submit_subtask`, `get_status`, `advance` - forces **explicit decomposition** and **scored subtasks**, not a single anonymous blob of edits. |
| **Multi-layer rubric** | **Gates** catch broken builds or missing artifacts early; **L1** is task-native (wire compat tests, notebook round-trips, type-checker scores, assembly benchmarks); **L2/L3** optionally add LLM code and plan review when grader env vars are set; **episode reward** blends plan quality, frozen subtask scores, completion, and tool usage. |

That combination is deliberately **stressful** in a good way: the agent must **coordinate** (plan → execute → advance), **respect verifier reality** (hidden tests, anti-cheat), and **earn** a dense scalar at the end of an episode that can run on the order of **45–90+ minutes** per run-so the environment is **challenging**, **creative** in how it composes rubrics, and **meaningful** for measuring behavior beyond single-turn chat.

---

## 2. The problem, the box, and what the agent actually does

**Problem.** Training or evaluating agents on real long-horizon SWE needs a **repeatable service**: same ports, same instructions, same scoring, same tool surface-whether you run locally, in CI, or on the Hub.

**Our box.** **frontier-swe-openenv** is a small monorepo: `tasks/<task-id>/` holds instructions and verifiers (what “correct” means operationally); `frontier_swe_env/` holds the **FastAPI** server, shared rubrics, and **TaskConfig** (how to invoke those verifiers inside the image); `spaces/` holds thin **Space** definitions synced from `main` after images build.

**Agent behavior (easy to follow for a demo).**

1. Connect (WebSocket client or baseline script).
2. `reset` → read observation / phase.
3. Loop: natural language or tool use → `step` → optional MCP calls to **submit a plan**, run **L1+L2** on a **subtask**, **advance** when satisfied.
4. Episode ends with a **terminal episode reward** and subtask history you can log.

For a **concrete walkthrough without writing your own client**, the repo ships [`scripts/run_baseline.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/run_baseline.py): point it at `http://localhost:8000` with a task container running, and you get a full **reset → step** episode over the wire-good for recordings and “here is one turn of the loop” explanations.

---

## 3. Observable training progress - rewards, curves

Long episodes make **online** RL on the live env impractical at scale, so we invested in **offline** learning: **collect once**, **score offline**, **fine-tune**, **log everything**.

**Public artifacts (HF-native story)**

| Artifact | Link | Role in the demo |
| --- | --- | --- |
| Raw trajectories (pg-01, Qwen 3.6 27B) | [`rycerzes/fswe-pg-01-traj-q36-27b`](https://huggingface.co/datasets/rycerzes/fswe-pg-01-traj-q36-27b) | Shows **what** we logged per episode (`result.json`, sessions, logs, hindsight when present). |
| HCAPO training JSONL | [`rycerzes/fswe-hcapo-pg-01-trajectories`](https://huggingface.co/datasets/rycerzes/fswe-hcapo-pg-01-trajectories) | **Step-level advantages** paired with messages for supervised fine-tuning. |
| Trackio dashboard | [`rycerzes/trackio`](https://huggingface.co/spaces/rycerzes/trackio) | **Observable** loss, epoch, learning rate, gradient norm, global step. |

On a **3 epoch / ~18 optimizer step** reference run (Space-backed trainer), the root README documents what we see in Trackio: **loss** trending down on the order of **~25%** over the plotted window (smoothed), **epoch** progressing toward **~2.7**, **LR** warmup-then-decay, **gradient norms** staying in a moderate band-i.e. a **sanity fine-tune** where optimization looks stable, not a mystery box.

We also ship a **static dashboard figure** in-repo for slides and blog embeds: [`assets/training-trackio-dashboard.png`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/assets/training-trackio-dashboard.png).

**Before / after.** The cleanest **before/after** we surface in tooling today is **training loss and optimization metrics** on the HCAPO dataset, plus **episode-level rewards inside collected trajectories** for analysis. A live **A/B rollout score** on the full Docker env after LoRA is the natural next chapter for the demo-and the pipeline is set up so you can **regenerate trajectories** with the adapted policy and compare distributions. For hackathon judging, the **curves + public datasets + reproducible launch script** are the evidence chain we stand behind *right now*.

---

## 4. Reward logic and training pipeline - coherent signal end to end

**Episode reward (macro).** The scalar \(R\) matches [`EpisodeRubric`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/frontier_swe_env/rubrics/episode_rubric.py): weighted **plan score**, mean **frozen subtask** scores, **completion**, and **tool density**-clipped into **[0, 1]** for filtering (e.g. `--min-reward 0.05` in the dataset builder).

**L1 (micro, task-specific).** Each task implements its own verifier output: **regex ratio** on test totals (Postgres), **`reward_json`** fields (notebook), or **`reward_json_score`** with anchors (type checker, libexpat). Same server code paths; different physics.

**Training path (why it should move policy behavior).**

1. [`collect_trajectories.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/collect_trajectories.py) - rollouts into `trajectories/episode_NNN/`.
2. [`backfill_rewards.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/backfill_rewards.py) - repair missing `episode_reward` when needed.
3. [`compute_hindsight_scores.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/compute_hindsight_scores.py) - SGLang `/generate` with bounded logprob windows (memory-safe), MCP-aware **step → subtask** mapping, hindsight \(Q^H\) and smoothing.
4. [`build_hcapo_dataset.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/build_hcapo_dataset.py) - GRPO-style macro advantages + normalized hindsight micro advantages → **JSONL** with **per-step weights**.
5. [`train_hcapo.py`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/training/train_hcapo.py) + [`launch_hf_space.sh`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/scripts/launch_hf_space.sh) - **weighted CE on assistant tokens** (chunked forward for large models), Trackio reporting.

Coherent design is means that environment reward defines **which episodes matter**; hindsight defines **which tokens inside those episodes** get gradient; the trainer respects **assistant masks** and **step weights** so the update is not “one scalar smeared across the whole transcript.” Details and equations live in [`training/README.md`](https://github.com/3xcaffeine/frontier-swe-openenv/blob/main/training/README.md)

---

## Where to go next

- **Run a Space** from the TL;DR links and narrate **one** subtask submission end to end.
- **Open Trackio** to the named run and zoom the **loss / LR** panel while you talk through the pipeline slide.
- **Clone the repo**, `uv sync`, and use **`./scripts/launch_hf_space.sh`** when you want the full HF training path on your own account.

