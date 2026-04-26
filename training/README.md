# HCAPO training pipeline

This document describes the **HCAPO-inspired** training flow used for Frontier SWE trajectory fine-tuning: how **episode rewards** are defined, how **hindsight** scores become **step advantages**, what the **training dataset** contains, and what **training / runtime** adjustments were made for **Qwen** models and **Hugging Face GPU** Spaces.

For a short end-to-end recipe (datasets on the Hub, Trackio, launch commands), see the **Training** section in the [root README](../README.md).

---

## Design rationale

### Why not online RL (e.g. GRPO on the live environment)?

Episodes often last on the order of **45–90+ minutes**. Online methods that need **many fresh rollouts per policy update** are **impractical**: orchestration, verifier time, and failures dominate before the optimiser sees enough data. We **collect trajectories once**, score them **offline**, build a **static** dataset, then fine-tune.

### Why not plain DPO or scalar reward-weighted SFT?

- **DPO** wants preference-style contrasts; our logs are **single** multi-turn trajectories with tools, not natural pairs per step.
- **Scalar reward-weighted SFT** applies **one weight per episode** and does not say **which assistant turns** helped. **HCAPO-style** credit assigns **macro** (trajectory) and **micro** (hindsight) signals per step.

### Relation to the [HCAPO paper](https://arxiv.org/abs/2603.08754) (2603.08754)

There is **no official end-to-end** public repo for the full paper stack (ALFWorld + WebShop + Search QA + multi-GPU online GRPO + generative verification). **Appendix B** of the [HTML version](https://arxiv.org/html/2603.08754v1) is essentially runnable pseudocode (rollouts, \(\pi_{\text{hind}}\), \(\rho_t\), composite advantage, PPO-style update). Helpful forks: [Awesome-GRPO](https://github.com/GITrans/Awesome-GRPO), [direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) (PPO/GRPO helpers).

| Paper (conceptual) | This repo |
| --- | --- |
| Online GRPO-style RL | **Offline** pipeline: [`collect_trajectories.py`](../scripts/collect_trajectories.py) → hindsight → [`build_hcapo_dataset.py`](../scripts/build_hcapo_dataset.py) → [`train_hcapo.py`](train_hcapo.py) |
| Terminal reward emphasis | **Dense** `plan_score` + `frozen_scores` in prompts and in \(Q^H\) when dense mode is on ([`compute_hindsight_scores.py`](../scripts/compute_hindsight_scores.py)) |
| Generic step alignment | **MCP tool boundaries**: [`map_steps_to_subtasks()`](../scripts/compute_hindsight_scores.py) unwraps outer `mcp` calls, parses `submit_plan` / `advance`, assigns **phase** and **subtask_id** |
| PPO-clipped policy gradient | **Step-weighted SFT**: combined advantages → JSONL → weighted CE in `HCAPOTrainer` |
| Generic logprob API | **SGLang** native `/generate`, `logprob_start_len`, bounded action scoring, retries ([`score_step_logprobs()`](../scripts/compute_hindsight_scores.py)) |

---

## Pipeline overview

1. **Collect trajectories** — [`scripts/collect_trajectories.py`](../scripts/collect_trajectories.py). Each `trajectories/episode_NNN/` holds `result.json`, `pi_session.jsonl`, logs, and later `hindsight_scores.json`.

2. **Backfill or read episode reward** — `result.json` stores final reward and subtask scores. If an episode does not reach `DONE`, [`scripts/backfill_rewards.py`](../scripts/backfill_rewards.py) (and collection-time logic in `collect_trajectories.py`) can fill **`episode_reward`** from captured state.

3. **Compute hindsight scores** — [`scripts/compute_hindsight_scores.py`](../scripts/compute_hindsight_scores.py) calls SGLang’s native **`/generate`** (via `httpx`) to score original assistant actions under hindsight context; writes **`hindsight_scores.json`**.

4. **Build and train** — [`scripts/build_hcapo_dataset.py`](../scripts/build_hcapo_dataset.py) merges trajectory-level advantages with step-level hindsight and writes `datasets/hcapo_train.jsonl`. [`train_hcapo.py`](train_hcapo.py) runs weighted SFT (Unsloth + TRL). [`launch_hf_space.sh`](../scripts/launch_hf_space.sh) wraps HF Space / dataset upload flows.

---

## Episode reward

The scalar **\(R\)** stored in trajectories and used by the dataset builder matches the **episode rubric** in code ([`EpisodeRubric.compute`](../frontier_swe_env/rubrics/episode_rubric.py)):

```text
R = plan_weight   * plan_score
  + subtask_weight * subtask_mean
  + completion_weight * completion
  + tool_weight   * tool_density
```

With default weights (`TaskConfig`): **0.25 / 0.60 / 0.10 / 0.05**:

```text
plan_count      = max(len(plan), 1)
subtask_mean    = mean(frozen subtask scores, padded with 0.0 to plan_count)
completion      = min(number_of_frozen_scores / plan_count, 1.0)
tool_density      = min(tool_call_count / (5 * plan_count), 1.0)
```

**\(R\)** is treated as lying in **[0, 1]** for reporting (and filtering with `--min-reward`).

Planning-only episodes can still get a small **\(R\)** via **`tool_density`**. Under **dense** hindsight scoring, steps often still carry **\(r_t = 0\)** until there is a nonzero **`plan_score`** or **`frozen_scores[subtask_id]`**, so they contribute little after advantage clipping.

---

## Step-to-subtask mapping

[`map_steps_to_subtasks()`](../scripts/compute_hindsight_scores.py) assigns each **assistant** message:

- **Planning** — until a **`submit_plan`** tool call succeeds (JSON tool response, no error prefix).
- **Executing** — after a successful plan; **`advance`** (on success) moves the current subtask index.

Per-step metadata includes:

```json
{
  "phase": "executing",
  "subtask_id": "S2",
  "subtask_reward": 0.13
}
```

**`subtask_reward`** is **`plan_score`** in planning, else **`frozen_scores[subtask_id]`** in executing.

**Outer `mcp` wrapper:** Pi/OpenEnv may emit tool calls under an outer function name `mcp` with nested JSON naming the real tool (e.g. `openenv_submit_plan`). [`_extract_effective_tool_names()`](../scripts/compute_hindsight_scores.py) unwraps that so transitions key off **`submit_plan`**, **`advance`**, etc.

---

## Hindsight prompt

For each assistant action, the scorer appends a block (see `HINDSIGHT_TEMPLATE` in [`compute_hindsight_scores.py`](../scripts/compute_hindsight_scores.py)) including:

```text
Final reward
Phase reached
Plan score
Subtask scores (summary)
Subtasks completed / plan count
Current subtask
Current subtask score
```

That text is **post-hoc** (not visible during the original rollout). The scoring model then receives a forward request whose labels are used only to read **input-token logprobs** for the **original** assistant tokens.

---

## Hindsight scoring via SGLang (`/generate`)

The script uses SGLang’s native **`POST .../generate`** with **`httpx.AsyncClient`**, not the OpenAI-compatible chat-completions path with `echo` + `logprobs` on the **full** prompt (which can force huge logits tensors and **OOM the server**).

Payload highlights:

```text
return_logprob     = true
logprob_start_len  = prefix_len + skipped_action_tokens
```

Here **`skipped_action_tokens`** trims the start of the **action** so only the last **`min(action_len, max_logprob_tokens)`** action tokens are scored—reducing work from roughly **`seq_len × vocab`** to **`max_logprob_tokens × vocab`** for the logprob slice.

**CLI defaults** (see argparse in [`compute_hindsight_scores.py`](../scripts/compute_hindsight_scores.py)):

```text
--concurrency        1
--max-context        32768
--max-logprob-tokens 2048    # increase (e.g. 4096) for longer actions if the server allows
--batch-size         4
```

**Retries:** exponential backoff on 500 / 502 / 503 / 504 / 204 and OOM-like error strings (`_MAX_RETRIES`, `_RETRY_BASE_DELAY`).

---

## Hindsight scoring formulae

Let **`mean_logprob_t`** be the mean log-probability of the **scored** action token suffix under the hindsight-augmented prefix.

```text
pi_hind_t = exp(mean_logprob_t / T_temp)     # default T_temp = 5.0
pi_mean   = mean_t(pi_hind_t)
rho_raw_t = pi_hind_t / pi_mean
rho_t     = clip(rho_raw_t, c_min, c_max)   # defaults 0.8, 1.2
```

**Dense rewards (default):**

```text
Q_H_t = rho_t * gamma^(group_end(t) - t) * r_t
```

- **`r_t`**: dense step reward (`subtask_reward` above).
- **`group_end(t)`**: last step index in the same **subtask id** (or planning phase bucket).

**Terminal fallback** (`--no-dense-rewards`):

```text
Q_H_t = rho_t * gamma^(T - 1 - t) * R
```

**Temporal smoothing** (`--alpha`, default `0.5`):

```text
Q_smooth_(T-1) = Q_H_(T-1)
Q_smooth_t       = alpha * Q_H_t + (1 - alpha) * Q_smooth_(t+1)   # backward pass
```

[`build_hcapo_dataset.py`](../scripts/build_hcapo_dataset.py) uses **`q_h_smoothed`** unless **`--no-smooth`**.

---

## HCAPO advantage construction

Episodes must pass **`--min-reward`** and contain **`hindsight_scores.json`**.

### Trajectory (macro) advantage

```text
A_grpo_i = (R_i - mean(R)) / std(R)
```

If **`std(R) == 0`**, the code uses **`1.0`** instead ([`compute_grpo_advantages()`](../scripts/build_hcapo_dataset.py)).

### Hindsight (micro) advantage

Over **all kept steps** in the batch:

```text
mu_h    = mean(q_h_smoothed_t)
sigma_h = std(q_h_smoothed_t)
A_micro_t = (q_h_smoothed_t - mu_h) / sigma_h
```

**Do-no-harm:** if **`A_grpo_i > 0`**, then **`A_micro_t ← max(A_micro_t, 0)`**.

### Combined advantage and JSONL weights

```text
A_hcapo_t = A_grpo_i + omega * A_micro_t          # default omega = 1.0
w_t_raw   = max(A_hcapo_t, 0)
w_t       = w_t_raw / mean(w_t_raw | w_t_raw > 0)
```

Rows where **all** **`w_t`** are zero are dropped.

---

## Dataset format

`datasets/hcapo_train.jsonl` — one JSON object per episode (example shape):

```json
{
  "messages": [...],
  "step_advantages": [1.23, 0.87, 1.45],
  "step_message_indices": [1, 4, 7],
  "_episode_id": 12,
  "_reward": 0.4058,
  "_grpo_advantage": 0.91,
  "_num_steps": 67
}
```

Example summary from a **pg-01** run (`hcapo_summary.json` after build):

```text
total_episodes_loaded = 20
episodes_in_dataset   = 14
total_steps           = 1414
nonzero_steps         = 1391
min_reward            = 0.05
omega                 = 1.0
use_smoothed          = true
```

(Exact counts depend on your local `trajectories/` and flags.)

---

## Training loss

**HCAPOTrainer** ([`train_hcapo.py`](train_hcapo.py)) applies **step-weighted** cross-entropy on **assistant** tokens only. Conceptually, for token position **`j`** belonging to assistant step **`t`**:

```text
CE_j            = cross_entropy(logits_j, label_j)
weighted_loss   = sum_j w_t(j) * CE_j / sum_j w_t(j) * mask_j
```

Only labels with supervision (and assistant spans) contribute; **`ignore_index = -100`** drops non-target positions. Long sequences are summed in **chunks** (e.g. 256 positions) inside **`compute_loss`** to cap peak memory.

---

## Training adjustments (Qwen, Unsloth, HF)

### Qwen 3.5 / 3.6 architecture and wrappers

Many Qwen 3.x checkpoints use **`Qwen3_5ForConditionalGeneration`**: a multimodal module tree that still includes **`language_model`** + **`lm_head`** for text. With **PEFT / Unsloth**, you often get:

```text
PeftModelForCausalLM
  └── LoraModel
        └── Qwen3_5ForConditionalGeneration
              ├── model (Qwen3_5Model)
              │     └── language_model  ← text backbone for loss
              └── lm_head
```

[`_get_backbone_and_lm_head()`](train_hcapo.py) unwraps **PeftModel → LoraModel → inner CausalLM**, then uses **`.model`** as the transformer backbone and follows **`.language_model`** when present so **`lm_head.in_features`** matches **hidden states**.

Reported sizes (for sanity checks):

```text
Qwen3.5-4B:   hidden_size = 2560,  vocab_size = 248320
Qwen3.6-27B: hidden_size = 5120,  vocab_size = 248320
```

[`_remove_qwen_vision_mappings()`](train_hcapo.py) strips vision-related **`auto_map`** entries so Unsloth does not treat a text-only checkpoint as a vision pipeline.

### Chat template and `assistant_masks`

Transformers only fills **`assistant_masks`** when the Jinja template wraps assistant generations with:

```jinja
{% generation %}
...
{% endgeneration %}
```

Qwen templates may omit this. The trainer **patches the tokenizer chat template in memory** (see [`_ensure_generation_chat_template()`](train_hcapo.py)) so **`apply_chat_template(..., return_assistant_tokens_mask=True)`** works in one pass—important for long Pi sessions.

### Pre-tokenization vs `formatting_func`

Unsloth’s SFT path often wants a **`formatting_func`** when there is no plain **`text`** column. We **pre-tokenize** rows to **`input_ids`** + **`assistant_masks`** + **`step_advantages`** so Unsloth can skip conversational re-formatting at train time. After that, **`assistant_only_loss`** is set **`False`** in **`SFTConfig`**; the **HCAPO collator** enforces assistant-only regions via masks.

### HCAPO data collator

[`_build_hcapo_data_collator()`](train_hcapo.py):

1. Strips metadata columns before the base collator runs.
2. Uses **`assistant_masks`** so non-assistant positions are **`ignore_index`**.
3. Finds contiguous **assistant label spans** in **`labels`**.
4. Assigns each span the corresponding **`step_advantages`** entry.
5. Adds **`step_weights`** to the batch for **`HCAPOTrainer`**.

If Unsloth swaps the collator during init, the trainer **re-applies** the HCAPO collator so **`step_weights`** are not dropped.

### Chunked backbone + `lm_head` projection

For **27B × long context**, a single **`model(**inputs)`** that returns full **`[batch, seq, vocab]`** logits can exceed **A100 80GB**. The custom **`compute_loss`** path:

1. Runs the **text backbone** with **`use_cache=False`**.
2. Drops the large activations that are not needed for the next chunk.
3. Applies **`lm_head`** in **chunks** (default width **256** tokens).
4. Accumulates weighted CE numerator and denominator across chunks.

Peak logits memory scales like **`O(chunk × vocab)`** instead of **`O(seq × vocab)`**.

### Liger

**`liger-kernel>=0.7.0`** is a project dependency. Fused kernels can still help **inside** transformer blocks during the backbone forward. The **custom** loss path does **not** call Liger’s fused CE for the final weighted loss (we need arbitrary **`step_weights`** per position).

### Adapter vs merged weights

Prefer saving the **LoRA adapter** (`save_merged_16bit: false` in config) to avoid multi‑tens‑of‑GB merged checkpoints. Load **base + adapter** at inference.

### No QLoRA for the A100 Qwen 3.6 recipe

The reference HF config keeps **`load_in_4bit: false`** for the 27B Space run so training stays on the **bf16 LoRA** path without 4-bit quant quirks on this stack.

---

## Configurations

Paths are wired in [`launch_hf_space.sh`](../scripts/launch_hf_space.sh) and copied in [`Dockerfile.train`](Dockerfile.train):

| File | Role |
| --- | --- |
| [`hcapo_config_4090_q35_4b.json`](hcapo_config_4090_q35_4b.json) | Local **4090** smoke: **`Qwen/Qwen3.5-4B`**, **`max_seq_length` 1024**, **`num_train_epochs` 1**, **`per_device_train_batch_size` 1**, **`gradient_accumulation_steps` 8**, **`warmup_steps` 5**, **`load_in_4bit` false**. |
| [`hcapo_config_a100_q36_27b.json`](hcapo_config_a100_q36_27b.json) | **A100** HF recipe: **`Qwen/Qwen3.6-27B`**, **`max_seq_length` 16384**, **`num_train_epochs` 3**, **`per_device_train_batch_size` 1**, **`gradient_accumulation_steps` 4**, **`warmup_steps` 2**, **`load_in_4bit` false**, **`save_merged_16bit` false**. |

**Step budget:** with **`per_device_train_batch_size = 1`** and **`gradient_accumulation_steps = 4`**, Hugging Face / TRL advance the optimiser roughly **`len(train_dataloader) // 4`** times per epoch (exact rounding depends on version and **`drop_last`**). For **~14** JSONL rows that is on the order of **three** updates per epoch, so **three epochs → ~nine** global steps unless **`--max-steps`** or a larger dataset changes the schedule. If Trackio shows a different total (e.g. **18**), compare the **`max_steps`** / dataset size / launch overrides for that run.

---

## HF Spaces behaviour

### Health check (port **7860**)

Spaces expect HTTP on **7860** within the startup window. [`Dockerfile.train`](Dockerfile.train) starts a tiny background server before training:

```bash
uv run python -m http.server 7860 &>/dev/null &
```

### Container lifecycle

Training should **not** `exec` into the trainer as **PID 1**: when the process exits, the container dies and the Space may restart. The image keeps **bash** as PID **1**, runs training, then **`sleep infinity`** so the Space stays up until you pause or delete it.

```bash
huggingface-cli space pause <user>/<space-name>
```

### Dependencies

Training extras live under **`[project.optional-dependencies] training`** in [`pyproject.toml`](../pyproject.toml). The training image installs with:

```text
uv sync --frozen --no-dev --extra training
```

### Naming (example)

| Artefact | Example id |
| --- | --- |
| Dataset repo | `fswe-hcapo-pg-01-trajectories` |
| Adapter output repo | `fswe-hcapo-pg-01-qwen36-27b` |
| Trackio Space | `<user>/fswe-hcapo-pg-01-monitor` |
| Trackio project | `fswe-hcapo-pg-01` |
| Run name | `fswe-hcapo-pg-01-qwen36-27b` |

Set **`report_to = trackio`**, **`TRACKIO_SPACE_ID`**, **`TRACKIO_PROJECT_NAME`**, and optionally the compatibility aliases **`TRACKIO_SPACE`**, **`TRACKIO_PROJECT`** (see [`train_hcapo.py`](train_hcapo.py) argparse / env handling).

---

## Typical commands

```bash
uv run python scripts/build_hcapo_dataset.py \
  --input-dir trajectories \
  --output-dir datasets \
  --min-reward 0.05 \
  --omega 1.0
```

```bash
./scripts/launch_hf_space.sh --upload-dataset
./scripts/launch_hf_space.sh --max-steps 1
./scripts/launch_hf_space.sh --with-dataset-upload --max-steps 1
./scripts/launch_hf_space.sh
./scripts/launch_hf_space.sh --delete
```

---

## Troubleshooting

### Planning-only episodes with reward **0.05**

Backfill / rubric can assign a small **\(R\)** via **`tool_density`**, but dense **`r_t`** on steps may stay **0** until a plan and subtask scores exist—little HCAPO signal after clipping.

### OOM on first training step

If failure is inside **`cross_entropy`** on full logits, ensure the **chunked backbone + `lm_head`** path is active (see **`HCAPOTrainer.compute_loss`**). Fallback: lower **`max_seq_length`**.

### `RuntimeError` … `lm_head` / hidden mismatch

Usually means the resolved “backbone” was still a **full CausalLM** instead of **`Qwen3_5TextModel`**. Check [`_get_backbone_and_lm_head()`](train_hcapo.py) unwrapping.

### SGLang OOM during hindsight

Avoid full-prompt logprob modes; keep **`/generate`** + **`logprob_start_len`** + a modest **`--max-logprob-tokens`**.

### Space killed before training finishes

Ensure the **7860** stub server is running and the main process is not **`exec`**’d as the only PID without a follow-up **`sleep`**.

### Wrong Trackio project

Verify **`REPORT_TO`**, **`TRACKIO_SPACE_ID`**, **`TRACKIO_PROJECT_NAME`**, **`RUN_NAME`**, and the **`TRACKIO_*`** aliases.

---

## File map

| Stage | Script / artefact |
| --- | --- |
| Collect | [`scripts/collect_trajectories.py`](../scripts/collect_trajectories.py) |
| Backfill reward | [`scripts/backfill_rewards.py`](../scripts/backfill_rewards.py) |
| Hindsight | [`scripts/compute_hindsight_scores.py`](../scripts/compute_hindsight_scores.py) |
| Build JSONL | [`scripts/build_hcapo_dataset.py`](../scripts/build_hcapo_dataset.py) |
| Train | [`training/train_hcapo.py`](train_hcapo.py) |
| HF Space | [`scripts/launch_hf_space.sh`](../scripts/launch_hf_space.sh), [`Dockerfile.train`](Dockerfile.train) |

---

## References

- HCAPO paper: [arXiv:2603.08754](https://arxiv.org/abs/2603.08754), [HTML + Appendix B](https://arxiv.org/html/2603.08754v1).
- Root README: [Training (offline RL)](../README.md#training-offline-rl).
