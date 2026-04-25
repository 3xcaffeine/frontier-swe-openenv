ARG FIRST_PARTY_CLI_BASE_IMAGE=ghcr.io/proximal-labs/frontier-swe/first-party-cli-base-python3.11-slim-bookworm:firstparty-cli-20260416-v2
FROM ${FIRST_PARTY_CLI_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TASK_BUDGET_SECS=28800
ENV UV_LINK_MODE=copy
ENV NVM_DIR=/root/.nvm
ENV PATH=/root/.local/bin:/usr/local/bin:${PATH}

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget tmux jq htop vim unzip procps \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel uv

# Install Tinker SDK + cookbook (the RL training framework — GPU work is remote)
RUN uv pip install --system \
    tinker==0.16.1 \
    tinker-cookbook==0.2.2

# Install common Python deps the agent will likely need
RUN uv pip install --system \
    numpy>=1.26 \
    scipy>=1.11 \
    pandas>=2.1 \
    matplotlib>=3.8 \
    transformers>=4.40 \
    tokenizers>=0.19

# Belt-and-suspenders: disable Codex web search even if CLI flags drift.
RUN mkdir -p /etc/codex \
    && printf 'allowed_web_search_modes = ["disabled"]\nforced_login_method = "api"\n' > /etc/codex/requirements.toml

WORKDIR /app

# Copy workspace files (prepare.py, train.py, etc.)
COPY workspace/ /app/

# Download Qwen3-8B tokenizer from HuggingFace at build time
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B').save_pretrained('/app/qwen3-8b-tokenizer')"

# No pre-generated boards provided — agent generates everything from the game spec

# Make timer and entrypoint executable (baked into image, agent cannot modify)
RUN chmod +x /app/entrypoint.sh /app/timer.sh

# Create output directories (agent generates all boards from scratch)
RUN mkdir -p /app/checkpoint /app/boards

# Timer daemon — two start mechanisms for robustness:
# 1. ENTRYPOINT: runs on container start (works in Docker, may work in Modal)
# 2. BASH_ENV fallback: runs on first exec() call (guaranteed in Modal)
# Timer startup explicitly clears shell startup hooks to avoid recursive
# re-entry through bash shebangs.
# Agent reads /app/.timer/remaining_secs to check time left.
RUN cat >/etc/profile.d/frontier-task-init.sh <<'EOF'
if [ -x /app/timer.sh ] && [ "${FRONTIER_TIMER_BOOTSTRAP:-0}" != "1" ]; then
  timer_pid_file=/app/.timer/timer.pid
  if [ ! -s "$timer_pid_file" ] || ! kill -0 "$(cat "$timer_pid_file" 2>/dev/null)" 2>/dev/null; then
    FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh >/dev/null 2>&1 &
  fi
fi
EOF
ENV BASH_ENV=/etc/profile.d/frontier-task-init.sh

ENTRYPOINT ["/app/entrypoint.sh"]
# Harbor/Modal creates the sandbox first, then execs into it later. The default
# command therefore needs to keep the container alive instead of starting a
# shell that can immediately exit and race Harbor's first filesystem call.
CMD ["tail", "-f", "/dev/null"]
