#!/usr/bin/env bash
set -euo pipefail


# launch_hf_job.sh — Upload DPO dataset & launch training on HF Jobs
#
# Prerequisites:
#   1. `hf` CLI installed  (curl -LsSf https://hf.co/cli/install.sh | bash)
#   2. HF_TOKEN set in .env or environment
#   3. datasets/dpo_train.jsonl exists (from build_training_dataset.py)
#
# Usage:
#   ./scripts/launch_hf_job.sh                  # defaults (a100-large, 16k ctx)
#   ./scripts/launch_hf_job.sh --max-seq 32768  # 32k context
#   ./scripts/launch_hf_job.sh --dry-run        # print command without running


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load HF_TOKEN from .env if not already set
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$PROJECT_ROOT/.env" ]]; then
    HF_TOKEN=$(grep -m1 '^HF_TOKEN=' "$PROJECT_ROOT/.env" | cut -d= -f2-)
    export HF_TOKEN
fi

# ---- Defaults (override with env vars or flags) ----
HF_USERNAME="${HF_USERNAME:-}"
DATASET_REPO="${DATASET_REPO:-}"
OUTPUT_REPO="${OUTPUT_REPO:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
MAX_SEQ="${MAX_SEQ:-16384}"
FLAVOR="${FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-4h}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --username)     HF_USERNAME="$2";  shift 2 ;;
        --dataset-repo) DATASET_REPO="$2"; shift 2 ;;
        --output-repo)  OUTPUT_REPO="$2";  shift 2 ;;
        --model)        MODEL_NAME="$2";   shift 2 ;;
        --max-seq)      MAX_SEQ="$2";      shift 2 ;;
        --flavor)       FLAVOR="$2";       shift 2 ;;
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift   ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Resolve HF username via API using HF_TOKEN (no login required)
if [[ -z "$HF_USERNAME" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "ERROR: HF_TOKEN not set. Add it to .env or export it."
        exit 1
    fi
    HF_USERNAME=$(uv run python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null || true)
    if [[ -z "$HF_USERNAME" ]]; then
        echo "ERROR: Could not determine HF username from HF_TOKEN. Check your token."
        exit 1
    fi
fi

DATASET_REPO="${DATASET_REPO:-${HF_USERNAME}/frontier-swe-pg-task-001-dpo-trajectories}"
OUTPUT_REPO="${OUTPUT_REPO:-${HF_USERNAME}/frontier-swe-dpo-qwen36-27b}"

DATASET_FILE="$PROJECT_ROOT/datasets/dpo_train.jsonl"
if [[ ! -f "$DATASET_FILE" ]]; then
    echo "ERROR: Dataset not found at $DATASET_FILE"
    echo "Run 'python scripts/build_training_dataset.py' first."
    exit 1
fi

# ---- Step 1: Upload dataset to HF Hub ----
echo "==> Uploading dataset to $DATASET_REPO ..."
if [[ "$DRY_RUN" == "false" ]]; then
    uv run python -c "
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = '${DATASET_REPO}'

try:
    create_repo(repo_id, repo_type='dataset', exist_ok=True, private=True)
except Exception as e:
    print(f'Repo creation note: {e}')

api.upload_file(
    path_or_fileobj='${DATASET_FILE}',
    path_in_repo='dpo_train.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
)
print(f'Dataset uploaded to https://huggingface.co/datasets/{repo_id}')
"
else
    echo "  [DRY RUN] Would upload $DATASET_FILE -> $DATASET_REPO"
fi

# ---- Step 2: Submit HF Job ----
echo ""
echo "==> Submitting HF Job..."
echo "    Flavor:   $FLAVOR"
echo "    Model:    $MODEL_NAME"
echo "    Dataset:  $DATASET_REPO"
echo "    Output:   $OUTPUT_REPO"
echo "    Max Seq:  $MAX_SEQ"
echo "    Timeout:  $TIMEOUT"
echo ""

JOB_CMD=(
    hf jobs uv run "$SCRIPT_DIR/train_dpo_hfjob.py"
    --flavor "$FLAVOR"
    --timeout "$TIMEOUT"
    --secrets HF_TOKEN
    --env "HF_ENDPOINT=https://hf-mirror.com"
    --
    --dataset-id "$DATASET_REPO"
    --model-name "$MODEL_NAME"
    --output-repo "$OUTPUT_REPO"
    --max-seq-length "$MAX_SEQ"
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  ${JOB_CMD[*]}"
else
    echo "Launching..."
    "${JOB_CMD[@]}"
fi
