#!/usr/bin/env bash
set -euo pipefail


# launch_hf_job.sh — Launch HCAPO training on HF Jobs
#
# Prerequisites:
#   1. `hf` CLI installed  (curl -LsSf https://hf.co/cli/install.sh | bash)
#   2. HF_TOKEN set in .env or environment
#   3. datasets/hcapo_train.jsonl exists if using --upload-dataset
#
# Usage:
#   ./scripts/launch_hf_job.sh                  # defaults (a100-large, Qwen 3.6 27B)
#   ./scripts/launch_hf_job.sh --upload-dataset # upload dataset only
#   ./scripts/launch_hf_job.sh --with-dataset-upload # upload dataset, then launch
#   ./scripts/launch_hf_job.sh --with-dataset-upload --max-steps 1
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
HCAPO_CONFIG="${HCAPO_CONFIG:-training/hcapo_config_a100_q36_27b.json}"
FLAVOR="${FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-4h}"
RUN_NAME="${RUN_NAME:-fswe-hcapo-pg-01-qwen36-27b}"
MAX_STEPS="${MAX_STEPS:-}"
DATASET_FILE="${DATASET_FILE:-$PROJECT_ROOT/datasets/hcapo_train.jsonl}"
DATASET_FILENAME="${DATASET_FILENAME:-hcapo_train.jsonl}"
UPLOAD_DATASET_ONLY=false
WITH_DATASET_UPLOAD=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --username)     HF_USERNAME="$2";  shift 2 ;;
        --dataset-repo) DATASET_REPO="$2"; shift 2 ;;
        --output-repo)  OUTPUT_REPO="$2";  shift 2 ;;
        --model)        MODEL_NAME="$2";   shift 2 ;;
        --config)       HCAPO_CONFIG="$2"; shift 2 ;;
        --flavor)       FLAVOR="$2";       shift 2 ;;
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --run-name)     RUN_NAME="$2";     shift 2 ;;
        --max-steps)    MAX_STEPS="$2";    shift 2 ;;
        --dataset-file) DATASET_FILE="$2"; shift 2 ;;
        --dataset-filename) DATASET_FILENAME="$2"; shift 2 ;;
        --upload-dataset) UPLOAD_DATASET_ONLY=true; shift ;;
        --with-dataset-upload) WITH_DATASET_UPLOAD=true; shift ;;
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

DATASET_REPO="${DATASET_REPO:-${HF_USERNAME}/fswe-hcapo-pg-01-trajectories}"
OUTPUT_REPO="${OUTPUT_REPO:-${HF_USERNAME}/fswe-hcapo-pg-01-qwen36-27b}"
TRACKIO_SPACE="${TRACKIO_SPACE:-${HF_USERNAME}/fswe-hcapo-pg-01-monitor}"

upload_dataset() {
echo "==> Uploading HCAPO dataset to $DATASET_REPO ..."
if [[ ! -f "$DATASET_FILE" ]]; then
    echo "ERROR: Dataset not found at $DATASET_FILE"
    echo "Run 'uv run python scripts/build_hcapo_dataset.py' first."
    exit 1
fi
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
    path_in_repo='${DATASET_FILENAME}',
    repo_id=repo_id,
    repo_type='dataset',
)
print(f'Dataset uploaded to https://huggingface.co/datasets/{repo_id}')
"
else
    echo "  [DRY RUN] Would upload $DATASET_FILE -> $DATASET_REPO"
fi
}

if [[ "$UPLOAD_DATASET_ONLY" == "true" ]]; then
    upload_dataset
    exit 0
fi

# ---- Step 1: Optionally upload dataset to HF Hub ----
if [[ "$WITH_DATASET_UPLOAD" == "true" ]]; then
    upload_dataset
else
    echo "==> Skipping dataset upload. Using existing dataset repo: $DATASET_REPO"
fi

# ---- Step 2: Submit HF Job ----
echo ""
echo "==> Submitting HF Job..."
echo "    Flavor:   $FLAVOR"
echo "    Model:    $MODEL_NAME"
echo "    Dataset:  $DATASET_REPO"
echo "    Output:   $OUTPUT_REPO"
echo "    Trackio:  https://huggingface.co/spaces/$TRACKIO_SPACE"
echo "    Config:   $HCAPO_CONFIG"
echo "    Run name: $RUN_NAME"
echo "    Max steps: ${MAX_STEPS:-full run}"
echo "    Timeout:  $TIMEOUT"
echo ""

JOB_CMD=(
    hf jobs uv run "$PROJECT_ROOT/training/train_hcapo.py"
    --flavor "$FLAVOR"
    --timeout "$TIMEOUT"
    --secrets HF_TOKEN
    --env "HF_ENDPOINT=https://hf-mirror.com"
    --
    --config "$HCAPO_CONFIG"
    --model-name "$MODEL_NAME"
    --dataset-id "$DATASET_REPO"
    --dataset-filename "$DATASET_FILENAME"
    --output-repo "$OUTPUT_REPO"
    --report-to trackio
    --trackio-space "$TRACKIO_SPACE"
    --trackio-project fswe-hcapo-pg-01
    --run-name "$RUN_NAME"
    --push-to-hub
    --hub-private
)

if [[ -n "$MAX_STEPS" ]]; then
    JOB_CMD+=(--max-steps "$MAX_STEPS")
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  ${JOB_CMD[*]}"
else
    echo "Launching..."
    "${JOB_CMD[@]}"
fi
