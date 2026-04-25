#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# launch_hf_space.sh — Create an HF Space for DPO training on A100
#
# Usage:
#   ./scripts/launch_hf_space.sh                  # create & launch
#   ./scripts/launch_hf_space.sh --dry-run        # print plan only
#   ./scripts/launch_hf_space.sh --delete         # tear down Space
#   ./scripts/launch_hf_space.sh --max-seq 32768  # 32k context
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load HF_TOKEN from .env if not already set
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$PROJECT_ROOT/.env" ]]; then
    HF_TOKEN=$(grep -m1 '^HF_TOKEN=' "$PROJECT_ROOT/.env" | cut -d= -f2-)
    export HF_TOKEN
fi

# ---- Defaults ----
HF_USERNAME="${HF_USERNAME:-}"
SPACE_ID="${SPACE_ID:-}"
DATASET_REPO="${DATASET_REPO:-}"
OUTPUT_REPO="${OUTPUT_REPO:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
MAX_SEQ="${MAX_SEQ:-16384}"
FLAVOR="${FLAVOR:-a100-large}"
RUN_NAME="${RUN_NAME:-dpo-qwen36-27b}"
DRY_RUN=false
DELETE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --username)     HF_USERNAME="$2";  shift 2 ;;
        --space-id)     SPACE_ID="$2";     shift 2 ;;
        --dataset-repo) DATASET_REPO="$2"; shift 2 ;;
        --output-repo)  OUTPUT_REPO="$2";  shift 2 ;;
        --model)        MODEL_NAME="$2";   shift 2 ;;
        --max-seq)      MAX_SEQ="$2";      shift 2 ;;
        --flavor)       FLAVOR="$2";       shift 2 ;;
        --run-name)     RUN_NAME="$2";     shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift   ;;
        --delete)       DELETE=true;       shift   ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Resolve HF username
if [[ -z "$HF_USERNAME" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "ERROR: HF_TOKEN not set. Add it to .env or export it."
        exit 1
    fi
    HF_USERNAME=$(uv run python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null || true)
    if [[ -z "$HF_USERNAME" ]]; then
        echo "ERROR: Could not determine HF username from HF_TOKEN."
        exit 1
    fi
fi

SPACE_ID="${SPACE_ID:-${HF_USERNAME}/frontier-swe-dpo-training}"
DATASET_REPO="${DATASET_REPO:-${HF_USERNAME}/frontier-swe-pg-task-001-dpo-trajectories}"
OUTPUT_REPO="${OUTPUT_REPO:-${HF_USERNAME}/frontier-swe-dpo-qwen36-27b}"
TRACKIO_SPACE="${HF_USERNAME}/frontier-swe-dpo-monitor"

# ---- Delete mode ----
if [[ "$DELETE" == "true" ]]; then
    echo "==> Deleting Space $SPACE_ID ..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would delete $SPACE_ID"
    else
        uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.delete_repo('${SPACE_ID}', repo_type='space')
    print('Space deleted: ${SPACE_ID}')
except Exception as e:
    print(f'Delete failed: {e}')
"
    fi
    exit 0
fi

# ---- Create & launch ----
echo "==> Creating HF Space for DPO training"
echo "    Space:    $SPACE_ID"
echo "    Flavor:   $FLAVOR"
echo "    Model:    $MODEL_NAME"
echo "    Dataset:  $DATASET_REPO"
echo "    Output:   $OUTPUT_REPO"
echo "    Trackio:  https://huggingface.co/spaces/$TRACKIO_SPACE"
echo "    Max Seq:  $MAX_SEQ"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would create Space and upload training files."
    exit 0
fi

uv run python -c "
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
space_id = '${SPACE_ID}'
project_root = '${PROJECT_ROOT}'

# 1. Create the Space repo
print('Creating Space repo...')
try:
    create_repo(
        space_id,
        repo_type='space',
        space_sdk='docker',
        space_hardware='${FLAVOR}',
        exist_ok=True,
        private=True,
    )
except Exception as e:
    print(f'Repo creation note: {e}')

# 2. Set secrets and env vars
print('Configuring secrets and environment variables...')
api.add_space_secret(space_id, 'HF_TOKEN', os.environ['HF_TOKEN'])
env_vars = {
    'DATASET_ID': '${DATASET_REPO}',
    'MODEL_NAME': '${MODEL_NAME}',
    'OUTPUT_REPO': '${OUTPUT_REPO}',
    'MAX_SEQ_LENGTH': '${MAX_SEQ}',
    'TRACKIO_SPACE': '${TRACKIO_SPACE}',
    'RUN_NAME': '${RUN_NAME}',
    'HF_ENDPOINT': 'https://hf-mirror.com',
}
for key, val in env_vars.items():
    api.add_space_variable(space_id, key, val)

# 3. Upload all files the Dockerfile needs
print('Uploading training files...')
files_to_upload = [
    ('training/Dockerfile.train', 'Dockerfile'),
    ('scripts/train_dpo_hfjob.py', 'scripts/train_dpo_hfjob.py'),
    ('pyproject.toml', 'pyproject.toml'),
    ('uv.lock', 'uv.lock'),
]
for local_path, repo_path in files_to_upload:
    full = os.path.join(project_root, local_path)
    if not os.path.exists(full):
        print(f'  SKIP (not found): {local_path}')
        continue
    print(f'  {local_path} -> {repo_path}')
    api.upload_file(
        path_or_fileobj=full,
        path_in_repo=repo_path,
        repo_id=space_id,
        repo_type='space',
    )

print()
print(f'Space created: https://huggingface.co/spaces/{space_id}')
print(f'Trackio:       https://huggingface.co/spaces/${TRACKIO_SPACE}')
print()
print('The Space will build the Docker image and start training automatically.')
print()
print('IMPORTANT: Delete the Space when training finishes to stop billing:')
print(f'  ./scripts/launch_hf_space.sh --delete')
"
