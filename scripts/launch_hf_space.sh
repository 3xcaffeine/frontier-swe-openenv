#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# launch_hf_space.sh — Create an HF Space for HCAPO training on A100
#
# Usage:
#   ./scripts/launch_hf_space.sh                  # create & launch
#   ./scripts/launch_hf_space.sh --dry-run        # print plan only
#   ./scripts/launch_hf_space.sh --delete         # tear down Space
#   ./scripts/launch_hf_space.sh --upload-dataset # upload dataset only
#   ./scripts/launch_hf_space.sh --with-dataset-upload # upload dataset, then launch
#   ./scripts/launch_hf_space.sh --with-dataset-upload --max-steps 1
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
HCAPO_CONFIG="${HCAPO_CONFIG:-training/hcapo_config_a100_q36_27b.json}"
FLAVOR="${FLAVOR:-a100-large}"
RUN_NAME="${RUN_NAME:-fswe-hcapo-pg-01-qwen36-27b}"
MAX_STEPS="${MAX_STEPS:-}"
DATASET_FILE="${DATASET_FILE:-$PROJECT_ROOT/datasets/hcapo_train.jsonl}"
DATASET_FILENAME="${DATASET_FILENAME:-hcapo_train.jsonl}"
UPLOAD_DATASET_ONLY=false
WITH_DATASET_UPLOAD=false
DRY_RUN=false
DELETE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --username)     HF_USERNAME="$2";  shift 2 ;;
        --space-id)     SPACE_ID="$2";     shift 2 ;;
        --dataset-repo) DATASET_REPO="$2"; shift 2 ;;
        --output-repo)  OUTPUT_REPO="$2";  shift 2 ;;
        --model)        MODEL_NAME="$2";   shift 2 ;;
        --config)       HCAPO_CONFIG="$2"; shift 2 ;;
        --flavor)       FLAVOR="$2";       shift 2 ;;
        --run-name)     RUN_NAME="$2";     shift 2 ;;
        --max-steps)    MAX_STEPS="$2";    shift 2 ;;
        --dataset-file) DATASET_FILE="$2"; shift 2 ;;
        --dataset-filename) DATASET_FILENAME="$2"; shift 2 ;;
        --upload-dataset) UPLOAD_DATASET_ONLY=true; shift ;;
        --with-dataset-upload) WITH_DATASET_UPLOAD=true; shift ;;
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

SPACE_ID="${SPACE_ID:-${HF_USERNAME}/fswe-hcapo-pg-01-training}"
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
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would upload $DATASET_FILE -> datasets/$DATASET_REPO/$DATASET_FILENAME"
        return
    fi
    uv run python -c "
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = '${DATASET_REPO}'
create_repo(repo_id, repo_type='dataset', exist_ok=True, private=True)
api.upload_file(
    path_or_fileobj='${DATASET_FILE}',
    path_in_repo='${DATASET_FILENAME}',
    repo_id=repo_id,
    repo_type='dataset',
)
print(f'Dataset uploaded to https://huggingface.co/datasets/{repo_id}')
"
}

if [[ "$UPLOAD_DATASET_ONLY" == "true" ]]; then
    upload_dataset
    exit 0
fi

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
echo "==> Creating HF Space for HCAPO training"
echo "    Space:    $SPACE_ID"
echo "    Flavor:   $FLAVOR"
echo "    Model:    $MODEL_NAME"
echo "    Dataset:  $DATASET_REPO"
echo "    Output:   $OUTPUT_REPO"
echo "    Trackio:  https://huggingface.co/spaces/$TRACKIO_SPACE"
echo "    Config:   $HCAPO_CONFIG"
echo "    Max steps: ${MAX_STEPS:-full run}"
echo "    Upload dataset before launch: $WITH_DATASET_UPLOAD"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would create Space and upload training files."
    if [[ "$WITH_DATASET_UPLOAD" == "true" ]]; then
        echo "[DRY RUN] Would upload $DATASET_FILE -> datasets/$DATASET_REPO/$DATASET_FILENAME"
    fi
    exit 0
fi

if [[ "$WITH_DATASET_UPLOAD" == "true" ]]; then
    upload_dataset
fi

uv run python -c "
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
space_id = '${SPACE_ID}'
project_root = '${PROJECT_ROOT}'
dataset_repo = '${DATASET_REPO}'

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
    'DATASET_FILENAME': '${DATASET_FILENAME}',
    'MODEL_NAME': '${MODEL_NAME}',
    'OUTPUT_REPO': '${OUTPUT_REPO}',
    'HCAPO_CONFIG': '${HCAPO_CONFIG}',
    'REPORT_TO': 'trackio',
    'TRACKIO_SPACE_ID': '${TRACKIO_SPACE}',
    'TRACKIO_PROJECT_NAME': 'fswe-hcapo-pg-01',
    'RUN_NAME': '${RUN_NAME}',
}
if '${MAX_STEPS}':
    env_vars['MAX_STEPS'] = '${MAX_STEPS}'
for key, val in env_vars.items():
    api.add_space_variable(space_id, key, val)

# 3. Upload all files the Dockerfile needs
print('Uploading training files...')
files_to_upload = [
    ('training/Dockerfile.train', 'Dockerfile'),
    ('training/train_hcapo.py', 'training/train_hcapo.py'),
    ('training/hcapo_config_a100_q36_27b.json', 'training/hcapo_config_a100_q36_27b.json'),
    ('training/hcapo_config_4090_q35_4b.json', 'training/hcapo_config_4090_q35_4b.json'),
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
