#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_gnn_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#
# Sanity-check variant of train_gnn.sh: same env setup + launch, but on a tiny
# amount of data (a few users per group, 1 epoch, small val/train-eval caps).
# Outputs go to experiments/sanity_test/ (and a separate MLflow experiment),
# so they never overlap with real runs.
#
# Usage (submit from experiments/_5_gnn_models_2026):
#   sbatch train_gnn_test.sh [extra train_gnn.py args...]
#
# Overridable via environment:
#   TEST_USERS_PER_GROUP  users taken from each group of user_splits.json (default: 4)
#   DATA_PATH             data dir with datasets/user_splits.json
#                         (default: train_gnn.py's default)
#   + everything train_gnn.sh accepts (VENV_DIR, TORCH_INDEX_URL, ...)

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_gnn.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

TEST_TAG="sanity_test"
TEST_USERS_PER_GROUP="${TEST_USERS_PER_GROUP:-4}"
DATA_PATH="${DATA_PATH:-/serafin/pcelayes/repos/sna_classifier/data}"
TEST_DIR="${SCRIPT_DIR}/experiments/${TEST_TAG}"

# --- Pre-build a tiny user sample -------------------------------------------
# train_gnn.py's --n-users would first load *every* user before sampling, which
# is slow. Instead we write user_sample.json up front: train_gnn.py then loads
# only these users. (Stdlib-only Python, so it runs before the venv exists.)
mkdir -p "${TEST_DIR}"
SAMPLE_FILE="${TEST_DIR}/user_sample.json"
python3 - "${DATA_PATH}/datasets/user_splits.json" "${SAMPLE_FILE}" "${TEST_USERS_PER_GROUP}" <<'EOF'
import json, sys
splits_path, out_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(splits_path) as f:
    splits = json.load(f)
sample = {group: uids[:n] for group, uids in splits.items()}
with open(out_path, "w") as f:
    json.dump(sample, f, indent=2)
print(f"Test user sample -> {out_path}: " + ", ".join(f"{g}={len(u)}" for g, u in sample.items()))
EOF

# Tiny-data settings come first; extra args passed to this script override them
# (argparse keeps the last occurrence).
exec bash "${SCRIPT_DIR}/train_gnn.sh" \
    --experiment-tag "${TEST_TAG}" \
    --n-users none \
    --data-path "${DATA_PATH}" \
    --reset-gnn-training \
    --epochs 1 \
    --log-every-n-steps 10 \
    --patience-steps none \
    --train-chunk-size 2 \
    --max-val-samples 500 \
    --max-train-eval-samples 500 \
    --seed 42 \
    "$@"
