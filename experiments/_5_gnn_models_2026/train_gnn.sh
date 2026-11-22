#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_gnn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Launch GNN training (train_gnn.py) as a SLURM job.
#
# Usage (submit from experiments/_5_gnn_models_2026):
#   sbatch train_gnn.sh [train_gnn.py args...]
# e.g.
#   sbatch train_gnn.sh --experiment-tag v3_full --epochs 10
#   sbatch train_gnn.sh --no-reset-gnn-training          # resume a run
#
# Overridable via environment (sbatch --export=ALL,VAR=value ...):
#   VENV_DIR          virtualenv location   (default: <project>/.venvs/gnn_cluster)
#   PYTHON_BIN        python used to create the venv (default: python3)
#   TORCH_INDEX_URL   PyTorch wheel index matching the cluster CUDA
#                     (default: https://download.pytorch.org/whl/cu121)
#   FORCE_REINSTALL=1 reinstall packages even if requirements are unchanged

set -euo pipefail

# --- Locate the script dir ---------------------------------------------------
# sbatch copies this file to a spool dir, so BASH_SOURCE is not reliable inside
# a job. Prefer the submit dir (if it holds train_gnn.py), else the file's dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_gnn.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/train_gnn.py" ]]; then
    echo "ERROR: train_gnn.py not found in ${SCRIPT_DIR}. Submit the job from experiments/_5_gnn_models_2026." >&2
    exit 1
fi
PROJECT_HOME="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VENV_DIR:-${PROJECT_HOME}/.venvs/gnn_cluster}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
REQUIREMENTS="${SCRIPT_DIR}/requirements_train_gnn.txt"
STAMP_FILE="${VENV_DIR}/.requirements.sha256"

# Uncomment/adapt if the cluster uses environment modules:
# module load python/3.11 cuda/12.1

echo "=================================================================="
echo "Job:          ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-train_gnn})"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Script dir:   ${SCRIPT_DIR}"
echo "Project home: ${PROJECT_HOME}"
echo "Venv:         ${VENV_DIR}"
echo "Started:      $(date)"
echo "=================================================================="
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not available"

# --- Create venv + install packages (under a lock, so concurrent jobs sharing
#     the same venv don't race each other) ------------------------------------
mkdir -p "$(dirname "${VENV_DIR}")"
LOCK_FILE="${VENV_DIR}.lock"

check_python_version() {
    # torch 2.x wheels need Python >= 3.9 (older ones: 3.8). Fail early and clearly.
    if ! "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
        echo "ERROR: $1 is $("$1" --version 2>&1); torch>=2.0 needs Python >= 3.9." >&2
        echo "       Load a newer Python (e.g. 'module avail python' / conda) and set PYTHON_BIN," >&2
        echo "       then delete the old venv: rm -rf ${VENV_DIR}" >&2
        exit 1
    fi
}

setup_env() {
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        check_python_version "${VENV_DIR}/bin/python"
    else
        check_python_version "${PYTHON_BIN}"
    fi
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        echo ">> Creating virtualenv at ${VENV_DIR} (using $(${PYTHON_BIN} --version))"
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        rm -f "${STAMP_FILE}"
    else
        echo ">> Virtualenv found at ${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    # Reinstall only when requirements (or the torch index) changed.
    local req_hash
    req_hash="$( (cat "${REQUIREMENTS}"; echo "${TORCH_INDEX_URL}") | sha256sum | cut -d' ' -f1)"
    if [[ "${FORCE_REINSTALL:-0}" == "1" || ! -f "${STAMP_FILE}" || "$(cat "${STAMP_FILE}")" != "${req_hash}" ]]; then
        echo ">> Installing packages..."
        python -m pip install --upgrade pip wheel setuptools
        # torch first, from the CUDA-specific index
        python -m pip install "torch>=2.0" --index-url "${TORCH_INDEX_URL}"
        python -m pip install -r "${REQUIREMENTS}"
        echo "${req_hash}" > "${STAMP_FILE}"
    else
        echo ">> Packages up to date (requirements unchanged)."
    fi
}

if command -v flock >/dev/null; then
    exec 9>"${LOCK_FILE}"
    flock 9
    setup_env
    flock -u 9
else
    setup_env
fi

# Make sure the venv is active in this shell (setup_env sources it, but be explicit).
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- Sanity check ------------------------------------------------------------
python - <<'EOF'
import torch, torch_geometric
from torch_geometric_signed_directed.nn.directed import MagNetConv  # noqa: F401
print(f"python OK | torch {torch.__version__} | PyG {torch_geometric.__version__} | "
      f"CUDA available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
EOF

# --- Launch training ---------------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SCRIPT_DIR}"

echo ">> Running: python train_gnn.py $*"
set +e
python train_gnn.py "$@"
EXIT_CODE=$?
set -e

echo "=================================================================="
echo "Finished:     $(date)  (exit code ${EXIT_CODE})"
echo "=================================================================="
exit "${EXIT_CODE}"
