#!/bin/bash
#SBATCH -J genlip
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
# echo "ROOT: ${ROOT}"
mkdir -p logs

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-log_trainings.txt}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${LOG_FILE}}"
mkdir -p "$(dirname "${LOG_PATH}")"

# usage:
# sbatch jobs/train_slurm_mutlinode.sh <main_func> <model_config> [extra args ...]
DEFAULT_MAIN_FUNC="${ROOT}/tasks/train_genlip_stage1.py"
DEFAULT_MODEL_CONFIG="${ROOT}/configs/pretrain/genlip/stage1/train_genlip_so16_224_recap.yaml"

MAIN_FUNC="${1:-${DEFAULT_MAIN_FUNC}}"
MODEL_CONFIG="${2:-${DEFAULT_MODEL_CONFIG}}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi
EXTRA_ARGS=("$@")

# ENV Variables 
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_NNODES}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

cat <<EOF
[GenLIP Slurm Launch]
ROOT=${ROOT}
SLURM_JOB_ID=${SLURM_JOB_ID}
SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}
NNODES=${NNODES}
GPUS_PER_NODE=${GPUS_PER_NODE}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
MAIN_FUNC=${MAIN_FUNC}
MODEL_CONFIG=${MODEL_CONFIG}
EOF

srun \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    set -x

    export NODE_RANK="${SLURM_NODEID}"
    export NNODES="'"${NNODES}"'"
    export MASTER_ADDR="'"${MASTER_ADDR}"'"
    export MASTER_PORT="'"${MASTER_PORT}"'"
    export NPROC_PER_NODE="'"${GPUS_PER_NODE}"'"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="'"${OMP_NUM_THREADS}"'"
    export LOG_DIR="'"${LOG_DIR}"'"
    export LOG_FILE="'"${LOG_FILE}"'"
    export LOG_PATH="'"${LOG_PATH}"'"

    cd "'"${ROOT}"'"
    mkdir -p "$(dirname "${LOG_PATH}")"

    torchrun \
      --nnodes="${NNODES}" \
      --nproc-per-node="${NPROC_PER_NODE}" \
      --node-rank="${NODE_RANK}" \
      --master-addr="${MASTER_ADDR}" \
      --master-port="${MASTER_PORT}" \
      "$@" 2>&1 | tee "${LOG_PATH}"
  ' _ "${MAIN_FUNC}" "${MODEL_CONFIG}" "${EXTRA_ARGS[@]}"

# Usages:
# 1) case 1:
#    sbatch jobs/train_slurm_mutlinode.sh
#
# 2) case 2:
#    sbatch jobs/train_slurm_mutlinode.sh \
#      tasks/train_genlip_stage1.py \
#      configs/pretrain/genlip/stage1/train_genlip_so16_224_recap.yaml
#
# 3) case 3:
#    sbatch --export=ALL,GPUS_PER_NODE=8,MASTER_PORT=29600 jobs/train_slurm_mutlinode.sh
#
# 4) case 4:
#    sbatch --export=ALL,LOG_DIR=logs/exp1,LOG_FILE=train.txt jobs/train_slurm_mutlinode.sh
#    sbatch --export=ALL,LOG_PATH=/tmp/genlip_train.log jobs/train_slurm_mutlinode.sh
