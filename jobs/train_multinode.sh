set -x

export TOKENIZERS_PARALLELISM=false

# you may need to set 

# please set the following environment variables before running the script
NODE_RANK=$NODE_RANK
NNODES=$NNODES
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
NPROC_PER_NODE=$NPROC_PER_NODE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-log_trainings.txt}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${LOG_FILE}}"
mkdir -p "$(dirname "${LOG_PATH}")"

# set your wandb token for recording
export WANDB_API_KEY="your_wandb_token"

# then directly launch multinode training using torchrun
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT "$@" 2>&1 | tee "${LOG_PATH}"

# use case:
# bash jobs/train.sh tasks/train_genlip_stage1.py configs/pretrain/genlip/stage2/train_genlip_l16_224_recap.yaml
# LOG_DIR=logs/exp1 LOG_FILE=train.txt bash jobs/train_multinode.sh <main_func> <model_config>
# LOG_PATH=/tmp/genlip_train.log bash jobs/train_multinode.sh <main_func> <model_config>
