set -x

export TOKENIZERS_PARALLELISM=false

export WANDB_API_KEY=85a9af306647386ae7b1fb87e60fb0fcbb2c11e9


NODE_RANK=0
NNODES=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
NPROC_PER_NODE=8

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee logs/training_log.txt

# use case:
# bash jobs/train.sh tasks/train_genlip_stage1.py configs/pretrain/genlip/stage2/train_genlip_l16_224_recap.yaml