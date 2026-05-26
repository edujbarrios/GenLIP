# GenLIP (Learning Fork)

This repository is a **fork** of the original GenLIP codebase, kept primarily to **understand how the training “world” works end-to-end** (configs → dataloading → training scripts → checkpoints).

- Upstream project/paper: https://arxiv.org/abs/2605.00809
- Upstream code: https://github.com/YanFangCS/GenLIP

<div align="center">
  <img src="assets/teaser.png" alt="GenLIP teaser" style="height: 200px; width: auto;">
</div>

## Setup

### 1) Install

```bash
git clone <your-fork-url>
cd GenLIP

python -m pip install -r requirements.txt
python -m pip install -e .
```

If you are using **PyTorch >= 2.6.0**, you may need to install **ByteCheckpoint** manually (the upstream note still applies):

```bash
git clone https://github.com/ByteDance-Seed/ByteCheckpoint.git
cd ByteCheckpoint
python -m pip install -e .
```

### 2) Configure datasets

Training configs live under `configs/pretrain/`. Before launching a run, update:

- dataset paths in the YAML (`data.*`)
- output directory (`train.output_dir`)
- model config path (`model.config_path`) if you move configs

## Usage

### Train (single node)

```bash
bash jobs/train.sh <main_func> <train_config>

# Stage 1 example:
bash jobs/train.sh tasks/train_genlip_stage1.py configs/pretrain/genlip/stage1/train_genlip_so16_224_recap.yaml

# Stage 2 example:
bash jobs/train.sh tasks/train_genlip_navit.py configs/pretrain/genlip/stage2/train_genlip_so16_navit.yaml
```

### Train (multi-node)

- `jobs/train_multinode.sh` for torchrun-based multi-node
- `jobs/train_slurm_mutlinode.sh` for Slurm clusters

## Model checkpoints

Pretrained checkpoints from the upstream release are on HuggingFace:
https://huggingface.co/collections/YanFang/genlip

## License

Apache-2.0 (see `LICENSE`).
