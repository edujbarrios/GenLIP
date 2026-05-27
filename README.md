# GenLIP (Learning Fork)

This repository is a **fork** of the original GenLIP codebase, kept primarily to **understand how the training “world” works end-to-end** (configs → dataloading → training scripts → checkpoints).

- Upstream project/paper: https://arxiv.org/abs/2605.00809
- Upstream code: https://github.com/YanFangCS/GenLIP

<div align="center">
  <img src="assets/teaser.png" alt="GenLIP teaser" style="height: 200px; width: auto;">
</div>

## Setup

### 1) Clone

```bash
git clone https://github.com/edujbarrios/GenLIP.git
cd GenLIP
```

If you are working from your own fork, replace the URL with your fork's URL.

### 2) Install

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install runtime dependencies and the package itself:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Notes:
- `requirements.txt` is a "full" environment intended for training/inference (it includes PyTorch, CUDA wheels, and `flash-attn`). On CPU-only machines or non-Linux platforms you may need to install PyTorch separately and adjust your dependency set.
- A lightweight editable install without dependencies is possible with `python -m pip install -e . --no-deps` (useful for code navigation and for running the small unit tests that don't require PyTorch).

If `bytecheckpoint` is not available from your environment, install it explicitly first:

```bash
python -m pip install bytecheckpoint
```

If that fails (or if you need a source install), install **ByteCheckpoint** manually:

```bash
git clone https://github.com/ByteDance-Seed/ByteCheckpoint.git
cd ByteCheckpoint
python -m pip install -e .
cd ..
```

### 3) Configure datasets

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

## Inference (Quickstart)

This fork includes a lightweight single-image inference utility under `genlip_infer/` plus a small CLI in `scripts/infer.py`.

### 1) Configure inference (YAML)

An example config is included at `configs/infer.yaml`. The CLI expects a YAML config with at least `model.config_path` pointing to a model config directory or config JSON.

### 2) Run via CLI

```bash
python scripts/infer.py \
  --image assets/teaser.png \
  --checkpoint /path/to/checkpoint_or_hf_weights_dir \
  --config configs/infer.yaml \
  --device cpu \
  --dtype fp32 \
  --json
```

### 3) Run via Python

```python
from genlip_infer import InferencePipeline, load_inference_config

cfg = load_inference_config("configs/infer.yaml")
pipe = InferencePipeline(cfg, device="cpu", dtype="fp32")
result = pipe.run(
    image_path="assets/teaser.png",
    checkpoint_path="/path/to/checkpoint_or_hf_weights_dir",
)
print(result.get("generated_text", ""))
```

### 4) Web demo (Gradio)

```bash
python web/app.py
```

## Tests (How to run)

The repository test suite lives under `tests/` and is runnable with the Python standard library:

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

Run a single test file (example):

```bash
python -m unittest tests.test_cli_args -v
```

Notes:
- `tests/test_checkpoint.py` and `tests/test_preprocessing.py` are skipped unless `torch/torchvision/Pillow` are installed.
- If you want the optional dev tools from `pyproject.toml`, install them with `python -m pip install -e ".[dev]"` (enables `pytest`/`ruff`).

## Model checkpoints

Pretrained checkpoints from the upstream release are on HuggingFace:
https://huggingface.co/collections/YanFang/genlip

## License

Apache-2.0 (see `LICENSE`).
