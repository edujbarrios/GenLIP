# Inference (Single Image)

This repo includes a small, dependency-light inference entrypoint for running a single-image forward pass (and optional text generation) from a config + checkpoint.

## CLI

Example:

```bash
python scripts/infer.py \
  --image assets/teaser.png \
  --checkpoint checkpoints/model.pt \
  --config configs/infer.yaml \
  --device cuda \
  --dtype fp16 \
  --json
```

### Arguments

- `--image`: Path to an input image.
- `--checkpoint`: Checkpoint path. Supported:
  - HuggingFace-style weights directory (e.g. `.../hf_ckpt/` containing `model.safetensors` or `pytorch_model.bin` shards)
  - A torch-saved file containing a model `state_dict` (or a dict with `model` / `state_dict`)
  - A distributed checkpoint directory (set `--ckpt-manager` if needed)
- `--config`: YAML config (see below).
- `--device`: `auto|cpu|cuda`.
- `--dtype`: `fp32|fp16|bf16` (`fp16` requires CUDA).
- `--json`: Print JSON output (recommended for scripting).

## Config format

`scripts/infer.py` reads a YAML file with (at minimum) `model.config_path`.

Example `configs/infer.yaml`:

```yaml
model:
  config_path: "configs/model_configs/genlip/genlip_so16_224.json"
  tokenizer_path: "Qwen/Qwen2-VL-7B-Instruct"   # optional
  attn_implementation: "sdpa"                   # optional

infer:
  prompt: "Describe this image. <|vision_start|><|image_pad|><|vision_end|>"
  do_generate: true
  max_new_tokens: 128
  temperature: 0.2
  top_p: 0.9
  image_size: 224
```

## Output

The CLI prints a JSON object including:

- `logits_shape` (when available on the model output)
- `generated_text` (when generation is enabled and supported)
- device/dtype metadata and basic timing
