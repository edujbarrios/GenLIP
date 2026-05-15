## Configs

We organize configuration files into **model_configs** (model architecture) and **pretrain** (training recipes) folders.

### Directory Structure

```
configs/
├── model_configs/
│   ├── genlip/              # GenLIP vision encoder configs
│   │   ├── genlip_l16_224.json      # ViT-L/16
│   │   ├── genlip_so16_224.json     # ViT-SO400M/16
│   │   └── genlip_g16_224.json      # ViT-g/16
│   └── qwen/                # LLM backbone configs
│       └── Qwen3-0.6B.json
├── pretrain/
│   ├── genlip/
│   │   ├── stage1/          # Stage 1: fixed-resolution pretraining
│   │   │   └── train_genlip_*_recap.yaml
│   │   └── stage2/          # Stage 2: NaViT multi-resolution pretraining
│   │       └── train_genlip_*_navit.yaml
│   └── llm/
│       └── qwen3-0.6B-pretrain.yaml   # LLM captioner pretraining
└── ...
```

### Model Configs

Currently, our method supports both ViT architectures (L/16, SO400M/16, g/16) and LLM architectures (Qwen3-0.6B).

| Model | Layers | Hidden Size | Intermediate Size | Config File |
|-------|--------|-------------|-------------------|-------------|
| GenLIP-L/16 | 24 | 1152 | 2752 | `genlip_l16_224.json` |
| GenLIP-SO400M/16 | 27 | 1152 | 3072 | `genlip_so16_224.json` |
| GenLIP-g/16 | 40 | 1536 | 4096 | `genlip_g16_224.json` |
| Qwen3-0.6B | 28 | 1024 | 3072 | `Qwen3-0.6B.json` |

The Qwen3-0.6B model has a comparable parameter count to ViT-SO400M when including the language model head (text embedding layer and lm_head).

### Training Configs

Each training YAML follows a three-section structure:

- **`model`**: model config path, tokenizer, image processor, and attention implementation.
- **`data`**: dataset paths, data loading strategy, sequence length, and data format.
- **`train`**: optimizer, learning rate schedule, parallelism strategy, checkpointing, and logging.

**Key differences between stages:**

| | Stage 1 | Stage 2 |
|---|---------|---------|
| Resolution | Fixed (224×224) | Dynamic (64×64 ~ 512×512 via NaViT) |
| Learning rate | 1e-3 | 1e-4 |
| Dataset type | `iterable_wds` | `iterable_wds_api` |
| Initialization | From scratch | From Stage 1 checkpoint (`model_path`) |

> **Note:** Remember to update `model.config_path`, dataset paths in `data.train_path`, and `train.output_dir` before launching training.

### Using Qwen3-0.6B as Backbone

When training the Qwen3-0.6B model with our method, it is recommended to train from scratch (i.e., do not set `pretrained_weights` in the model config file). Both Qwen/Qwen3-0.6B-Base and Qwen/Qwen3-0.6B initialization leads to inferior performance than from scratch. (Interestingly, our pre-training results suggest that strong language priors do not necessarily translate into stronger visual representations.)

Based on our experiments, pretraining Qwen3-0.6B from scratch achieves results comparable to the ViT-SO400M model.

Currently, our models are trained with `tie_word_embeddings` set to `false` (i.e., the text embedding layer and lm_head do not share parameters). We expect that enabling weight tying would not have a significant impact on performance.
