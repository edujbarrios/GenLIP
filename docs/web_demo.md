# Web Demo

The repository includes a small Gradio demo for single-image inference.

## Run

```bash
python web/app.py
```

Then open the local URL printed by Gradio.

## UI

The demo supports:

- image upload
- checkpoint path input
- config path input
- CPU/CUDA selection
- fp16/bf16 option
- lazy model loading (loads on first run, and caches by config/checkpoint/device/dtype)
- clear errors when checkpoint/config is missing

