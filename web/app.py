from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _require_gradio():
    try:
        import gradio as gr  # type: ignore

        return gr
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Gradio is required for the web demo. Install dependencies (see requirements.txt) and retry."
        ) from e


def _validate_paths(checkpoint_path: str, config_path: str) -> Tuple[str, str]:
    if not config_path:
        raise ValueError("Missing config path.")
    if not checkpoint_path:
        raise ValueError("Missing checkpoint path.")

    cfg = Path(config_path)
    ckpt = Path(checkpoint_path)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, config_path


_PIPELINE_CACHE: Dict[Tuple[str, str, str, str], Any] = {}


def _get_pipeline(config_path: str, checkpoint_path: str, device: str, dtype: str):
    from genlip_infer.pipeline import InferencePipeline, load_inference_config

    key = (config_path, checkpoint_path, device, dtype)
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is not None:
        return pipe

    cfg = load_inference_config(config_path)
    pipe = InferencePipeline(cfg, device=device, dtype=dtype)
    _PIPELINE_CACHE[key] = pipe
    return pipe


def infer_one(
    image,
    checkpoint_path: str,
    config_path: str,
    device: str,
    dtype: str,
    ckpt_manager: str,
) -> Tuple[str, str]:
    gr = _require_gradio()

    try:
        checkpoint_path, config_path = _validate_paths(checkpoint_path, config_path)
        if image is None:
            raise ValueError("Please upload an image.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            image.save(tmp_path)
            pipe = _get_pipeline(config_path, checkpoint_path, device, dtype)
            result: Dict[str, Any] = pipe.run(
                image_path=tmp_path,
                checkpoint_path=checkpoint_path,
                ckpt_manager=None if ckpt_manager == "auto" else ckpt_manager,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        text_out = result.get("generated_text") or ""
        json_out = json.dumps(result, indent=2)
        return text_out, json_out
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}") from e


def build_demo():
    gr = _require_gradio()

    with gr.Blocks(title="GenLIP Inference Demo") as demo:
        gr.Markdown("## GenLIP single-image inference")

        with gr.Row():
            image = gr.Image(type="pil", label="Image")
            with gr.Column():
                checkpoint = gr.Textbox(label="Checkpoint Path", placeholder="checkpoints/model.pt")
                config = gr.Textbox(label="Config Path", placeholder="configs/infer.yaml")
                device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="Device")
                dtype = gr.Dropdown(choices=["fp32", "fp16", "bf16"], value="fp32", label="Dtype")
                ckpt_manager = gr.Dropdown(
                    choices=["auto", "bytecheckpoint", "dcp", "native"],
                    value="auto",
                    label="Checkpoint Manager",
                )
                run_btn = gr.Button("Run")

        text_out = gr.Textbox(label="Output Text", lines=6)
        json_out = gr.Code(label="JSON", language="json")

        run_btn.click(
            fn=infer_one,
            inputs=[image, checkpoint, config, device, dtype, ckpt_manager],
            outputs=[text_out, json_out],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()

