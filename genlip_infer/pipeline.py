from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import yaml

from .checkpoint import load_checkpoint
from .preprocessing import BasicPreprocessConfig, load_image, preprocess_image_basic


class InferenceError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise InferenceError("PyTorch is required for inference. Install dependencies (see requirements.txt).") from e


def _require_veomni_models():
    try:
        from veomni.models import build_foundation_model, build_processor  # type: ignore

        return build_foundation_model, build_processor
    except Exception as e:  # pragma: no cover
        raise InferenceError("Failed to import veomni model utilities. Ensure `pip install -e .` is done.") from e


@dataclass(frozen=True)
class InferenceConfig:
    # Model/config assets
    model_config_path: str
    tokenizer_path: Optional[str] = None
    attn_implementation: Optional[str] = "flash_attention_2"
    moe_implementation: Optional[str] = None

    # Prompt/generation (optional)
    prompt: str = "Describe this image. <|vision_start|><|image_pad|><|vision_end|>"
    do_generate: bool = True
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9

    # Fallback preprocessing
    image_size: int = 224


def load_inference_config(config_path: str) -> InferenceConfig:
    path = Path(config_path)
    if not path.exists():
        raise InferenceError(f"Config not found: {config_path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise InferenceError(f"Failed to read YAML config ({config_path}): {type(e).__name__}: {e}") from e

    if not isinstance(data, dict):
        raise InferenceError(f"Config YAML must be a mapping/dict: {config_path}")

    model = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}
    infer = data.get("infer", {}) if isinstance(data.get("infer", {}), dict) else {}

    model_config_path = model.get("config_path") or model.get("model_config_path")
    if not model_config_path:
        raise InferenceError("Config missing `model.config_path` (path to a HF config directory or config json).")

    return InferenceConfig(
        model_config_path=str(model_config_path),
        tokenizer_path=model.get("tokenizer_path") or model.get("tokenizer"),
        attn_implementation=model.get("attn_implementation", "flash_attention_2"),
        moe_implementation=model.get("moe_implementation"),
        prompt=infer.get("prompt", InferenceConfig.prompt),
        do_generate=bool(infer.get("do_generate", True)),
        max_new_tokens=int(infer.get("max_new_tokens", 128)),
        temperature=float(infer.get("temperature", 0.2)),
        top_p=float(infer.get("top_p", 0.9)),
        image_size=int(infer.get("image_size", 224)),
    )


def _resolve_device(device: str) -> torch.device:
    torch = _require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise InferenceError("Requested device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise InferenceError(f"Unsupported device: {device}. Use cpu|cuda|auto.")


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    torch = _require_torch()
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        if device.type != "cuda":
            raise InferenceError("dtype=fp16 is only supported on CUDA.")
        return torch.float16
    if dtype == "bf16":
        # bfloat16 works on CPU and on newer GPUs; allow both and let torch error if unsupported.
        return torch.bfloat16
    raise InferenceError(f"Unsupported dtype: {dtype}. Use fp32|fp16|bf16.")


def _maybe_to_device(inputs: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    torch = _require_torch()
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


class InferencePipeline:
    def __init__(
        self,
        cfg: InferenceConfig,
        *,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        dtype: Literal["fp32", "fp16", "bf16"] = "fp32",
    ) -> None:
        self.cfg = cfg
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)

        self._model = None
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            raise InferenceError("Model is not loaded yet. Call `load(...)` or `run(...)` first.")
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            raise InferenceError("Processor is not loaded yet. Call `load(...)` or `run(...)` first.")
        return self._processor

    def load(self, checkpoint_path: str, *, ckpt_manager: Optional[str] = None) -> None:
        torch = _require_torch()
        build_foundation_model, build_processor = _require_veomni_models()
        loaded = load_checkpoint(checkpoint_path, ckpt_manager=ckpt_manager)

        tokenizer_path = self.cfg.tokenizer_path or self.cfg.model_config_path
        try:
            processor = build_processor(tokenizer_path)
        except Exception as e:
            raise InferenceError(
                f"Failed to build processor from {tokenizer_path}: {type(e).__name__}: {e}"
            ) from e

        if loaded.kind == "hf_dir":
            try:
                model = build_foundation_model(
                    config_path=self.cfg.model_config_path,
                    weights_path=loaded.path,
                    torch_dtype="float32",
                    attn_implementation=self.cfg.attn_implementation,  # type: ignore[arg-type]
                    moe_implementation=self.cfg.moe_implementation,  # type: ignore[arg-type]
                    init_device="cpu",
                )
            except Exception as e:
                raise InferenceError(
                    f"Failed to build model from config={self.cfg.model_config_path} and HF weights dir={loaded.path}: "
                    f"{type(e).__name__}: {e}"
                ) from e
        else:
            try:
                model = build_foundation_model(
                    config_path=self.cfg.model_config_path,
                    weights_path=None,
                    torch_dtype="float32",
                    attn_implementation=self.cfg.attn_implementation,  # type: ignore[arg-type]
                    moe_implementation=self.cfg.moe_implementation,  # type: ignore[arg-type]
                    init_device="cpu",
                )
            except Exception as e:
                raise InferenceError(f"Failed to build model from config={self.cfg.model_config_path}: {type(e).__name__}: {e}") from e

            state_dict = loaded.state_dict or {}
            try:
                incompatible = model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise InferenceError(f"Failed to load state_dict into model: {type(e).__name__}: {e}") from e
            # For inference, prefer loading best-effort and surfacing mismatches in the output.
            self._load_missing_keys = list(getattr(incompatible, "missing_keys", []))
            self._load_unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

        model.eval()
        model.to(device=self.device)
        if self.dtype != torch.float32:
            model.to(dtype=self.dtype)

        self._model = model
        self._processor = processor

    def run(
        self,
        *,
        image_path: str,
        checkpoint_path: str,
        ckpt_manager: Optional[str] = None,
    ) -> Dict[str, Any]:
        torch = _require_torch()
        if self._model is None or self._processor is None:
            self.load(checkpoint_path, ckpt_manager=ckpt_manager)

        image = load_image(image_path)

        processor = self._processor
        model = self._model

        # Build inputs: try multimodal processor path, otherwise fall back to simple tensor input.
        inputs: Dict[str, Any]
        used_processor = False
        input_error: Optional[str] = None
        try:
            text = self.cfg.prompt
            if hasattr(processor, "apply_chat_template") and isinstance(getattr(processor, "chat_template", None), str):
                messages = [{"role": "user", "content": text}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            used_processor = True
        except Exception as e:
            input_error = f"{type(e).__name__}: {e}"
            pixel_values = preprocess_image_basic(
                image, cfg=BasicPreprocessConfig(image_size=self.cfg.image_size)
            ).to(dtype=torch.float32)
            inputs = {"pixel_values": pixel_values}

        inputs = _maybe_to_device(inputs, self.device, self.dtype)

        with torch.no_grad():
            t0 = time.time()
            out = model(**inputs)
            forward_s = time.time() - t0

        result: Dict[str, Any] = {
            "ok": True,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "config_path": self.cfg.model_config_path,
            "checkpoint": checkpoint_path,
            "image": {"path": image_path, "size": [image.width, image.height]},
            "used_processor_inputs": used_processor,
            "input_build_error": input_error,
            "forward_seconds": forward_s,
        }
        missing_keys = getattr(self, "_load_missing_keys", [])
        unexpected_keys = getattr(self, "_load_unexpected_keys", [])
        if missing_keys:
            result["checkpoint_missing_keys_count"] = len(missing_keys)
            result["checkpoint_missing_keys_head"] = missing_keys[:20]
        if unexpected_keys:
            result["checkpoint_unexpected_keys_count"] = len(unexpected_keys)
            result["checkpoint_unexpected_keys_head"] = unexpected_keys[:20]

        # Summarize outputs without dumping tensors.
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            result["logits_shape"] = list(out.logits.shape)
        if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
            result["last_hidden_state_shape"] = list(out.last_hidden_state.shape)

        if self.cfg.do_generate and hasattr(model, "generate") and used_processor:
            gen_kwargs = {
                "do_sample": self.cfg.temperature > 0,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "max_new_tokens": self.cfg.max_new_tokens,
            }
            with torch.no_grad():
                t1 = time.time()
                generated = model.generate(**inputs, **gen_kwargs)
                gen_s = time.time() - t1
            try:
                prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
                text_out = processor.decode(generated[0, prompt_len:], skip_special_tokens=True)
            except Exception:
                text_out = None
            result["generate_seconds"] = gen_s
            result["generated_text"] = text_out

        return json.loads(json.dumps(result, default=str))
