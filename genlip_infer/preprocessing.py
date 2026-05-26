from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

def _require_pil_image():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception as e:  # pragma: no cover
        raise PreprocessError("Pillow (PIL) is required for image loading. Install dependencies (see requirements.txt).") from e


class PreprocessError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise PreprocessError("PyTorch is required for preprocessing. Install dependencies (see requirements.txt).") from e


def _require_torchvision_transforms():
    try:
        from torchvision import transforms as T  # type: ignore

        return T
    except Exception as e:  # pragma: no cover
        raise PreprocessError(
            "torchvision is required for preprocessing. Install dependencies (see requirements.txt)."
        ) from e


def load_image(image_path: str):
    Image = _require_pil_image()
    path = Path(image_path)
    if not path.exists():
        raise PreprocessError(f"Image not found: {image_path}")
    try:
        image = Image.open(path)
        image = image.convert("RGB")
        return image
    except Exception as e:
        raise PreprocessError(f"Failed to load image ({image_path}): {type(e).__name__}: {e}") from e


@dataclass(frozen=True)
class BasicPreprocessConfig:
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)  # OpenAI CLIP
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)  # OpenAI CLIP


def preprocess_image_basic(image, *, cfg: Optional[BasicPreprocessConfig] = None):
    """
    A dependency-light preprocessing path used for tests and as a fallback.
    Returns a float32 tensor of shape [1, 3, H, W].
    """
    if cfg is None:
        cfg = BasicPreprocessConfig()
    torch = _require_torch()
    T = _require_torchvision_transforms()
    transform = T.Compose(
        [
            T.Resize((cfg.image_size, cfg.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
    x = transform(image).unsqueeze(0)
    return x


def preprocess_with_processor(processor: Any, image) -> Dict[str, Any]:
    """
    Uses a HuggingFace processor/image_processor when available.
    """
    if processor is None:
        raise PreprocessError("Processor is None.")
    if not hasattr(processor, "image_processor"):
        raise PreprocessError("Processor has no attribute `image_processor`.")
    try:
        return processor.image_processor(images=[image], return_tensors="pt")
    except Exception as e:
        raise PreprocessError(f"Processor image preprocessing failed: {type(e).__name__}: {e}") from e
