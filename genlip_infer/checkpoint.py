from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise CheckpointLoadError(
            "PyTorch is required for checkpoint loading. Install dependencies (see requirements.txt)."
        ) from e


def _require_veomni_ckpt_to_state_dict():
    try:
        from veomni.checkpoint import ckpt_to_state_dict  # type: ignore

        return ckpt_to_state_dict
    except Exception as e:  # pragma: no cover
        raise CheckpointLoadError("Failed to import veomni.checkpoint utilities.") from e


class CheckpointLoadError(RuntimeError):
    pass


def _looks_like_hf_weights_dir(path: Path) -> bool:
    # A minimal heuristic: if it contains typical HF weight filenames.
    candidates = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model.pt",
    ]
    return any((path / name).exists() for name in candidates)


def _default_ckpt_cache_dir(checkpoint_path: Path) -> Path:
    key = hashlib.sha256(str(checkpoint_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / "genlip_infer_ckpt_cache" / key


@dataclass(frozen=True)
class LoadedCheckpoint:
    kind: str  # "hf_dir" | "state_dict"
    path: str
    state_dict: Optional[Dict[str, Any]] = None


def load_checkpoint(
    checkpoint_path: str,
    *,
    ckpt_manager: Optional[str] = None,
    map_location: str = "cpu",
) -> LoadedCheckpoint:
    """
    Loads a checkpoint for inference.

    Supports:
    - HuggingFace-style weights directory (safetensors/bin shards).
    - A torch-saved file containing a state_dict or a dict with common keys.
    - VeOmni distributed checkpoints via `veomni.checkpoint.ckpt_to_state_dict`.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointLoadError(f"Checkpoint not found: {checkpoint_path}")

    if path.is_dir():
        if _looks_like_hf_weights_dir(path):
            return LoadedCheckpoint(kind="hf_dir", path=str(path))

        # Assume distributed checkpoint directory that needs conversion.
        if ckpt_manager is None:
            ckpt_manager = "bytecheckpoint"

        cache_dir = _default_ckpt_cache_dir(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            ckpt_to_state_dict = _require_veomni_ckpt_to_state_dict()
            state_dict = ckpt_to_state_dict(
                save_checkpoint_path=str(path),
                output_dir=str(cache_dir),
                ckpt_manager=ckpt_manager,
            )
        except Exception as e:  # pragma: no cover (varies with checkpoint managers installed)
            raise CheckpointLoadError(
                f"Failed to convert distributed checkpoint directory ({checkpoint_path}) "
                f"with ckpt_manager={ckpt_manager}: {type(e).__name__}: {e}"
            ) from e
        return LoadedCheckpoint(kind="state_dict", path=str(path), state_dict=state_dict)

    # File checkpoint
    torch = _require_torch()
    try:
        # PyTorch 2.6 supports weights_only=True; fall back if older.
        try:
            obj = torch.load(str(path), map_location=map_location, weights_only=True)
        except TypeError:
            obj = torch.load(str(path), map_location=map_location)
    except Exception as e:
        raise CheckpointLoadError(f"Failed to load checkpoint file ({checkpoint_path}): {type(e).__name__}: {e}") from e

    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return LoadedCheckpoint(kind="state_dict", path=str(path), state_dict=obj["model"])
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return LoadedCheckpoint(kind="state_dict", path=str(path), state_dict=obj["state_dict"])
        if all(isinstance(k, str) for k in obj.keys()):
            # Assume it's already a state_dict
            return LoadedCheckpoint(kind="state_dict", path=str(path), state_dict=obj)

    raise CheckpointLoadError(
        f"Unsupported checkpoint contents in {checkpoint_path}. "
        "Expected a state_dict dict, or a dict containing 'model'/'state_dict'."
    )


def load_state_dict(checkpoint_path: str, *, ckpt_manager: Optional[str] = None) -> Dict[str, Any]:
    loaded = load_checkpoint(checkpoint_path, ckpt_manager=ckpt_manager)
    if loaded.kind != "state_dict" or loaded.state_dict is None:
        raise CheckpointLoadError(f"Checkpoint is not a state_dict file/dir: {checkpoint_path}")
    return loaded.state_dict
