from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from genlip_infer.pipeline import InferencePipeline, load_inference_config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-image inference for GenLIP/VeOmni models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="Path to an input image.")
    p.add_argument("--checkpoint", required=True, help="Path to a checkpoint file/dir (or HF weights dir).")
    p.add_argument("--config", required=True, help="Path to an inference YAML config.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"], help="Computation dtype.")
    p.add_argument(
        "--ckpt-manager",
        default="auto",
        choices=["auto", "bytecheckpoint", "dcp", "native"],
        help="Checkpoint manager for distributed checkpoints (directory checkpoints). Use auto to default to bytecheckpoint.",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return p


def _print_error(message: str, *, json_mode: bool) -> None:
    if json_mode:
        sys.stdout.write(json.dumps({"ok": False, "error": message}) + "\n")
    else:
        sys.stderr.write(f"error: {message}\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    json_mode = bool(args.json)

    try:
        cfg = load_inference_config(args.config)
        pipe = InferencePipeline(cfg, device=args.device, dtype=args.dtype)
        result: Dict[str, Any] = pipe.run(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            ckpt_manager=None if args.ckpt_manager == "auto" else args.ckpt_manager,
        )
    except Exception as e:
        _print_error(f"{type(e).__name__}: {e}", json_mode=json_mode)
        return 2

    if json_mode:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        if result.get("generated_text"):
            sys.stdout.write("\n")
            sys.stdout.write(str(result["generated_text"]).rstrip() + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
