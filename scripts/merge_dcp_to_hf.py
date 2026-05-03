import argparse
import os

from transformers import AutoConfig, AutoProcessor

from veomni.checkpoint import bytecheckpoint_ckpt_to_state_dict
from veomni.models import save_model_weights
from veomni.utils import helper


logger = helper.create_logger(__name__)


def merge_to_hf_pt(load_dir: str, save_path: str, model_assets_dir: str = None):
    # save model in huggingface's format
    state_dict = bytecheckpoint_ckpt_to_state_dict(
        save_checkpoint_path=load_dir,
        output_dir=save_path,
    )
    if model_assets_dir is not None:
        config = AutoConfig.from_pretrained(model_assets_dir)
        processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)

        save_model_weights(save_path, state_dict, model_assets=[config, processor])
    else:
        save_model_weights(save_path, state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model_assets_dir", type=str, default=None)
    args = parser.parse_args()
    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    logger.info(f"Merge Args: {args}")
    merge_to_hf_pt(load_dir, save_dir, model_assets_dir)
    logger.info(f"Merge to hf pt success! Save to: {save_dir}")


# python scripts/merge_dcp_to_hf.py --load-dir "/mnt/bn/pistis/fangyan/exps/ViT-So16-Res224-v3-Pretrain-S8B/checkpoints/global_step_193119" --save-dir "/mnt/bn/pistis/fangyan/exps/ViT-So16-Res224-v3-Pretrain-S8B/checkpoints/global_step_193119/hf_ckpt" --model_assets_dir "/mnt/bn/pistis/fangyan/exps/ViT-So16-Res224-v3-Pretrain-S8B/model_assets"
# python scripts/merge_dcp_to_hf.py --load-dir "/mnt/bn/pistis/fangyan/exps/ViT-L16-Res224-v3-Pretrain-S8B/checkpoints/global_step_128748" --save-dir "/mnt/bn/pistis/fangyan/exps/ViT-L16-Res224-v3-Pretrain-S8B/checkpoints/global_step_128748/hf_ckpt" --model_assets_dir "/mnt/bn/pistis/fangyan/exps/ViT-L16-Res224-v3-Pretrain-S8B/model_assets"
# /mnt/bn/pistis/fangyan/exps/ViT-g16-Res224-v3-Pretrain-S8B/checkpoints/global_step_32187
# python scripts/merge_dcp_to_hf.py --load-dir "/mnt/bn/pistis/fangyan/exps/ViT-g16-Res224-v3-Pretrain-S8B/checkpoints/global_step_32187" --save-dir "/mnt/bn/pistis/fangyan/exps/ViT-g16-Res224-v3-Pretrain-S8B/checkpoints/global_step_32187/hf_ckpt" --model_assets_dir "/mnt/bn/pistis/fangyan/exps/ViT-g16-Res224-v3-Pretrain-S8B/model_assets"