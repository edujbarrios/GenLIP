import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal

import numpy as np
import torch
import torch.distributed as dist
import wandb
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_mapping_dataset,
    build_iterative_dataset,
    build_iterative_webdataset,
    build_multimodal_chat_template,
)
from veomni.data.constants import IMAGE_INPUT_INDEX
from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce
from veomni.utils.flex_attn_utils import create_flex_mask, create_flex_mask_padding

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate

logger = helper.create_logger(__name__)

def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    **kwargs,
):
    """
    Processes multimodal example with qwen2vl's pre-processor.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source"]
    )

    image = Image.open(BytesIO(sample["jpg"]["bytes"]))
    if image.mode == 'P':
        image = image.convert('RGBA').convert('RGB')
    else:
        image = image.convert('RGB')
    meta = sample["json"]

    # conversations = conv_preprocess(source, meta, **kwargs)
    # print(f"conversations {conversations}")
    # print(f"meta {meta}")
    conversations = conv_preprocess(source, meta, **kwargs)

    token_num_inputs, image_inputs = {}, {}
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    # clip image preprocessor returns pixel_values of shape [bs, channel, height, width]
    # image_grid_thw = image_inputs["image_grid_thw"]
    image_token_num = 1 + (image_inputs['pixel_values'].shape[-2] // 14) * (image_inputs['pixel_values'].shape[-1] // 14)
    token_num_inputs["image"] = [image_token_num]

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs, no_special_tokens=True)
    # tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    tokenized_example = {k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in tokenized_example.items()}
    # input_ids [id_vision_start, id_I, ..., id_I, id_vision_end, t1, ..., t10, id_im_end, eos]

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX 
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example.update(image_inputs) # 'pixel_values'
    return [tokenized_example]

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"name {name}, param, precision {param.dtype}")
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]

@dataclass
class MyModelArguments(ModelArguments):
    clip_preprocessor: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    use_flex_attn: bool = field(default=False)

@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    datasets_type: Literal["mapping", "iterable_wds", "iterable"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    

@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=MyModelArguments)
    data: "DataArguments" = field(default_factory=MyDataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)

def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    if args.model.clip_preprocessor is not None:
        processor.image_processor = build_processor(args.model.clip_preprocessor).image_processor
    
    print(processor)
    chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)

    transform = partial(
        process_sample,
        processor=processor,
        chat_template=chat_template,
        source_name=args.data.source_name,
    )

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    data_collate_fn.append(OmniDataCollatorWithPacking())
    # if args.train.rmpad_with_pos_ids:
    #     data_collate_fn.append(OmniDataCollatorWithPacking())
    # else:
    #     data_collate_fn.append(OmniDataCollatorWithPadding())

    # if get_parallel_state().sp_enabled:
    #     data_collate_fn.append(
    #         OmniSequenceShardCollator(
    #             padding_scale={
    #                 "pixel_values": processor.image_processor.merge_size**2,
    #             },
    #             rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
    #         )
    #     )

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable_wds":
            logger.info_rank0("Start building iterative webdataset")
            train_dataset = build_iterative_webdataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")
    
    fsdp_kwargs = {}
    
    global_step = 0
    start_epoch = 0
    start_step = 0

    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                # batch data_format
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()
            # micro_batches -- list 
            # process into [pixel_values: torch.FloatTensor, input_ids: torch.LongTensor], labels = input_ids
            for micro_batch in micro_batches:
                if global_step == 1:
                    # data_save = micro_batch['input_ids']
                    print(f"micro_batch, {micro_batch['input_ids']} {micro_batch['input_ids'].shape}")
                    print(f"image_mask, {micro_batch['image_mask']}, {micro_batch['image_mask'].shape}")
                    print(f"attention mask, {micro_batch['attention_mask']}, {micro_batch['attention_mask'].shape}")
                    print(f"labels, {micro_batch['labels']}, {micro_batch['labels'].shape}")
                    print(f"pixel values {micro_batch['pixel_values']}, {micro_batch['pixel_values'].shape}")
                    # json.dump(data_save, open("packing_data_case.json", "w")
                    torch.save(micro_batch, open("packing_data_case.pth", "wb"))
                    return 0
                # print(f"micro_batch, {micro_batch} {micro_batch['input_ids'].shape}")
                


                # if args.data.max_seq_len > 0 and micro_batch["input_ids"].shape[-1] > args.data.max_seq_len:
                #     micro_batch["image_mask"] = micro_batch["image_mask"][:, :args.data.max_seq_len]
                #     micro_batch["input_ids"] = torch.cat([micro_batch["input_ids"][:, :args.data.max_seq_len-1], micro_batch["input_ids"][:, -1].view(-1, 1)], dim=-1)
                #     micro_batch["labels"] = torch.cat([micro_batch["labels"][:, :args.data.max_seq_len-1], micro_batch["labels"][:, -1].view(-1, 1)], dim=-1)
                #     logger.info_rank0(f"input ids exceeds the max context length, cut it into max_seq_len {args.data.max_seq_len}.")

                # micro_batch = {
                #     k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                # }

                # if args.model.use_flex_attn:
                #     flex_mask = create_flex_mask_padding(
                #         micro_batch["input_ids"][0].squeeze(0),
                #         micro_batch["image_mask"][0].squeeze(0), 
                #         args.data.max_seq_len
                #     )
                #     flex_args = dict(
                #         flex_mask=flex_mask,
                #         max_seqlen=args.data.max_seq_len,
                #     )
                #     micro_batch["flex_attn_args"] = flex_args
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
