# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import _calculate_fan_in_and_fan_out
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Optional, Tuple, Union, List, Dict, Callable
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, rope_config_validation
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.utils import (
    is_flash_attn_2_available,
    is_torch_flex_attn_available,
    logging
)

from timm.layers import DropPath

from ....distributed.sequence_parallel import (
    reduce_sequence_parallel_loss,
)

from torch.nn.attention.flex_attention import BlockMask, flex_attention
from ....utils.flex_attn_utils import calculate_pad_length

from ....data.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....distributed.parallel_state import get_parallel_state

from ....utils.import_utils import is_liger_kernel_available

if is_flash_attn_2_available() and torch.cuda.is_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None

if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

logger = logging.get_logger(__name__)

def get_flex_indicators(main_func, self, **kwargs):
    flex_indicators = main_func(self, **kwargs)
    return flex_indicators.squeeze(0)

# we further add swigluffn, layerscale, droppath, qknorm to improve training efficiency and stability
class GenLIPConfig(PretrainedConfig):
    model_type = "genlip"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=32768,
        use_swiglu_ffn=False,                   # whether to use swigluffn
        ls_init_value=0.0,                      # whether to use layersacle and its value, default 1e-5, 0.0 for off
        drop_path_rate=0.0,                     # whether to use drop path and its prob 
        gated_attention=False,
        layer_norm_eps=1e-6,
        spatial_merge_size=1, 
        temporal_patch_size=1,
        tokens_per_second=4,
        initializer_range=0.02,
        initializer_factor=1.0,
        text_embed_dim=None,
        use_llm_head=False,
        llm_decoder=None,
        lm_head_hdsz=1024,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,                     # we set 1e4 for mrope theta
        attention_dropout=0.0,
        mrope_interleaved=False,
        rope_scaling=None,
        use_liger_kernel=False,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):
        # model basic
        self.vocab_size = vocab_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        # vision spec
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # attention implementation
        self.num_attention_heads = num_attention_heads
        self.gated_attention = gated_attention
        self.use_swiglu_ffn = use_swiglu_ffn
        self.ls_init_value = ls_init_value
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
    
        # language modeling part
        self.text_embed_dim = text_embed_dim if text_embed_dim is not None else hidden_size
        self.use_llm_head = use_llm_head
        self.llm_decoder = llm_decoder
        self.lm_head_hdsz = lm_head_hdsz

        # optim and rope 
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.use_liger_kernel = use_liger_kernel
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.mrope_interleaved = mrope_interleaved

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)



class GenLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor of shape (seq_len, num_channels * patch_size * patch_size)
            grid_thw: Tensor of shape (num_images_or_videos, 3) containing temporal, height, width
        """
        # Apply patch embeddings 
        pixel_values_dim = pixel_values.dim()
        target_dtype = self.patch_embedding.weight.dtype

        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype)) # [bs, 3, h, w]
        # support both fixed resolution training(stage1) and navit training(stage2)
        if patch_embeds.shape[-1] * patch_embeds.shape[-2] == 1:
            # navit process, use postion embedding computed from image_thw
            # [all_tokens, 3, ps, ps] -> [all_tokens, dim, 1, 1]
            patch_embeds = patch_embeds.squeeze()
        else:
            # fixed res [bs, 3, h, w] -> [bs, dim, p_h * p_w] -> [bs, p_h * p_w, dim]
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # [bs, h*w, dim]

        embeddings = patch_embeds
        return embeddings

class GenLIPRotaryEmbeddings(nn.Module):
    def __init__(self, config: GenLIPConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn: Callable = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_interleaved = config.mrope_interleaved

        self.mrope_section = config.rope_scaling.get("mrope_section", [12, 12, 12])

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[GenLIPConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_theta
        dim  = config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        # equation: inv_freq = base ** (-(2*i)/d)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
            
    @torch.no_grad()
    def forward(self, x, position_ids): # position_ids [3, bs, position_ids]
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            # [3, bs, dim / 2, 1] @ [3, bs, 1, positions] -> [3, bs, dim / 2, positions] -> [3, bs, positions, dim / 2]
            if self.mrope_interleaved:
                # use mrope_interleaved version, with better frequency allocation
                freqs = self.apply_interleaved_mrope(freqs, self.mrope_section) # [bs, seq, dim / 2]
            emb = torch.cat((freqs, freqs), dim=-1) # [bs, seq, dim]
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3 # [8, 12, 12]
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

# ######################################## #
# The followings are copied from qwen2_5vl #
# ######################################## #
def get_position_id(main_func, self, **kwargs):
    """
    This function is used during the data preprocessing stage to generate position_ids
    and associated parameters (e.g., rope_deltas) for a **single sample** (bs = 1).
    This function is a global function for multiprocessing serialization.
    Args:
        main_func: model.get_position_id
        self: An object holding model-specific information (e.g., SimpleNamespace(config=...)).
        **kwargs: Additional arguments passed to `main_func` (e.g., input_ids).
    Returns:
        dict:
            - "position_ids": Tensor of shape (dim, l), with the batch dimension squeezed.
            - other necessary parameters with the batch dimension squeezed (e.g., rope_deltas).

    Example usage:
        class Model:
            def get_position_id_func(self):  # Used in data_transform during training
                fake_model = SimpleNamespace(config=self.config)
                return partial(get_position_id, main_func, fake_model)

        model = Model()
        func = model.get_position_id_func()
        position_func_returns = func(input_ids=input_ids.unsqueeze(0), **kwargs)
        position_ids = position_func_returns['position_ids']  # shape: (dim, l)

    If a model does not implement `get_position_id_func()`, a default fallback for position_ids can be:
        position_id_returns = {
            "position_ids": torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)  # shape: (dim, l)
        }
    """
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, 1, l), rope_deltas (1, 1)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim   
    ) # [3, bs, positions, dim] -> [bs, 1, positions, dim]
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    ) # [3, 3, L, 64] -> [3, 1, 3, L, 64]

    q_embed = (q * cos) + (rotate_half(q) * sin) # q [bs, n_heads, L, dim] * cos [bs, 1, L, dim]
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_mrope_interleaved_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        
        cos, sin are with shape of [bs, seq_len, dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GenLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        self.gated_attention = config.gated_attention

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        if config.gated_attention:
            # also generate gate_score for gated attention
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.rope_scaling = config.rope_scaling

        self.rotary_func = apply_mrope_interleaved_rotary_pos_emb if config.mrope_interleaved else apply_multimodal_rotary_pos_emb

        self.flex_attn = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # rope
        flex_attn_args: Optional[Dict] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if past_key_value is not None or use_cache:
            raise NotImplementedError(
                "KV Cache is only supported on GenLIPSdpaAttention (attn_implementation='sdpa')."
            )
        use_flex_attn = flex_attn_args is not None

        bs, seq_length, embed_dim = hidden_states.shape

        if self.gated_attention:
            # [bs, seq_len, dim] -> [bs, seq_len, 2*dim] -> (q,[bs, seq_len, dim]),(g,[bs, seq_len, dim])
            q, gate_score = self.q_proj(hidden_states).chunk(2, dim=-1)
            gate_score = gate_score.reshape(bs, seq_length, self.num_heads, self.head_dim)
        else:
            q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention: (seq_len, num_heads, head_dim)
        q = q.view(bs, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # use rope in attn calculation 
        cos, sin = position_embeddings
        q, k = self.rotary_func(
            q, k, cos, sin, self.rope_scaling["mrope_section"]
        )

        if use_flex_attn:
            # to enable flex_attn, the last dim of q, k,v b should be 2 power
            padlen = calculate_pad_length(q.shape[2], flex_attn_args.get("max_seqlen", self.config.max_position_embeddings))
            # pad_head_dim = 128 - self.head_dim
            pad_head_dim = 64 - (self.head_dim % 64)
            attn_output = self.flex_attn(
                F.pad(q, (0, pad_head_dim, 0, padlen)),
                F.pad(k, (0, pad_head_dim, 0, padlen)),
                F.pad(v, (0, pad_head_dim, 0, padlen)),
                block_mask=flex_attn_args["flex_mask"],
                scale=self.scale,
                kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
            )[:, :, : q.shape[2], :self.head_dim].transpose(1, 2).contiguous()
            attn_weights = None
        else:
            if attention_mask is None:
                attention_mask = torch.full(
                    [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
                )
                for i in range(1, len(cu_seqlens)):
                    attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
            
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [bs, n_head, seq, seq]
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous() # [bs, seq_len, n_head, head_dim]
        
        if self.gated_attention:
            attn_output = attn_output * torch.sigmoid(gate_score)
        # attn_output [bs, seq_len, num_head, head_dim]
        attn_output = attn_output.reshape(bs, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, None  # (attn_output, present_key_value)
    
class GenLIPSdpaAttention(GenLIPAttention):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # rope
        flex_attn_args: Optional[Dict] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Incremental decoding path forbids flex_attn; fall back to SDPA.
        use_flex_attn = (flex_attn_args is not None) and (past_key_value is None)

        bs, seq_length, embed_dim = hidden_states.shape
        if self.gated_attention:
            # [bs, seq_len, dim] -> [bs, seq_len, 2*dim] -> (q,[bs, seq_len, dim]),(g,[bs, seq_len, dim])
            q, gate_score = self.q_proj(hidden_states).chunk(2, dim=-1)
            gate_score = gate_score.reshape(bs, seq_length, -1, self.head_dim)
        else:
            q = self.q_proj(hidden_states)
        # q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention: (seq_len, num_heads, head_dim)
        q = q.view(bs, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_length, self.num_heads, self.head_dim).transpose(1, 2)    

        cos, sin = position_embeddings
  
        q, k = self.rotary_func(
            q, k, cos, sin, self.rope_scaling["mrope_section"]
        )

        # Concatenate cached keys/values for incremental decoding.
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if use_cache else None

        if use_flex_attn:
            padlen = calculate_pad_length(q.shape[2], flex_attn_args.get("max_seqlen",  self.config.max_position_embeddings))
            pad_head_dim = 128 - self.head_dim
            attn_output = self.flex_attn(
                F.pad(q, (0, pad_head_dim, 0, padlen)),
                F.pad(k, (0, pad_head_dim, 0, padlen)),
                F.pad(v, (0, pad_head_dim, 0, padlen)),
                block_mask=flex_attn_args["flex_mask"],
                scale=self.scale,
                kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
            )[:, :, : q.shape[2], :self.head_dim].transpose(1, 2).contiguous()
            # logger.info_rank0(f"flex attn_output {attn_output.shape}")
        else:
            if attention_mask is None:
                kv_len = k.shape[2]
                if past_key_value is not None:
                    # Incremental step: new tokens can attend to all cached positions.
                    attention_mask = torch.ones(
                        [1, seq_length, kv_len], device=q.device, dtype=torch.bool
                    )
                else:
                    attention_mask = torch.zeros(
                        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
                    )
                    for i in range(1, len(cu_seqlens)):
                        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

            attn_output = F.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, 
                dropout_p=self.dropout if self.training else 0, scale=self.scale, is_causal=self.is_causal
            )
            attn_output = attn_output.squeeze(0).transpose(1, 2).contiguous()

        if self.gated_attention:
            attn_output = attn_output * torch.sigmoid(gate_score)
        attn_output = attn_output.reshape(bs, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, present_key_value

GenLIP_VISION_ATTENTION_CLASSES = {
    "eager": GenLIPAttention,
    "sdpa": GenLIPSdpaAttention,
}

class GenLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class GenLIPSwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # gated control
        self.gate_fc = nn.Linear(config.hidden_size, config.intermediate_size) 
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated_hs = self.activation_fn(self.gate_fc(hidden_states))
        hidden_states = gated_hs * self.fc1(hidden_states)
        return self.fc2(hidden_states)

class GenLIPLayerScale(nn.Module):
    def __init__(self, config, inplace=False):
        super().__init__()
        self.lambda1 = nn.Parameter(config.ls_init_value * torch.ones(config.hidden_size))
        self.inplace = inplace

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.mul_(self.lambda1) if self.inplace else hidden_states * self.lambda1

class GenLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = GenLIP_VISION_ATTENTION_CLASSES[config._attn_implementation](config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = GenLIPMLP(config) if not config.use_swiglu_ffn else GenLIPSwiGLUFFN(config)
        
        # layer_scale works before computing residual connection
        self.layer_scale1 = GenLIPLayerScale(config, inplace=True) if config.ls_init_value > 1e-6 else nn.Identity()
        self.layer_scale2 = GenLIPLayerScale(config, inplace=True) if config.ls_init_value > 1e-6 else nn.Identity()

        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 1e-6 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # rope
        flex_attn_args: Optional[Dict] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input of shape `(seq_len, embed_dim)`.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            flex_attn_args=flex_attn_args,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        # layerscale1
        hidden_states = self.drop_path(self.layer_scale1(hidden_states))
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # layerscale2
        hidden_states = self.drop_path(self.layer_scale2(hidden_states))
        hidden_states = hidden_states + residual

        return hidden_states, present_key_value

class GenLIPEncoder(nn.Module):
    """Transformer encoder for packed sequences without batch dimension."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GenLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # rope
        flex_attn_args: Optional[Dict] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Args:
            inputs_embeds: (seq_len, hidden_size)
            cu_seqlens: cumulative sequence lengths
            past_key_values: per-layer (k, v) cache list; None disables caching.
            use_cache: whether to return updated key/value caches.
        Returns:
            hidden_states, present_key_values (None when use_cache=False)
        """
        hidden_states = inputs_embeds
        present_key_values: List = []
        for layer_idx, encoder_layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_output, present_kv = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    cu_seqlens,
                    attention_mask,
                    position_embeddings,
                    flex_attn_args,
                    past_kv,
                    use_cache,
                )
            else:
                layer_output, present_kv = encoder_layer(
                    hidden_states,
                    cu_seqlens,
                    attention_mask,
                    position_embeddings,
                    flex_attn_args,
                    past_kv,
                    use_cache,
                )
            hidden_states = layer_output
            if use_cache:
                present_key_values.append(present_kv)

        return hidden_states, (present_key_values if use_cache else None)


class GenLIPVisionTransformer(nn.Module):
    """GenLIP Vision Transformer modified for Qwen25VL-style packed input."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = GenLIPVisionEmbeddings(config)
        self.encoder = GenLIPEncoder(config)
        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor of shape (seq_len, num_channels * patch_size * patch_size)
            grid_thw: Tensor of shape (num_images_or_videos, 3) containing temporal, height, width
        
        Returns:
            Hidden states of shape (seq_len, hidden_size)
        """
        # Get embeddings with position encodings
        hidden_states = self.embeddings(pixel_values, grid_thw)

        # Calculate cumulative sequence lengths for packed sequences
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Pass through encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
        )

        hidden_states = encoder_outputs

        return hidden_states

    def dummy_forward(self):
        """Dummy forward for testing."""
        if getattr(self, "_dummy_data", None) is None:
            # Create dummy data similar to Qwen25VL format
            pixel_values = torch.randn(
                (4, self.config.num_channels * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size),
                dtype=self.dtype, 
                device=self.device
            )
            grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
            self._dummy_data = {"pixel_values": pixel_values, "grid_thw": grid_thw}
        return self(**self._dummy_data)


class GenLIPPreTrainedModel(PreTrainedModel):
    config_class = GenLIPConfig
    base_model_prefix = "GenLIP"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GenLIPEncoder", "GenLIPVisionEmbeddings"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if getattr(module, "padding_idx", None) is not None:
                module.weight.data[module.padding_idx].zero_()
            logger.info_rank0(f"Initialize embedding layers with N(0, {std}) Normal Distribution.")
        elif isinstance(module, (GenLIPAttention, GenLIPSdpaAttention)):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, GenLIPMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, nn.Conv2d):
            fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
            nn.init.trunc_normal_(module.weight, std=math.sqrt(1.0 / fan_in))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

@dataclass
class GenLIPModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    past_key_values: Optional[List] = None

### GenLIPModel implementation ##
class GenLIPModel(GenLIPPreTrainedModel):
    """GenLIP Vision Transformer modified for Qwen25VL-style packed input."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_embeddings = GenLIPVisionEmbeddings(config)
        self.visual = GenLIPEncoder(config)
        self.rotary_emb = GenLIPRotaryEmbeddings(config) # MROPE 1e5 [vision] + [text]

        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) 

        self.use_llm_head = self.config.use_llm_head

        logger.info_rank0(f"Random initializing model embedding layer and lm_head.")

        self.post_init()
        if config.text_embed_dim is not None:
            self.embeddings = nn.Embedding(config.vocab_size, config.text_embed_dim)
            if config.text_embed_dim != config.hidden_size:
                self.in_proj = nn.Linear(config.text_embed_dim, config.hidden_size)
                self.proj = nn.Linear(config.hidden_size, config.text_embed_dim)
            else:
                self.in_proj = nn.Identity()
                self.proj = nn.Identity()

        if self.use_llm_head and config.llm_decoder is not None:
            decoder = AutoModelForCausalLM.from_pretrained(config.llm_decoder)
            decoder_hdsz = decoder.config.hidden_size
            
            if decoder_hdsz != config.text_embed_dim:
                self.in_proj = nn.Linear(decoder_hdsz, config.hidden_size)
                self.proj = nn.Linear(config.hidden_size, decoder_hdsz)

            self.embeddings = nn.Embedding.from_pretrained(
                decoder.model.embed_tokens.weight,
                freeze=False,
            )
            self.lm_head = decoder.lm_head # use lm head from llm to speedup convergence
            self.lm_head_dim = decoder_hdsz
            logger.info_rank0(f"Initializing model embedding layer and lm_head from {config.llm_decoder}.")
        else:
            self.lm_head_dim = config.text_embed_dim
            self.lm_head = nn.Linear(config.text_embed_dim, config.vocab_size, bias=False)
            logger.info_rank0(f"Random initialization for full model, set in_proj and out_proj Identity.")
        
        self.gradient_checkpointing = False
        
        # whether use liger_kernel inproved version for speedup, can save ~10~15% memory for so400m models
        if self.config.use_liger_kernel:
            if is_liger_kernel_available():
                self.loss_fct = LigerFusedLinearCrossEntropyLoss(
                        ignore_index=-100,
                        reduction="mean",
                    )
            else:
                raise ValueError("Can not import LigerFusedLinearCrossEntropyLoss.")
        else:
            self.loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
            )

    @torch.no_grad()
    def initialize_weights(self):  
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.

        Note that the `torch.no_grad()` decorator is very important as well, as most of our `_init_weights` do not use
        `torch.nn.init` functions (which are all no_grad by default), but simply do in-place ops such as
        `module.weight.data.zero_()`.
        """
        layers_skip = ("lm_head", "embeddings")
        use_llm_head = self.config.use_llm_head and (self.config.llm_decoder is not None)
        if not hasattr(torch.nn.Module, "smart_apply"):
            # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
            # to apply as we go down the graph
            def smart_apply(self, fn):
                for name, module in self.named_children():
                    if use_llm_head and name in layers_skip:
                        logger.info_rank0(f"Skip initialization for: {name}")
                        continue
                    # We found a sub-model: recursively dispatch its own init function now!
                    if isinstance(module, PreTrainedModel):
                        module.smart_apply(module._initialize_weights)
                    else:
                        module.smart_apply(fn)
                fn(self)
                return self

            torch.nn.Module.smart_apply = smart_apply

        # Let the magic happen with this simple call
        self.smart_apply(self._initialize_weights)
        logger.info_rank0(f"Call default initialization func for all model layers.")
    
    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st) # 
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            return self._compute_text_only_position_ids(input_ids, attention_mask)

    def get_1D_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D position IDs (ignores vision grid structure, treats all as flat sequence)."""
        return self._compute_text_only_position_ids(input_ids, attention_mask)

    def _compute_text_only_position_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared helper for computing position IDs for pure-text (1D) sequences."""
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas

    def get_position_id_func(self):  # used in data_transform during training
        fake_model = SimpleNamespace(
            config=self.config, image_token_id=IMAGE_INPUT_INDEX, video_token_id=VIDEO_INPUT_INDEX
        )
        return partial(get_position_id, GenLIPModel.get_rope_index, fake_model)
    
    def get_1D_rope_position_id_func(self):  # used in data_transform during training
        fake_model = SimpleNamespace(
            config=self.config, image_token_id=IMAGE_INPUT_INDEX, video_token_id=VIDEO_INPUT_INDEX
        )
        return partial(get_position_id, GenLIPModel.get_1D_rope_index, fake_model)

    def get_flex_indicators(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spatial_merge_size = self.config.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            
            flex_indicators = torch.full_like(total_input_ids, -1)
            
            image_index, video_index = 0, 0
            frame_indicator = 1  # start from 1
            
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                if len(vision_start_indices.shape) == 0 and vision_start_indices.numel() > 0:
                    vision_start_indices = vision_start_indices.unsqueeze(0)
                
                if vision_start_indices.numel() > 0:
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum().item()
                    video_nums = (vision_tokens == video_token_id).sum().item()
                
                input_tokens = input_ids.tolist()
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens[st:] and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                        
                    if video_token_id in input_tokens[st:] and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                        is_image = True
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                        is_image = False
                    
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )

                    st_idx = st + (ed - st)
                    
                    if is_image:
                        start_pos = st_idx
                        end_pos = min(start_pos + llm_grid_h * llm_grid_w, len(input_tokens), flex_indicators[i].shape[0])
                        if end_pos > start_pos:
                            flex_indicators[i, start_pos:end_pos] = frame_indicator
                        frame_indicator += 1
                    else:
                        for t_idx in range(llm_grid_t):
                            start_pos = st_idx + t_idx * llm_grid_h * llm_grid_w
                            end_pos = min(start_pos + llm_grid_h * llm_grid_w, len(input_tokens), flex_indicators[i].shape[0])
                            if end_pos > start_pos:
                                flex_indicators[i, start_pos:end_pos] = frame_indicator
                            frame_indicator += 1  # each frame +1 for indicator
                    
                    st = st_idx + llm_grid_t * llm_grid_h * llm_grid_w
            
            return flex_indicators
        else:
            # pure text all -1
            if attention_mask is not None:
                flex_indicators = torch.full_like(input_ids, -1)
            else:
                flex_indicators = torch.full(
                    [input_ids.shape[0], input_ids.shape[1]],
                    -1,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            
            return flex_indicators

    def get_flex_indicators_func(self):  # used in data_transform during training
        fake_model = SimpleNamespace(
            config=self.config, image_token_id=IMAGE_INPUT_INDEX, video_token_id=VIDEO_INPUT_INDEX
        )
        return partial(get_flex_indicators, GenLIPModel.get_flex_indicators, fake_model)


    def _create_single_modality_attn_mask(self, mask: torch.Tensor):
            bs, seq_len = mask.shape
            device = mask.device
            
            is_modality = mask.int()
            prepended_input = torch.cat(
                (torch.zeros(bs, 1, device=device, dtype=is_modality.dtype), is_modality), dim=-1
            )
            diffs = torch.diff(prepended_input, dim=-1)
            
            segment_starts = (diffs != 0).int()
            segment_ids = torch.cumsum(segment_starts, dim=-1)
            pair_ids = (segment_ids + 1) // 2

            q_pair_ids = pair_ids.unsqueeze(2)  # (bs, seq_len, 1)
            k_pair_ids = pair_ids.unsqueeze(1)  # (bs, 1, seq_len)
            
            same_pair_mask = q_pair_ids == k_pair_ids
            
            k_is_modality = mask.unsqueeze(1)
            
            hybrid_mask = same_pair_mask & k_is_modality
            
            q_is_text = (~mask).unsqueeze(2)
            k_is_text = (~mask).unsqueeze(1)
            
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0)
            
            text_to_text_mask = same_pair_mask & q_is_text & k_is_text & causal_mask
            
            hybrid_mask = hybrid_mask | text_to_text_mask
            
            last_token_pair_ids = pair_ids[:, -1].unsqueeze(1)
            is_in_last_pair = pair_ids == last_token_pair_ids
            hybrid_mask[:, -1, :] = hybrid_mask[:, -1, :] | is_in_last_pair
            
            return hybrid_mask

    def get_hybrid_attn_mask(
        self,
        image_mask=None,
        video_mask=None,
    ):
        if image_mask is None and video_mask is None:
            return None

        if image_mask is not None:
            bs, seq_len = image_mask.shape
            device = image_mask.device
        else:
            bs, seq_len = video_mask.shape
            device = video_mask.device

        final_mask = torch.zeros(bs, seq_len, seq_len, dtype=torch.bool, device=device)

        if image_mask is not None:
            final_mask = final_mask | self._create_single_modality_attn_mask(image_mask)
        
        if video_mask is not None:
            final_mask = final_mask | self._create_single_modality_attn_mask(video_mask)
            
        return final_mask

    # todo2: get flex attn args
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_mask: Optional[torch.FloatTensor] = None,
        video_mask: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        flex_attn_args: Optional[Dict] = None,
    ):
        if input_embeds is None:
            input_embeds = self.embeddings(input_ids) # [bs, seq, dim]
            input_embeds = self.in_proj(input_embeds)

            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.lm_head.weight.dtype)
                image_embeds = self.vision_embeddings(pixel_values, image_grid_thw)
                scatter_image_mask = image_mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)
                input_embeds = input_embeds.masked_scatter(scatter_image_mask, image_embeds)

            # currently do not consider video inputs, but left support for future impl
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.to(dtype=self.lm_head.weight.dtype)
                video_embeds = self.vision_embeddings(pixel_values_videos, video_grid_thw)
                scatter_video_mask = video_mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)
                input_embeds = input_embeds.masked_scatter(scatter_video_mask, video_embeds)
            
            if flex_attn_args is None:
                attention_mask = self.get_hybrid_attn_mask(
                    image_mask,
                    video_mask,
                )
            
            if position_ids is None:
                # Fallback: generate 1D position ids from sequence length.
                seq_len = input_embeds.shape[1]
                position_ids = (
                    torch.arange(seq_len, device=input_embeds.device)
                    .view(1, 1, -1)
                    .expand(3, input_embeds.shape[0], -1)
                )

            # Normalize position_ids to shape [3, bs, seq_len]
            if position_ids.dim() == 2:
                position_ids = position_ids[None, ...].expand(3, input_embeds.shape[0], -1)
            elif position_ids.dim() == 3 and position_ids.shape[0] != 3:
                # Handle [bs, 3, seq_len] -> [3, bs, seq_len]
                position_ids = position_ids.transpose(0, 1)

            hid_states = input_embeds
            position_embeddings = self.rotary_emb(hid_states, position_ids) # [3, bs, positions, dim]
            
            hidden_states, present_key_values = self.visual(
                input_embeds,
                cu_seqlens=None,    # cu_seq_len is used to generate attn mask for the packed sequence
                attention_mask=attention_mask,
                flex_attn_args=flex_attn_args,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache or False,
            )

            hidden_states = self.proj(self.ln_post(hidden_states))

            # calculate loss
            if labels is not None:
                if not get_parallel_state().sp_enabled:
                    labels = labels[..., 1:].contiguous()

                labels = labels.view(-1)

                if self.config.use_liger_kernel and is_liger_kernel_available():
                    if not get_parallel_state().sp_enabled:
                        hidden_states = hidden_states[..., :-1, :].contiguous()

                    output = hidden_states.view(-1, self.lm_head_dim)
                    loss = self.loss_fct(self.lm_head.weight, output, labels)
                    logits = None
                else:
                    logits = self.lm_head(hidden_states)
                    if not get_parallel_state().sp_enabled:
                        logits = logits[..., :-1, :].contiguous()  # shift logits

                    logits = logits.float().view(-1, self.lm_head_dim)
                    loss = self.loss_fct(logits, labels)

                if get_parallel_state().sp_enabled:
                    num_valid_tokens = (labels != IGNORE_INDEX).sum()
                    loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)

            else:
                logits = self.lm_head(hidden_states)
                loss = None

            if not return_dict:
                output = (logits, hidden_states)
                return (loss,) + output if loss is not None else output

            return GenLIPModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                rope_deltas=rope_deltas,
                past_key_values=present_key_values,
            )

ModelClass = GenLIPModel
__all__ = ['GenLIPModel', 'GenLIPPreTrainedModel', 'GenLIPConfig', 'GenLIPModelOutput']
            