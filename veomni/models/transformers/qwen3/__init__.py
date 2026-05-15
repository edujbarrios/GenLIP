from transformers import (
    AutoConfig,
    AutoModel,
)

from .configuration_qwen3 import Qwen3_Config
from .modeling_qwen3 import Qwen3ForCausalLM

AutoConfig.register("qwen3g", Qwen3_Config)
AutoModel.register(Qwen3_Config, Qwen3ForCausalLM)