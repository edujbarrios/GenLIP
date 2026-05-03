from transformers import (
    AutoConfig,
    AutoModel,
)

from .genlip_modeling import GenLIPConfig, GenLIPModel, GenLIPPreTrainedModel, GenLIPModelOutput
from .image_processor import GenLIPNaViTImageProcessor

AutoConfig.register("genlip", GenLIPConfig)
AutoModel.register(GenLIPConfig, GenLIPModel)
