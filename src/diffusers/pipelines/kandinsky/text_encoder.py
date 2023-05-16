import torch
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


class MCLIPConfig(PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(self, modelBase="xlm-roberta-large", transformerDimSize=1024, imageDimSize=768, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)


class MultilingualCLIP(PreTrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = XLMRobertaModel(config)
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs2 = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        return self.LinearTransformation(embs2), embs

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []
