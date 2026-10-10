# src\flaird\modeling\configuration_flaird.py
from typing import Literal

from transformers import AutoConfig, ModernBertConfig, PretrainedConfig

DEFAULT_ID2LABEL = {0: "machine"}
DEFAULT_LABEL2ID = {v: k for k, v in DEFAULT_ID2LABEL.items()}
DEFAULT_ID2GENERATOR_LABEL = {
    0: "GPT",
    1: "Meta-LLaMA",
    2: "MPT",
    3: "Cohere",
    4: "Mistral",
    5: "Gemini",
    6: "DeepSeek",
    7: "Falcon",
    8: "Bison",
    9: "Qwen",
    10: "human",
}
DEFAULT_GENERATOR_ID2LABEL = {v: k for k, v in DEFAULT_ID2GENERATOR_LABEL.items()}

DEFAULT_FEATURE_GROUPS = {
    "lexical_diversity": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    "function_word_style": (10, 11, 12, 13, 14, 15, 16),
    "surface_composition": (17, 21, 22, 23, 24),
    "structural_organization": (18, 27, 28, 34),
    "punctuation_and_rhetoric": (19, 20, 31, 32, 33),
    "information_entropy": (25, 26),
    "repetition_and_burstiness": (29, 30),
}


class FlairdConfig(PretrainedConfig):
    model_type = "flaird"
    sub_configs = {"encoder_config": PretrainedConfig}
    keys_to_ignore_at_inference = ["fusion_states"]

    def __init__(
        self,
        encoder_config: PretrainedConfig | dict | None = None,
        forensic_feature_dim: int = 35,
        feature_group_names: list[str] | None = None,
        feature_group_indices: list[list[int]] | None = None,
        feature_token_dim: int = 128,
        feature_dropout: float = 0.1,
        freeze_encoder: bool = True,
        fusion_type: Literal["attention", "concatenation"] = "attention",
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        classifier_pooling: Literal["cls", "mean"] = "mean",
        classifier_dropout: float = 0.1,
        classifier_bias: bool = True,
        classifier_activation: str = "gelu",
        num_labels: int = 1,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        pos_weight: float | None = None,
        use_generator_classifier: bool = True,
        num_generator_labels: int = 11,
        id2generator_label: dict[int, str] | None = None,
        label2generator_id: dict[str, int] | None = None,
        generator_class_weights: list[float] | None = None,
        generator_loss_weight: float = 0.2,
        output_fusion_states: bool = False,
        **kwargs,
    ):

        super().__init__(
            id2label=id2label or DEFAULT_ID2LABEL,
            label2id=label2id or DEFAULT_LABEL2ID,
            num_labels=num_labels,
            **kwargs,
        )

        if encoder_config is not None:
            if isinstance(encoder_config, dict):
                encoder_type = encoder_config.pop("model_type", None)
                if encoder_type is None:
                    raise ValueError("`encoder_config` dict must include a 'model_type' key.")
                self.encoder_config = AutoConfig.for_model(encoder_type, **encoder_config)
            elif isinstance(encoder_config, PretrainedConfig):
                self.encoder_config = encoder_config
            else:
                raise ValueError(f"Invalid encoder_config type: {type(encoder_config)}")

        else:
            self.encoder_config = ModernBertConfig()

        self.forensic_feature_dim = forensic_feature_dim
        self.feature_group_names = feature_group_names or list(DEFAULT_FEATURE_GROUPS.keys())
        self.feature_group_indices = feature_group_indices or [
            DEFAULT_FEATURE_GROUPS[name] for name in self.feature_group_names
        ]
        self.feature_token_dim = feature_token_dim
        self.feature_dropout = feature_dropout
        self.freeze_encoder = freeze_encoder

        self.fusion_type = fusion_type
        self.fusion_num_heads = fusion_num_heads
        self.fusion_dropout = fusion_dropout

        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation

        self.num_labels = num_labels
        self.id2label = id2label or DEFAULT_ID2LABEL
        self.label2id = label2id or DEFAULT_LABEL2ID
        self.pos_weight = pos_weight

        self.use_generator_classifier = use_generator_classifier
        self.num_generator_labels = num_generator_labels
        self.id2generator_label = id2generator_label or DEFAULT_ID2GENERATOR_LABEL
        self.label2generator_id = label2generator_id or DEFAULT_GENERATOR_ID2LABEL
        self.generator_class_weights = generator_class_weights
        self.generator_loss_weight = generator_loss_weight
        self.output_fusion_states = output_fusion_states

    @property
    def hidden_size(self):
        return self.encoder_config.hidden_size
