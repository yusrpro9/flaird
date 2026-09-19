# src\flaird\modeling\modeling_flaird.py
"""Hybrid multi-task FLAIRD architecture."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from .configuration_flaird import FlairdConfig
from .features import ForensicFeatureExtractor
from .modeling_fusion import AttentionFusion, ConcatenationFusion, FusionOutput


class FeatureEncoder(nn.Module):
    """Encode linguistically related scalar features as group states."""

    def __init__(self, config: FlairdConfig):
        super().__init__()
        self.feature_dim = config.forensic_feature_dim
        self.group_indices = tuple(tuple(group) for group in config.feature_group_indices)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(len(indices), config.feature_token_dim),
                ACT2FN["gelu"],
                nn.LayerNorm(config.feature_token_dim),
                nn.Dropout(config.feature_dropout),
            )
            for indices in self.group_indices
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(-1) != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, received {features.size(-1)}")

        features = torch.sign(features) * torch.log1p(features.abs())
        return torch.stack(
            [
                projection(features[:, indices])
                for projection, indices in zip(self.projections, self.group_indices, strict=True)
            ],
            dim=1,
        )


class FlairdPreTrainedModel(PreTrainedModel):
    config_class = FlairdConfig
    main_input_name = "input_ids"
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    all_tied_weights_keys = OrderedDict()
    feature_extractor = ForensicFeatureExtractor()

    def extract_forensic_features(self, texts: list[str], return_tensors: bool = True):
        features = [self.feature_extractor(text) for text in texts]
        return (
            torch.tensor(features, dtype=torch.float32, device=self.device)
            if return_tensors
            else features
        )

    def extract_forensic_features_dict(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.feature_extractor.extract_dict(text) for text in texts]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(
                module.weight, std=getattr(self.config, "initializer_range", 0.02)
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


@dataclass
class FlairdModelOutput(ModelOutput):
    feature_states: torch.Tensor | None = None
    semantic_states: torch.Tensor | None = None
    fusion_outputs: FusionOutput | None = None


class FlairdModel(FlairdPreTrainedModel):
    def __init__(self, config: FlairdConfig):
        super().__init__(config)
        self.config = config

        self.feature_encoder = FeatureEncoder(config)
        self.encoder = AutoModel.from_config(config.encoder_config)

        if config.fusion_type == "attention":
            self.fusion = AttentionFusion(config=config)
        else:
            self.fusion = ConcatenationFusion(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        forensic_features: torch.Tensor | None = None,
        **kwargs,
    ) -> FlairdModelOutput:

        if self.config.fusion_type in ["attention", "concatenation"] and (
            input_ids is None or forensic_features is None
        ):
            raise ValueError(f"{self.config.fusion_type} requires input_ids and forensic_features")

        feature_encoder_outputs, encoder_outputs = None, None
        feature_states, semantic_states = None, None
        fusion_outputs = None

        feature_encoder_outputs = self.feature_encoder(forensic_features)
        feature_states = feature_encoder_outputs

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        semantic_states = encoder_outputs.last_hidden_state

        fusion_outputs = self.fusion(semantic_states, feature_states, attention_mask)

        return FlairdModelOutput(
            feature_states=feature_states,
            semantic_states=semantic_states,
            fusion_outputs=fusion_outputs,
        )


class FlairdPredictionHead(nn.Module):
    def __init__(self, config: FlairdConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


@dataclass
class FlairdSequenceClassifierOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    generator_logits: torch.Tensor | None = None
    fusion_states: FlairdModelOutput | None = None


class FlairdForSequenceClassification(FlairdPreTrainedModel):
    def __init__(self, config: FlairdConfig):
        super().__init__(config)
        self.config = config

        self.model = FlairdModel(config=config)
        self.head = FlairdPredictionHead(config=config)

        self.classifier = nn.Sequential(
            torch.nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        if config.use_generator_classifier:
            self.generator_classifier = nn.Sequential(
                torch.nn.Dropout(config.classifier_dropout),
                nn.Linear(config.hidden_size, config.num_generator_labels),
            )
        else:
            self.generator_classifier = None

        if config.freeze_encoder:
            self.freeze_encoder()

        self.post_init()

    def freeze_encoder(self):
        if self.model.encoder is not None:
            self.model.encoder.requires_grad_(False)

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def feature_encoder(self):
        return self.model.feature_encoder

    @classmethod
    def from_pretrained_encoder(cls, encoder_name_or_path: str, **kwargs):
        pretrained_encoder = AutoModel.from_pretrained(encoder_name_or_path, trust_remote_code=True)
        config = FlairdConfig(
            encoder_config=pretrained_encoder.config,
            **kwargs,
        )
        model = cls(config)
        model.model.encoder.load_state_dict(pretrained_encoder.state_dict())
        return model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        forensic_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        generator_labels: torch.Tensor | None = None,
        output_fusion_states: bool | None = None,
        **kwargs,
    ) -> FlairdSequenceClassifierOutput:

        output_fusion_states = (
            output_fusion_states
            if output_fusion_states is not None
            else self.config.output_fusion_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forensic_features=forensic_features,
            **kwargs,
        )

        fused_state = outputs.fusion_outputs.fused_state

        pooled_output = self.head(fused_state)
        logits = self.classifier(pooled_output)
        generator_logits = (
            self.generator_classifier(pooled_output)
            if self.generator_classifier is not None
            else None
        )

        loss = None
        if labels is not None:
            pos_weight = (
                None
                if self.config.pos_weight is None
                else torch.tensor(self.config.pos_weight, device=logits.device, dtype=logits.dtype)
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.reshape(-1),
                labels.reshape(-1).float(),
                pos_weight=pos_weight,
                reduction="mean",
            )

            if (
                self.config.generator_loss_weight > 0
                and generator_labels is not None
                and generator_logits is not None
            ):
                generator_class_weights = (
                    None
                    if self.config.generator_class_weights is None
                    else torch.tensor(
                        self.config.generator_class_weights,
                        device=generator_logits.device,
                        dtype=generator_logits.dtype,
                    )
                )
                generator_loss = nn.functional.cross_entropy(
                    generator_logits, generator_labels.long(), weight=generator_class_weights
                )
                loss = loss + self.config.generator_loss_weight * generator_loss

        return FlairdSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            generator_logits=generator_logits,
            fusion_states=outputs if output_fusion_states else None,
        )
