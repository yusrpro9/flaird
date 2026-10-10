# src\flaird\modeling\modeling_fusion.py
from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from .configuration_flaird import FlairdConfig


def pool_semantic_states(
    states: torch.Tensor,
    mask: torch.Tensor | None,
    pooling_type: str,
) -> torch.Tensor:
    if pooling_type == "cls":
        return states[:, 0]
    if mask is None:
        return states.mean(dim=1)

    weights = mask.to(dtype=states.dtype).unsqueeze(-1)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@dataclass
class FusionOutput(ModelOutput):
    fused_state: torch.Tensor | None = None
    feature_attention: torch.Tensor | None = None
    fusion_gate: torch.Tensor | None = None


class AttentionFusion(nn.Module):
    def __init__(self, config: FlairdConfig):
        super().__init__()
        self.config = config

        self.semantic_normalization = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feature_normalization = nn.LayerNorm(
            config.feature_token_dim, eps=config.layer_norm_eps
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            kdim=config.feature_token_dim,
            vdim=config.feature_token_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
            batch_first=True,
        )
        self.context_normalization = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size), nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            ACT2FN["gelu"],
            nn.Dropout(config.fusion_dropout),
        )

    def forward(
        self,
        semantic_states: torch.Tensor,
        feature_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> FusionOutput:

        semantic_state = self.semantic_normalization(
            pool_semantic_states(semantic_states, attention_mask, self.config.classifier_pooling)
        ).unsqueeze(1)
        feature_states = self.feature_normalization(feature_states)

        context, attention_weights = self.attention.forward(
            query=semantic_state,
            key=feature_states,
            value=feature_states,
            need_weights=True,
            average_attn_weights=True,
        )

        semantic_state = semantic_state.squeeze(1)
        context = self.context_normalization(context.squeeze(1))
        gate = self.gate(torch.cat((semantic_state, context), dim=-1))
        mixed_state = context + gate * (semantic_state - context)

        fused_state = self.output(torch.cat((mixed_state, semantic_state * context), dim=-1))

        return FusionOutput(
            fused_state=fused_state,
            feature_attention=attention_weights.squeeze(1),
            fusion_gate=gate.mean(dim=-1),
        )


class ConcatenationFusion(nn.Module):
    def __init__(self, config: FlairdConfig):
        super().__init__()
        self.config = config

        self.output = nn.Sequential(
            nn.Linear(config.hidden_size + config.feature_token_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            ACT2FN["gelu"],
            nn.Dropout(config.fusion_dropout),
        )

    def forward(
        self,
        semantic_states: torch.Tensor,
        feature_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> FusionOutput:
        semantic_state = pool_semantic_states(
            semantic_states,
            attention_mask,
            self.config.classifier_pooling,
        )
        feature_state = feature_states.mean(dim=1)

        fused_state = self.output(torch.cat((semantic_state, feature_state), dim=-1))
        return FusionOutput(fused_state=fused_state, feature_attention=None, fusion_gate=None)
