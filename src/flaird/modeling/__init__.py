from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from .configuration_flaird import FlairdConfig
from .features import FEATURE_NAMES, ForensicFeatureExtractor
from .modeling_flaird import FlairdForSequenceClassification, FlairdModel
from .modeling_fusion import AttentionFusion, ConcatenationFusion

__all__ = [
    "FEATURE_NAMES",
    "ForensicFeatureExtractor",
    "FlairdConfig",
    "FlairdModel",
    "FlairdForSequenceClassification",
    "AttentionFusion",
    "ConcatenationFusion",
]


AutoConfig.register(FlairdConfig.model_type, FlairdConfig, exist_ok=True)
FlairdConfig.register_for_auto_class("AutoConfig")
AutoModel.register(FlairdConfig, FlairdModel, exist_ok=True)
FlairdModel.register_for_auto_class("AutoModel")
AutoModelForSequenceClassification.register(
    FlairdConfig, FlairdForSequenceClassification, exist_ok=True
)
FlairdForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
