import re
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from flaird.modeling.features import ForensicFeatureExtractor


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    text_column: str = "text"
    feature_column: str | None = None
    feature_extractor: ForensicFeatureExtractor | None = None
    use_forensic_features: bool = True
    apply_text_preprocessing: bool = True
    include_labels: bool = True

    def __post_init__(self):
        self.feature_extractor = self.feature_extractor or ForensicFeatureExtractor()

    def __call__(self, examples):

        texts = [
            preprocess_text(str(example[self.text_column]))
            if self.apply_text_preprocessing
            else str(example[self.text_column])
            for example in examples
        ]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features = None
        if self.use_forensic_features:
            features = []
            for example in examples:
                stored = (
                    example.get(self.feature_column) if self.feature_column else None
                )
                features.append(
                    stored
                    if stored is not None
                    else self.feature_extractor(str(example[self.text_column]))
                )
        if features is not None:
            batch["forensic_features"] = torch.tensor(features, dtype=torch.float32)
        if self.include_labels:
            binary_labels = [int(example["label"]) for example in examples]
            generator_labels = [int(example["generator_label"]) for example in examples]
            batch["labels"] = torch.tensor(binary_labels, dtype=torch.long)
            batch["generator_labels"] = torch.tensor(generator_labels, dtype=torch.long)
        return batch


def preprocess_text(text):
    EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}"
    )
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
    text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace("  [PHONE]", " [PHONE]")
    return text.strip()
