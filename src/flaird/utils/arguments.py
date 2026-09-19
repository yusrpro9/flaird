import os
import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from transformers import TrainingArguments as HfTrainingArguments


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class TrainingArguments(HfTrainingArguments):
    pass


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path/Hub id of an existing FLAIRD checkpoint to resume/evaluate. "
                "When unset, a fresh model is built from the architecture fields below."
            )
        },
    )

    encoder_name_or_path: str = field(
        default="answerdotai/ModernBERT-large",
        metadata={"help": "Hub id/path of the pretrained text encoder to compose FLAIRD with."},
    )

    tokenizer_name_or_path: str | None = field(
        default="answerdotai/ModernBERT-large",
        metadata={
            "help": "Path to pretrained tokenizer. If not specified, uses model_name_or_path."
        },
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code when loading the model."},
    )

    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Keep the semantic encoder frozen during this run."},
    )

    max_seq_length: int | None = field(
        default=512,
        metadata={
            "help": "Maximum sequence length for tokenization. Sequences longer than this will be truncated."
        },
    )

    fusion_type: str = field(
        default="attention",
        metadata={"help": "attention, concatenation."},
    )

    use_generator_classifier: bool = field(default=True)
    num_generator_labels: int = field(default=11)
    generator_loss_weight: float = field(default=0.2)
    pos_weight: float | None = list_field(
        default=None,
        metadata={"help": "Optional machine cross-entropy weights."},
    )
    generator_class_weights: list[float] | None = list_field(
        default=None,
        metadata={"help": "Optional generator class weights cross-entropy weights."},
    )


@dataclass
class DataArguments:
    dataset_validation_path: str = field(
        default="MahmoodAnaam/flaird-raid-pan26",
        metadata={"help": "Hugging Face Hub dataset repository ID or local dataset path."},
    )

    dataset_validation_config_name: str | None = field(
        default="flaird_validation_features",
        metadata={"help": "Dataset configuration name passed to datasets.load_dataset."},
    )

    dataset_train_path: str = field(
        default="MahmoodAnaam/flaird-raid-pan26",
        metadata={"help": "Hugging Face Hub dataset repository ID or local dataset path."},
    )
    dataset_train_config_name: str | None = field(
        default="flaird_train_features",
        metadata={"help": "Dataset configuration name passed to datasets.load_dataset."},
    )

    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be used to shuffle the train dataset."},
    )

    dataset_cache_dir: str | None = field(
        default=None,
        metadata={"help": "Path to cache directory for datasets."},
    )

    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    text_column: str = field(
        default="text",
        metadata={"help": "Name of the column containing the text data."},
    )

    feature_column: str = field(
        default="forensic_features",
        metadata={"help": "Name of the column containing the features."},
    )


def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parse command-line arguments into (ModelArguments, DataArguments, TrainingArguments)."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


if __name__ == "__main__":
    model_args, data_args, training_args = parse_args()
    print(model_args, data_args, training_args)
