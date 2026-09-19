import json
import os
import random
from datetime import date
from pathlib import Path

import click
import numpy as np
import torch
from datasets import load_dataset
from transformers import set_seed

from .tira_test import test

HF_USERNAME = os.getenv("HF_USERNAME", "yusr9")

DEFAULT_MODELS = [
    f"{HF_USERNAME}/flaird-modernbert-large-attention-multitask",
    f"{HF_USERNAME}/flaird-modernbert-large-concatenation-multitask",
    f"{HF_USERNAME}/flaird-modernbert-large-attention-single-task",
    f"{HF_USERNAME}/flaird-modernbert-large-concatenation-single-task",
    f"{HF_USERNAME}/flaird-modernbert-large-attention-multitask-frozen",
    f"{HF_USERNAME}/flaird-modernbert-large-concatenation-multitask-frozen",
    f"{HF_USERNAME}/flaird-modernbert-large-attention-single-task-frozen",
    f"{HF_USERNAME}/flaird-modernbert-large-concatenation-single-task-frozen",
]


def create_metadata(
    date_released: str,
    detector_name: str,
    hf_link: str,
    contact_info: str,
) -> dict:
    """Create metadata dictionary following RAID template format."""
    return {
        "date_released": date_released,
        "detector_name": detector_name,
        "huggingface_link": hf_link,
        "contact_info": contact_info,
    }


@click.command()
@click.option(
    "--model",
    "model_path",
    default="yusr9/flaird-modernbert-large-attention-multitask",
    help="Hugging Face model repository name or local model path.",
)
@click.option(
    "--all-models",
    is_flag=True,
    help="If set, run predictions for all 8 trained FLAIRD models.",
)
@click.option(
    "--dataset-path",
    default="liamdugan/raid",
    help="Path or name of the RAID test dataset.",
)
@click.option(
    "--dataset-name",
    default="raid_test",
    help="Defining the name of the RAID test dataset configuration.",
)
@click.option(
    "--feature-column",
    default=None,
    help="forensic features  column of the RAID test dataset.",
)
@click.option(
    "--apply-text-preprocessing",
    is_flag=True,
    help="apply-text-preprocessing.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="raid_submissions",
    help="Base output directory where RAID submissions will be stored.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="Inference batch size.",
)
@click.option(
    "--contact-info",
    default="Email Address: yusrpro9@gmail.com",
    help="Email Address.",
)
def main(
    model_path,
    all_models,
    dataset_path,
    dataset_name,
    feature_column,
    apply_text_preprocessing,
    output_dir,
    batch_size,
    contact_info,
):
    """
    RAID Leaderboard Submission Generator for FLAIRD models.
    Generates predictions.json and metadata.json according to RAID leaderboard requirements.
    """

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    models_to_run = DEFAULT_MODELS if all_models else [model_path]

    test_ds = load_dataset(dataset_path, dataset_name, split="test")

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    for model_id in models_to_run:
        detector_name = model_id.split("/")[-1]
        detector_dir = base_out / detector_name
        detector_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\n--- Processing RAID Submission for: {detector_name} ---")
        predictions_df = test(
            dataset=test_ds,
            model_path=model_id,
            text_column="generation",
            feature_column = feature_column,
            apply_text_preprocessing=apply_text_preprocessing,
            score_column="score",
            batch_size=batch_size,
        )

        # Structure predictions format: list of scores or dict mapping id -> score
        # RAID accepts list of predictions or dict matching test dataset order
        predictions_file = detector_dir / "predictions.json"
        predictions_df.to_json(predictions_file, orient="records", lines=True)

        metadata = create_metadata(
            date_released=date.today().isoformat(),
            detector_name=detector_name,
            hf_link=f"https://huggingface.co/{model_id}",
            contact_info=contact_info,
        )
        metadata_file = detector_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        click.echo("Successfully generated:")
        click.echo(f"  - {predictions_file}")
        click.echo(f"  - {metadata_file}")


if __name__ == "__main__":
    main()
