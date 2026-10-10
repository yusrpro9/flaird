"""Generate RAID leaderboard submissions for FLAIRD detectors using dataset shards."""

import json
import os
from datetime import date
from pathlib import Path

import click
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

from flaird.data.data_collator import DataCollator


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_USERNAME = os.getenv("HF_USERNAME", "MahmoodAnaam")

DEFAULT_MODEL = (
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask"
)

DEFAULT_MODELS = [
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-multitask",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-single-task",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-single-task",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask-frozen",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-multitask-frozen",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-single-task-frozen",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-single-task-frozen",
]

SEED = 42
DEFAULT_NUM_SHARDS = 5


# =============================================================================
# Metadata
# =============================================================================

def create_metadata(
    detector_name: str,
    model_id: str,
    contact_info: str,
) -> dict[str, str]:
    """Create RAID metadata."""

    return {
        "date_released": date.today().isoformat(),
        "detector_name": detector_name,
        "huggingface_link": f"https://huggingface.co/{model_id}",
        "contact_info": contact_info,
    }


def save_metadata(
    output_dir: Path,
    model_id: str,
    contact_info: str,
) -> None:
    """Save RAID metadata."""

    model_name = model_id.rsplit("/", 1)[-1]
    model_dir = output_dir / model_name

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = create_metadata(
        detector_name=model_name,
        model_id=model_id,
        contact_info=contact_info,
    )

    with (model_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# Model prediction
# =============================================================================

def scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert classifier logits to P(machine-generated)."""

    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1))

    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]

    raise RuntimeError(
        f"Expected one or two output logits, got {tuple(logits.shape)}"
    )


def predict(
    dataset: Dataset,
    model_id: str,
    feature_column: str | None,
    apply_text_preprocessing: bool,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> list[float]:
    """Run inference on one dataset shard."""

    if "generation" not in dataset.column_names:
        raise click.ClickException(
            "The dataset must contain a 'generation' column."
        )

    if feature_column and feature_column not in dataset.column_names:
        raise click.ClickException(
            f"Feature column '{feature_column}' is not present. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    click.echo(f"Loading {model_id} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
    ).to(device)

    model.eval()

    collator = DataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        text_column="generation",
        feature_column=feature_column,
        apply_text_preprocessing=apply_text_preprocessing,
        include_labels=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    scores = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"Predicting {model_id.split('/')[-1]}",
            unit="batch",
        ):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            logits = model(**batch).logits
            batch_scores = scores_from_logits(logits)

            scores.extend(
                batch_scores.float().cpu().tolist()
            )

    if len(scores) != len(dataset):
        raise RuntimeError(
            f"Model returned {len(scores)} scores "
            f"for {len(dataset)} rows."
        )

    del model
    del tokenizer
    del loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


# =============================================================================
# Dataset sharding
# =============================================================================

def get_shard(
    dataset: Dataset,
    shard_index: int,
    num_shards: int,
) -> Dataset:
    """Return the requested shard."""

    return dataset.shard(
        num_shards=num_shards,
        index=shard_index,
        contiguous=True,
    )


# =============================================================================
# Shard files
# =============================================================================

def get_shard_file(
    output_dir: Path,
    model_id: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    """Return the output path for a shard."""

    model_name = model_id.rsplit("/", 1)[-1]

    shard_dir = (
        output_dir
        / model_name
        / f"shards_{num_shards}"
    )

    shard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return shard_dir / f"shard_{shard_index + 1}.jsonl"


# =============================================================================
# Save predictions
# =============================================================================

def save_shard_predictions(
    file_path: Path,
    dataset: Dataset,
    scores: list[float],
) -> None:
    """Save predictions for one shard."""

    ids = (
        list(dataset["id"])
        if "id" in dataset.column_names
        else list(range(len(dataset)))
    )

    if len(ids) != len(scores):
        raise RuntimeError(
            f"Number of IDs ({len(ids)}) does not match "
            f"number of scores ({len(scores)})."
        )

    predictions = pd.DataFrame(
        {
            "id": ids,
            "score": scores,
        }
    )

    predictions.to_json(
        file_path,
        orient="records",
        lines=True,
    )

    click.echo(f"Saved: {file_path}")


# =============================================================================
# Merge shards
# =============================================================================

def merge_shards(
    output_dir: Path,
    model_id: str,
    num_shards: int,
) -> None:
    """Merge all shards when they are available."""

    model_name = model_id.rsplit("/", 1)[-1]

    model_dir = output_dir / model_name
    shard_dir = model_dir / f"shards_{num_shards}"

    files = [
        shard_dir / f"shard_{i + 1}.jsonl"
        for i in range(num_shards)
    ]

    if not all(file.exists() for file in files):
        completed = sum(
            file.exists()
            for file in files
        )

        click.echo(
            f"Shard progress: "
            f"{completed}/{num_shards}"
        )

        return

    click.echo(
        f"All {num_shards} shards completed. "
        f"Merging predictions..."
    )

    predictions = pd.concat(
        [
            pd.read_json(
                file,
                orient="records",
                lines=True,
            )
            for file in files
        ],
        ignore_index=True,
    )

    predictions.to_json(
        model_dir / "predictions.json",
        orient="records",
    )

    click.echo(
        f"Final predictions saved to: "
        f"{model_dir / 'predictions.json'}"
    )


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--model",
    "model_id",
    default=DEFAULT_MODEL,
    show_default=True,
)
@click.option(
    "--all-models",
    is_flag=True,
    help="Run all eight FLAIRD models.",
)
@click.option(
    "--dataset-path",
    default="liamdugan/raid",
    show_default=True,
)
@click.option(
    "--dataset-name",
    default="raid_test",
    show_default=True,
)
@click.option(
    "--feature-column",
    default=None,
)
@click.option(
    "--apply-text-preprocessing",
    is_flag=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="raid_submissions",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
)
@click.option(
    "--max-length",
    type=click.IntRange(min=1),
    default=512,
    show_default=True,
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0),
    default=min(4, os.cpu_count() or 1),
    show_default=True,
)
@click.option(
    "--contact-info",
    default="Email Address: yusrpro9@gmail.com",
    show_default=True,
)
@click.option(
    "--num-shards",
    type=click.IntRange(min=1),
    default=DEFAULT_NUM_SHARDS,
    show_default=True,
    help="Number of shards to split the test dataset into.",
)
@click.option(
    "--shard-index",
    type=click.IntRange(min=0),
    required=True,
    help="Zero-based shard index to evaluate.",
)
def main(
    model_id: str,
    all_models: bool,
    dataset_path: str,
    dataset_name: str,
    feature_column: str | None,
    apply_text_preprocessing: bool,
    output_dir: str,
    batch_size: int,
    max_length: int,
    num_workers: int,
    contact_info: str,
    num_shards: int,
    shard_index: int,
) -> None:
    """Evaluate FLAIRD models on a dataset shard."""

    if shard_index >= num_shards:
        raise click.UsageError(
            f"--shard-index must be between 0 and {num_shards - 1} "
            f"when --num-shards={num_shards}."
        )

    set_seed(SEED)

    output = Path(output_dir)

    click.echo(
        f"Loading {dataset_path}/{dataset_name} ..."
    )

    dataset = load_dataset(
        dataset_path,
        name=dataset_name,
        split="test",
    )

    click.echo(
        f"Total test rows: {len(dataset):,}"
    )

    shard = get_shard(
        dataset=dataset,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    click.echo(
        f"Shard {shard_index + 1}/{num_shards}: "
        f"{len(shard):,} rows "
        f"({len(shard) / len(dataset) * 100:.2f}%)"
    )

    models = (
        DEFAULT_MODELS
        if all_models
        else [model_id]
    )

    for current_model in models:

        model_name = current_model.rsplit("/", 1)[-1]

        click.echo("")
        click.echo("=" * 80)
        click.echo(f"Model: {model_name}")
        click.echo(
            f"Shard: {shard_index + 1}/{num_shards}"
        )
        click.echo("=" * 80)

        save_metadata(
            output_dir=output,
            model_id=current_model,
            contact_info=contact_info,
        )

        output_file = get_shard_file(
            output_dir=output,
            model_id=current_model,
            shard_index=shard_index,
            num_shards=num_shards,
        )

        # Skip if this shard was already evaluated.
        if output_file.exists():
            click.echo(
                f"Shard already exists. Skipping:\n"
                f"{output_file}"
            )

            merge_shards(
                output_dir=output,
                model_id=current_model,
                num_shards=num_shards,
            )

            continue

        scores = predict(
            dataset=shard,
            model_id=current_model,
            feature_column=feature_column,
            apply_text_preprocessing=apply_text_preprocessing,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
        )

        save_shard_predictions(
            file_path=output_file,
            dataset=shard,
            scores=scores,
        )

        merge_shards(
            output_dir=output,
            model_id=current_model,
            num_shards=num_shards,
        )

    click.echo("")
    click.echo("Evaluation completed.")


if __name__ == "__main__":
    main()
