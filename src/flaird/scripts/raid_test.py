"""Generate RAID leaderboard submissions for FLAIRD detectors."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import click
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from flaird.data.data_collator import DataCollator

DEFAULT_USERNAME = os.getenv("HF_USERNAME", "yusr9")
DEFAULT_MODEL = f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask"
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


def create_metadata(
    detector_name: str,
    model_id: str,
    contact_info: str,
) -> dict[str, str]:
    """Return metadata in the format expected by the RAID leaderboard."""
    return {
        "date_released": date.today().isoformat(),
        "detector_name": detector_name,
        "huggingface_link": f"https://huggingface.co/{model_id}",
        "contact_info": contact_info,
    }


def _scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert binary classifier logits to P(machine-generated)."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1))
    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise RuntimeError(f"Expected one or two output logits, got shape {tuple(logits.shape)}")


def predict(
    dataset: Dataset,
    model_id: str,
    feature_column: str | None,
    apply_text_preprocessing: bool,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> list[float]:
    """Run batched inference for one model and return scores in dataset order."""
    if "generation" not in dataset.column_names:
        raise click.ClickException("The RAID dataset must contain a 'generation' column.")
    if feature_column and feature_column not in dataset.column_names:
        raise click.ClickException(
            f"Feature column '{feature_column}' is not present in the dataset. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    click.echo(f"Loading {model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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

    scores: list[float] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Predicting {model_id.split('/')[-1]}", unit="batch"):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            batch_scores = _scores_from_logits(model(**batch).logits)
            scores.extend(batch_scores.float().cpu().tolist())

    if len(scores) != len(dataset):
        raise RuntimeError(
            f"Model returned {len(scores)} scores for {len(dataset)} dataset rows."
        )
    return scores


def write_submission(
    output_dir: Path,
    model_id: str,
    ids: list[Any],
    scores: list[float],
    contact_info: str,
) -> None:
    """Write the RAID-required predictions.json and metadata.json files."""
    detector_name = model_id.rsplit("/", 1)[-1]
    detector_dir = output_dir / detector_name
    detector_dir.mkdir(parents=True, exist_ok=True)


    predictions_df = pd.DataFrame({"id": ids, "score": scores})
    predictions_file = detector_dir / "predictions.json"
    predictions_df.to_json(predictions_file, orient="records",lines=True)

    metadata = create_metadata(detector_name, model_id, contact_info)
    metadata_file = detector_dir / "metadata.json"
    with (metadata_file).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )
    click.echo(f"Wrote {predictions_file}")
    click.echo(f"Wrote {metadata_file}")


@click.command()
@click.option("--model", "model_id", default=DEFAULT_MODEL, show_default=True)
@click.option("--all-models", is_flag=True, help="Run all eight trained FLAIRD models.")
@click.option("--dataset-path", default="liamdugan/raid", show_default=True)
@click.option("--dataset-name", default="raid_test", show_default=True)
@click.option("--feature-column", default=None)
@click.option("--apply-text-preprocessing", is_flag=True)
@click.option("--output-dir", type=click.Path(file_okay=False), default="raid_submissions", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--max-length", type=click.IntRange(min=1), default=512, show_default=True)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0),
    default=min(4, os.cpu_count() or 1),
    show_default=True,
    help="DataLoader workers; use 0 when running in a constrained notebook.",
)
@click.option("--contact-info", default="Email Address: yusrpro9@gmail.com", show_default=True)
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
) -> None:
    """Generate one RAID submission directory per selected FLAIRD model."""
    set_seed(SEED)
    dataset = load_dataset(dataset_path, name=dataset_name, split="test")
    ids = list(dataset["id"]) if "id" in dataset.column_names else list(range(len(dataset)))
    models = DEFAULT_MODELS if all_models else [model_id]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loaded {len(dataset)} rows from {dataset_path}/{dataset_name}")
    for current_model in models:
        scores = predict(
            dataset=dataset,
            model_id=current_model,
            feature_column=feature_column,
            apply_text_preprocessing=apply_text_preprocessing,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
        )

        write_submission(output, current_model, ids, scores, contact_info)

    click.echo(f"Completed {len(models)} detector submission(s) in {output}")


if __name__ == "__main__":
    main()
