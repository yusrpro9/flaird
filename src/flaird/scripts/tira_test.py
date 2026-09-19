import os
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

from flaird.data.data_collator import DataCollator


def resolve_input_path(input_directory: str | None, input_file: str | None) -> Path:
    """Resolve the JSONL input exposed by TIRA or supplied via the legacy CLI."""
    if input_file is not None:
        return Path(input_file)

    if input_directory is None:
        raise click.UsageError(
            "Missing input. Provide --input-directory or set inputDataset."
        )

    input_dir = Path(input_directory)
    for filename in ("dataset.jsonl", "input.jsonl"):
        candidate = input_dir / filename
        if candidate.is_file():
            return candidate

    candidates = sorted(path for path in input_dir.rglob("*.jsonl") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]

    available = ", ".join(str(path.relative_to(input_dir)) for path in candidates)
    raise click.UsageError(
        "Could not identify a single JSONL input file in "
        f"{input_dir}. Candidates: {available or 'none'}."
    )


def test(
    dataset: Dataset,
    model_path: str,
    device: str | torch.device = "auto",
    batch_size: int = 8,
    text_column: str = "text",
    feature_column: str | None = None,
    apply_text_preprocessing: bool = True,
    score_column: str = "label",
) -> pd.DataFrame:
    device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    predictions_df = pd.DataFrame({"id": list(dataset["id"])})
    data_collator = DataCollator(
        tokenizer=tokenizer,
        text_column=text_column,
        feature_column=feature_column,
        max_length=512,
        use_forensic_features=True,
        apply_text_preprocessing=apply_text_preprocessing,
        include_labels=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )
    scores = []
    for batch in tqdm(test_loader, desc="Predicting on test dataset"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
        batch_scores = torch.sigmoid(outputs.logits).reshape(-1).cpu().tolist()
        scores.extend(batch_scores)

    predictions_df[score_column] = scores
    return predictions_df


@click.command()
@click.option(
    "--input-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=None,
    help="Directory containing the task input file dataset.jsonl.",
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=None,
    help="Directory where predictions.jsonl will be written.",
)
@click.option(
    "--model",
    default="MahmoodAnaam/flaird-modernbert-large-attention-multitask-frozen",
    required=False,
    help="The model to use",
)
@click.argument(
    "input_file",
    required=False,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument(
    "legacy_output_directory",
    required=False,
    type=click.Path(file_okay=False, writable=True),
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Inference batch size.",
)
def main(
    input_directory,
    output_directory,
    input_file,
    legacy_output_directory,
    model,
    batch_size,
):
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    # TIRA exposes these variables in the runtime environment.
    input_directory = input_directory or os.getenv("inputDataset")
    output_directory = output_directory or os.getenv("outputDir")

    # Backward compatibility with previous positional CLI usage.
    if input_file is not None:
        input_path = Path(input_file)
        output_directory = output_directory or legacy_output_directory
    input_path = resolve_input_path(input_directory, input_file)

    if output_directory is None:
        raise click.UsageError(
            "Missing output directory. Provide --output-directory or set outputDir environment variable."
        )

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = Dataset.from_json(str(input_path))

    predictions_df = test(
        dataset=test_ds,
        model_path=model,
        device=device,
        batch_size=batch_size,
        text_column="text",
        feature_column=None,
        apply_text_preprocessing=True,
        score_column="label",
    )

    output_path = Path(output_directory) / "predictions.jsonl"
    predictions_df.to_json(output_path, orient="records", lines=True)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
