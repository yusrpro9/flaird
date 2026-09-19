import os
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from flaird.data.data_collator import DataCollator


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Main prediction pipeline for test set
# ---------------------------------------------------------------------------


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
    test_df: pd.DataFrame,
    model_path: str,
    device: str | torch.device = "auto",
    batch_size: int = 8,
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

    test_ds = Dataset.from_pandas(test_df)

    data_collator = DataCollator(tokenizer, include_labels=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )
    scores = []
    for batch in tqdm(test_loader, desc="Predicting on test dataset"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
        batch_scores = torch.sigmoid(outputs.logits).reshape(-1).cpu().tolist()
        scores.extend(batch_scores)

    predictions_df = pd.DataFrame({"id": test_df["id"].tolist(), "label": scores})
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
    default="yusr9/flaird-modernbert-large-attention-multitask-frozen",
    required=False,
    help="The model to use",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Inference batch size.",
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
def main(
    input_directory,
    output_directory,
    input_file,
    legacy_output_directory,
    model,
    batch_size,
):
    set_seed(RANDOM_SEED)
    # TIRA exposes these variables in the runtime environment.
    input_directory = input_directory or os.getenv("inputDataset")
    output_directory = output_directory or os.getenv("outputDir")

    if input_file is not None:
        output_directory = output_directory or legacy_output_directory
    input_path = resolve_input_path(input_directory, input_file)

    if output_directory is None:
        raise click.UsageError(
            "Missing output directory. Provide --output-directory or set outputDir environment variable."
        )

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_json(input_path, lines=True)
    if "id" not in test_df.columns:
        test_df["id"] = test_df.index

    predictions_df = test(
        test_df,
        model_path=model,
        device=device,
        batch_size=batch_size,
    )
    output_path = Path(output_directory) / "predictions.jsonl"
    predictions_df.to_json(output_path, orient="records", lines=True)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
