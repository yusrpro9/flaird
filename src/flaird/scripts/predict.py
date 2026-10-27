"""Command line interface: predict and explain machine-generated text with FLAIRD.

Example::

    flaird-predict --model MahmoodAnaam/flaird-modernbert-large-attention-multitask \\
                   --text "Some text to analyze ..." --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flaird.utils.explain import (
    ExplainConfig,
    FlairdExplainer,
    FlairdExplanation,
    explain_to_markdown,
)

DEFAULT_MODEL = "MahmoodAnaam/flaird-modernbert-large-attention-multitask"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flaird-predict",
        description="Predict whether a text is machine-generated and explain the decision.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to analyze.")
    source.add_argument(
        "--input",
        type=Path,
        help="A .txt file (one text) or .jsonl file (field 'text' per line).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument(
        "--generator-target",
        default=None,
        help="Generator class to attribute (default: the predicted class). Multitask models only.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the full explanation as JSON/JSONL."
    )
    parser.add_argument(
        "--report", action="store_true", help="Print a readable report."
    )
    parser.add_argument("--device", default=None, help="Device, e.g. cpu or cuda.")
    parser.add_argument(
        "--owen-method",
        choices=["exact", "permutation"],
        default="exact",
        help="Feature-attribution method (exact is the default).",
    )
    parser.add_argument("--owen-permutations", type=int, default=512)
    return parser


def read_texts(args: argparse.Namespace) -> tuple[list[str], bool]:
    """Return the input texts and whether the input was JSONL."""
    if args.text is not None:
        return [args.text], False
    path: Path = args.input
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        texts = []
        for number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Line {number}: expected a JSON object with 'text'."
                ) from error
        if not texts:
            raise ValueError("The JSONL file contains no texts.")
        return texts, True
    return [content], False


def write_output(
    path: Path, explanations: list[FlairdExplanation], jsonl: bool
) -> None:
    if jsonl:
        body = "\n".join(e.to_json() for e in explanations) + "\n"
    else:
        body = explanations[0].to_json(indent=2)
    path.write_text(body, encoding="utf-8")


def main(
    argv: list[str] | None = None, explainer: FlairdExplainer | None = None
) -> int:
    """CLI entry point. ``explainer`` can be injected (used by tests)."""
    args = build_parser().parse_args(argv)
    try:
        texts, jsonl = read_texts(args)
        if explainer is None:
            config = ExplainConfig(
                device=args.device,
                owen_method=args.owen_method,
                owen_permutations=args.owen_permutations,
            )
            explainer = FlairdExplainer.from_pretrained(args.model, config)
        explanations = [explainer.explain(t, args.generator_target) for t in texts]
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output:
        write_output(args.output, explanations, jsonl)
    if args.report or not args.output:
        print(
            "\n\n".join(
                explain_to_markdown(e, explainer.config.top_k_features)
                for e in explanations
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
