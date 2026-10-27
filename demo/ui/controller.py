"""Application logic of the dashboard: model cache, example loading and the analyze action."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import spaces

from flaird.utils.explain import ExplainConfig, FlairdExplainer, FlairdExplanation

from . import components as ui
from . import plots
from .theme import TOKENS

logger = logging.getLogger("flaird.demo")

MAX_CHARS = 30_000
PRECISION_EXACT = "Exact (recommended)"
PRECISION_FAST = "Fast estimate (sampled)"
PRECISION_CHOICES = [PRECISION_EXACT, PRECISION_FAST]

Loader = Callable[[str, str | None], tuple[Any, Any]]


@dataclass(frozen=True)
class ModelOption:
    label: str
    model_id: str


DEFAULT_MODELS = (
    ModelOption(
        "Attention fusion (recommended)",
        "yusr9/flaird-modernbert-large-attention-multitask",
    ),
    ModelOption(
        "Concatenation fusion",
        "yusr9/flaird-modernbert-large-concatenation-multitask",
    ),
)


class ModelRegistry:
    """Loads checkpoints lazily and keeps at most ``max_loaded`` in memory."""

    def __init__(
        self, loader: Loader = FlairdExplainer.load_components, max_loaded: int = 1
    ):
        self._loader = loader
        self._max_loaded = max_loaded
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def get(self, model_id: str) -> tuple[Any, Any]:
        with self._lock:
            if model_id not in self._cache:
                while len(self._cache) >= self._max_loaded:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[model_id] = self._loader(model_id, None)
            return self._cache[model_id]


@dataclass
class View:
    """Values of every dashboard output component, in display order."""

    hero: str
    notice: str
    balance_panel: str
    balance_plot: Any
    generator_panel: str
    generator_plot: Any
    score_plot: Any
    score_meta: str
    reliability_plot: Any
    reliability_meta: str
    gate_plot: Any
    gate_meta: str
    waterfall_plot: Any
    waterfall_caption: str
    sunburst_plot: Any
    sunburst_caption: str
    attention_plot: Any
    attention_caption: str
    groups_plot: Any
    groups_caption: str
    indicators: str
    features_table: pd.DataFrame

    def as_tuple(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))


OUTPUT_FIELDS = [f.name for f in fields(View)]


def empty_view(message: str = "", kind: str = "info") -> View:
    """Initial / error state: placeholders everywhere, optional notice."""
    waiting = plots.empty_figure("Run an analysis to see this chart.", 220)
    return View(
        hero=ui.hero_html(None),
        notice=ui.notice(kind, message),
        balance_panel="",
        balance_plot=plots.empty_figure("Run an analysis to see this chart.", 250),
        generator_panel="",
        generator_plot=plots.empty_figure("Run an analysis to see this chart.", 280),
        score_plot=plots.donut(0.0, TOKENS["primary"], "—", "machine score"),
        score_meta="",
        reliability_plot=plots.donut(0.0, TOKENS["good"], "—", "reliability"),
        reliability_meta="",
        gate_plot=plots.donut(0.0, TOKENS["accent"], "—", "mean fusion gate"),
        gate_meta="",
        waterfall_plot=plots.empty_figure("Run an analysis to see this chart.", 320),
        waterfall_caption="",
        sunburst_plot=plots.empty_figure("Run an analysis to see this chart.", 420),
        sunburst_caption="",
        attention_plot=waiting,
        attention_caption="",
        groups_plot=plots.empty_figure("Run an analysis to see this chart.", 320),
        groups_caption="",
        indicators=ui.empty_indicators(),
        features_table=ui.empty_features_dataframe(),
    )


def build_view(
    explanation: FlairdExplanation, top_k: int = 10, extra_notice: str = ""
) -> View:
    """Turn an explanation into the values of all dashboard components."""
    d, b, diag = explanation.decision, explanation.balance, explanation.input
    reliability_color = {
        "high": TOKENS["good"],
        "medium": TOKENS["fair"],
        "low": TOKENS["poor"],
    }
    reliability_value = {"high": 1.0, "medium": 0.6, "low": 0.25}[diag.reliability]
    word = ui.RELIABILITY_STYLE[diag.reliability][1]
    gate = b.gate
    return View(
        hero=ui.hero_html(explanation),
        notice=ui.notice("info", extra_notice),
        balance_panel=ui.balance_panel(b),
        balance_plot=plots.balance_donut(b),
        generator_panel=ui.generator_panel(explanation.generator),
        generator_plot=(
            plots.generator_bars(explanation.generator)
            if explanation.generator
            else plots.empty_figure("This checkpoint has no generator head.", 280)
        ),
        score_plot=plots.donut(
            d.machine_score,
            ui.VERDICT_COLOR[d.verdict_band],
            f"{d.machine_score * 100:.0f}%",
            "machine score",
        ),
        score_meta=ui.score_meta(d.machine_score, d.verdict_band, d.verdict),
        reliability_plot=plots.donut(
            reliability_value,
            reliability_color[diag.reliability],
            word,
            "input reliability",
        ),
        reliability_meta=ui.reliability_meta(diag),
        gate_plot=(
            plots.donut(gate, TOKENS["primary"], f"{gate:.2f}", "mean fusion gate")
            if gate is not None
            else plots.donut(
                0.0, TOKENS["muted_series"], "n/a", "attention fusion only"
            )
        ),
        gate_meta=ui.gate_meta(b),
        waterfall_plot=plots.waterfall(b, explanation.groups),
        waterfall_caption=ui.caption_waterfall(b),
        sunburst_plot=plots.sunburst(explanation.groups, explanation.features),
        sunburst_caption=ui.caption_features(explanation.features),
        attention_plot=plots.attention_vs_causal(explanation.groups),
        attention_caption=ui.caption_attention(
            explanation.checks.attention_vs_phi_spearman
        ),
        groups_plot=plots.group_contributions(explanation.groups),
        groups_caption=ui.caption_groups(explanation.groups),
        indicators=ui.indicators_html(explanation.features, explanation.groups, top_k),
        features_table=ui.features_dataframe(explanation.features, explanation.groups),
    )


class DashboardController:
    """Glue between the Gradio events and the explanation module."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        models: tuple[ModelOption, ...] = DEFAULT_MODELS,
        examples_path: Path | str | None = None,
    ):
        self.registry = registry or ModelRegistry()
        self.models = {m.label: m for m in models}
        self.examples = self._load_examples(examples_path)

    # -- examples ----------------------------------------------------------------------
    @staticmethod
    def _load_examples(path: Path | str | None) -> pd.DataFrame:
        if path is None or not Path(path).is_file():
            return pd.DataFrame(
                columns=["text", "attack", "model", "label", "generator_label"]
            )
        return pd.read_csv(path)

    def example_choices(self) -> list[str]:
        return [
            f"Example {i + 1}: {row.generator_label}, {row.attack}"
            for i, row in enumerate(self.examples.itertuples())
        ]

    def load_example(self, choice: str | None) -> tuple[str, str]:
        """Return the example text and its ground-truth chips."""
        if not choice:
            return "", ""
        index = int(choice.split(":")[0].split()[1]) - 1
        row = self.examples.iloc[index]
        return str(row["text"]), ui.example_chip(row.to_dict())

    # -- analysis ----------------------------------------------------------------------
    @spaces.GPU
    def analyze(self, text: str, model_label: str, precision: str) -> tuple:
        text = (text or "").strip()
        if not text:
            return empty_view(
                "Please paste a text or pick an example first.", "error"
            ).as_tuple()
        note = ""
        if len(text) > MAX_CHARS:
            text, note = (
                text[:MAX_CHARS],
                f"Text shortened to the first {MAX_CHARS:,} characters.",
            )
        option = self.models.get(model_label) or next(iter(self.models.values()))
        try:
            model, tokenizer = self.registry.get(option.model_id)
            config = ExplainConfig(
                owen_method="permutation" if precision == PRECISION_FAST else "exact"
            )
            explainer = FlairdExplainer(
                model, tokenizer, config, model_id=option.model_id
            )
            explanation = explainer.explain(text)
            return build_view(explanation, config.top_k_features, note).as_tuple()
        except ValueError as error:
            return empty_view(
                f"<b>Cannot analyze this input.</b> {error}", "error"
            ).as_tuple()
        except Exception as error:
            logger.exception("Analysis failed")
            return empty_view(
                f"<b>The analysis failed.</b> {type(error).__name__}: {error}", "error"
            ).as_tuple()
