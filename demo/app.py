from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from ui import components as ui
from ui.controller import (
    OUTPUT_FIELDS,
    PRECISION_CHOICES,
    PRECISION_EXACT,
    DashboardController,
    empty_view,
)
from ui.theme import HEAD, build_css, build_theme

EXAMPLES_PATH = Path(__file__).parent / "examples.csv"
FOOTER = (
    "FLAIRD is a decision-support tool. Scores are prior-adjusted model outputs, not calibrated "
    "probabilities, and explanations describe the model's behavior, not ground truth."
)


def _card(title: str, subtitle: str = "") -> gr.Column:
    """Open a white card with a title; use as a context manager."""
    column = gr.Column(elem_classes="fx-card")
    with column:
        gr.HTML(
            f'<div class="fx-title">{title}</div>'
            + (f'<div class="fx-sub">{subtitle}</div>' if subtitle else "")
        )
    return column


def build_app(controller: DashboardController | None = None) -> gr.Blocks:
    controller = controller or DashboardController(examples_path=EXAMPLES_PATH)
    initial = empty_view()
    components: dict[str, gr.components.Component] = {}

    with gr.Blocks(title="FLAIRD - AI-text forensic analysis", fill_width=True) as app:
        gr.HTML(ui.header_html(), elem_classes="fx-bleed")
        components["hero"] = gr.HTML(initial.hero, elem_classes="fx-bleed")

        with gr.Column(elem_classes="fx-page"):
            components["notice"] = gr.HTML(initial.notice)
            with gr.Tabs(elem_classes="fx-tabs"):
                # ------------------------------------------------------------ Dashboard
                with gr.Tab("Dashboard"):
                    with _card(
                        "Analyze a text",
                        "Paste a text of at least a few sentences, or pick an example.",
                    ):
                        with gr.Row(elem_classes="fx-row"):
                            with gr.Column(scale=3, min_width=340):
                                text = gr.Textbox(
                                    label="Text to analyze",
                                    lines=9,
                                    max_lines=14,
                                    placeholder="Paste the text you want to check ...",
                                )
                            with gr.Column(scale=2, min_width=300):
                                model = gr.Dropdown(
                                    choices=list(controller.models),
                                    value=next(iter(controller.models)),
                                    label="Model",
                                    info="Both are the released multitask checkpoints.",
                                )
                                precision = gr.Radio(
                                    PRECISION_CHOICES,
                                    value=PRECISION_EXACT,
                                    label="Feature attribution",
                                    info="Exact enumerates ~78,000 combinations; the sampled estimate is faster on CPU.",
                                )
                                example = gr.Dropdown(
                                    choices=controller.example_choices(),
                                    value=None,
                                    label="Example",
                                    info="Loads a text from the project's example set.",
                                )
                        example_chip = gr.HTML("")
                        with gr.Row():
                            analyze = gr.Button("Analyze", variant="primary", scale=3)
                            clear = gr.Button("Clear", scale=1)

                    with gr.Row(elem_classes="fx-row"):
                        with _card(
                            "Path balance",
                            "How meaning and writing style contributed to the decision",
                        ):
                            components["balance_panel"] = gr.HTML(initial.balance_panel)
                            components["balance_plot"] = gr.Plot(
                                initial.balance_plot, show_label=False, container=False
                            )
                        with _card(
                            "Generator distribution",
                            "Estimated model family of the text",
                        ):
                            components["generator_panel"] = gr.HTML(
                                initial.generator_panel
                            )
                            components["generator_plot"] = gr.Plot(
                                initial.generator_plot,
                                show_label=False,
                                container=False,
                            )

                    with gr.Row(elem_classes="fx-row"):
                        with _card("Detection score"):
                            components["score_plot"] = gr.Plot(
                                initial.score_plot, show_label=False, container=False
                            )
                            components["score_meta"] = gr.HTML(initial.score_meta)
                        with _card("Input reliability"):
                            components["reliability_plot"] = gr.Plot(
                                initial.reliability_plot,
                                show_label=False,
                                container=False,
                            )
                            components["reliability_meta"] = gr.HTML(
                                initial.reliability_meta
                            )
                        with _card("Fusion gate"):
                            components["gate_plot"] = gr.Plot(
                                initial.gate_plot, show_label=False, container=False
                            )
                            components["gate_meta"] = gr.HTML(initial.gate_meta)

                # ------------------------------------------------------------ Reasoning
                with gr.Tab("Reasoning"):
                    with _card(
                        "From semantic evidence to the final decision",
                        "Each step shows how a feature group moved the machine logit",
                    ):
                        components["waterfall_plot"] = gr.Plot(
                            initial.waterfall_plot, show_label=False, container=False
                        )
                        components["waterfall_caption"] = gr.HTML(
                            initial.waterfall_caption
                        )
                    with gr.Row(elem_classes="fx-row"):
                        with _card(
                            "Groups and features",
                            "Inner ring: groups. Outer ring: features, sized by absolute contribution",
                        ):
                            components["sunburst_plot"] = gr.Plot(
                                initial.sunburst_plot, show_label=False, container=False
                            )
                            components["sunburst_caption"] = gr.HTML(
                                initial.sunburst_caption
                            )
                        with _card(
                            "Attention versus contribution",
                            "Where the model looked compared with what changed the decision",
                        ):
                            components["attention_plot"] = gr.Plot(
                                initial.attention_plot,
                                show_label=False,
                                container=False,
                            )
                            components["attention_caption"] = gr.HTML(
                                initial.attention_caption
                            )

                # ------------------------------------------------------------ Forensic evidence
                with gr.Tab("Forensic Evidence"):
                    with (
                        gr.Row(elem_classes="fx-row"),
                        _card(
                            "Feature groups",
                            "Positive pushes toward machine, negative toward human",
                        ),
                    ):
                        components["groups_plot"] = gr.Plot(
                            initial.groups_plot, show_label=False, container=False
                        )
                        components["groups_caption"] = gr.HTML(initial.groups_caption)
                    with _card(
                        "Top forensic indicators",
                        "Ranked by absolute contribution to the decision",
                    ):
                        components["indicators"] = gr.HTML(initial.indicators)
                    with _card("All 35 features", "Click a column header to sort"):
                        components["features_table"] = gr.Dataframe(
                            value=initial.features_table,
                            interactive=False,
                            wrap=True,
                            show_label=False,
                            max_height=520,
                            column_widths=[
                                "70px",
                                "210px",
                                "190px",
                                "90px",
                                "120px",
                                "100px",
                                "380px",
                            ],
                        )

                # ------------------------------------------------------------ Method
                with gr.Tab("Method"), gr.Column(elem_classes="fx-card"):
                    gr.Markdown(ui.METHOD_MARKDOWN, elem_classes="fx-prose")

        gr.HTML(f'<div class="fx-footer">{FOOTER}</div>')

        outputs = [components[name] for name in OUTPUT_FIELDS]

        def run(text_value, model_label, precision_value):
            return controller.analyze(text_value, model_label, precision_value)

        analyze.click(run, [text, model, precision], outputs, show_progress="minimal")
        example.change(
            controller.load_example,
            example,
            [text, example_chip],
            show_progress="minimal",
        )
        clear.click(
            lambda: ("", None, "", *empty_view().as_tuple()),
            None,
            [text, example, example_chip, *outputs],
            show_progress="minimal",
        )
    return app


def main() -> None:
    app = build_app()
    app.queue(max_size=8).launch(theme=build_theme(), css=build_css(), head=HEAD)


if __name__ == "__main__":
    main()
