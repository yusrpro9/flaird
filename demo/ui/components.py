"""HTML snippets and tables of the dashboard (pure functions, no Gradio state)."""

from __future__ import annotations

from html import escape

import pandas as pd

from flaird.utils.explain import (
    FeatureAttribution,
    FlairdExplanation,
    GeneratorAnalysis,
    GroupAttribution,
    InputDiagnostics,
    PathBalance,
)

from .theme import TOKENS

RELIABILITY_STYLE = {
    "high": ("good", "Good"),
    "medium": ("fair", "Fair"),
    "low": ("poor", "Poor"),
}
VERDICT_COLOR = {
    "machine": TOKENS["machine"],
    "human": TOKENS["human"],
    "inconclusive": TOKENS["fair"],
}


def _sign_class(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


def header_html() -> str:
    return """
<div class="fx-header"><div class="fx-inner">
  <div class="fx-brand"><span class="fx-logomark" aria-hidden="true">F</span>
    <div class="fx-logo">FLAIRD<small>Forensic Linguistic and AI-Integrated Robust Detector</small></div></div>
  <nav aria-label="Project links">
    <a href="https://github.com/yusrpro9/flaird" target="_blank" rel="noopener">GitHub</a>
    <a href="https://huggingface.co/collections/yusr9/flaird" target="_blank" rel="noopener">Model collection ↗</a>
  </nav>
</div></div>"""


def _kpi(
    label: str, value: str, sub: str = "", small: bool = False, dot: str | None = None
) -> str:
    marker = f'<span class="fx-dot" style="background:{dot}"></span>' if dot else ""
    cls = "val small" if small else "val"
    return (
        f'<div class="fx-kpi"><div class="lab">{escape(label)}</div>'
        f'<div class="{cls}">{marker}{escape(value)}</div>'
        f'<div class="sub">{escape(sub) if sub else "&nbsp;"}</div></div>'
    )


def hero_html(explanation: FlairdExplanation | None = None) -> str:
    """Research header with four high-contrast result tiles and initial placeholders."""
    if explanation is None:
        tiles = [
            _kpi("Machine score", "—", "run an analysis"),
            _kpi("Verdict", "—", "", small=True),
            _kpi("Input reliability", "—", "", small=True),
            _kpi("Predicted generator", "—", "", small=True),
        ]
    else:
        d, g = explanation.decision, explanation.generator
        level, word = RELIABILITY_STYLE[explanation.input.reliability]
        reliability_color = TOKENS[level]
        tiles = [
            _kpi(
                "Machine score",
                f"{d.machine_score * 100:.1f} %",
                f"logit {d.logit:+.2f}",
            ),
            _kpi(
                "Verdict",
                d.verdict,
                "prior-adjusted score",
                small=True,
                dot=VERDICT_COLOR[d.verdict_band],
            ),
            _kpi(
                "Input reliability",
                word,
                f"{explanation.input.num_words} words",
                small=True,
                dot=reliability_color,
            ),
            _kpi(
                "Predicted generator",
                g.predicted if g else "—",
                f"{g.probs[g.predicted] * 100:.0f} % probability"
                if g
                else "no generator head",
                small=True,
            ),
        ]
    return f"""
<div class="fx-hero"><div class="fx-inner">
  <div class="fx-eyebrow">RESEARCH DEMONSTRATOR · ENGLISH TEXT</div>
  <h1>AI-Text Forensic Analysis</h1>
  <p>Detects machine-generated text and explains the decision: how meaning and writing style
  contributed, and which forensic indicators mattered.</p>
  <div class="fx-kpis">{"".join(tiles)}</div>
</div></div>"""


def badge(text: str, kind: str) -> str:
    return f'<span class="fx-badge {kind}">{escape(text)}</span>'


def notice(kind: str, message: str) -> str:
    if not message:
        return ""
    return f'<div class="fx-notice {kind}" role="status">{message}</div>'


def stat(label: str, value: str, css: str = "") -> str:
    return f'<div class="fx-stat"><span class="k">{label}</span><span class="v {css}">{value}</span></div>'


def balance_panel(balance: PathBalance) -> str:
    gate = (
        stat("Mean fusion gate", f"{balance.gate:.2f}")
        if balance.gate is not None
        else stat("Fusion gate", "n/a for this architecture")
    )
    return (
        '<div class="fx-panel"><div class="fx-bigcap">Semantic-anchored decomposition</div>'
        + stat("Baseline-forensic logit", f"{balance.semantic_logit:+.2f}")
        + stat(
            "Forensic contribution",
            f"{balance.forensic_delta:+.2f}",
            _sign_class(balance.forensic_delta),
        )
        + stat("Final logit", f"{balance.final_logit:+.2f}")
        + stat(
            "Baseline / forensic magnitudes",
            f"{balance.semantic_share:.0%} / {balance.forensic_share:.0%}",
        )
        + gate
        + "</div>"
    )


def generator_panel(generator: GeneratorAnalysis | None) -> str:
    if generator is None:
        return '<div class="fx-panel"><div class="fx-bigcap">This checkpoint has no generator head.</div></div>'
    rows = "".join(
        stat(escape(label), f"{generator.probs[label] * 100:.1f} %")
        for label in generator.top3
    )
    if generator.consistent is False:
        flag = '<div class="fx-warn">The binary and generator heads disagree on this text.</div>'
    elif generator.consistent:
        flag = '<div class="fx-ok">Binary and generator heads agree.</div>'
    else:
        flag = ""
    return (
        '<div class="fx-panel"><div class="fx-bigcap">Top three model families (estimate)</div>'
        + rows
        + stat("Entropy", f"{generator.entropy:.2f}")
        + "</div>"
        + flag
    )


def score_meta(machine_score: float, band: str, verdict: str) -> str:
    kind = {"machine": "poor", "human": "good", "inconclusive": "fair"}[band]
    return (
        '<div class="fx-gauge-meta">'
        + badge(verdict, kind)
        + '<div class="fx-caption">Decision threshold 0.5; scores between 0.35 and 0.65 are '
        "reported as inconclusive. The score is prior-adjusted, not a calibrated probability.</div></div>"
    )


def reliability_meta(diag: InputDiagnostics) -> str:
    kind, word = RELIABILITY_STYLE[diag.reliability]
    warnings = "".join(
        f'<div class="fx-warn">{escape(w)}</div>' for w in diag.warnings[:4]
    )
    ok = "" if diag.warnings else '<div class="fx-ok">No input problems detected.</div>'
    return (
        f'<div class="fx-gauge-meta">{badge(word, kind)}</div>'
        f'<div class="fx-caption">{diag.num_words} words, {diag.num_tokens} tokens '
        f"({diag.coverage:.0%} read by the encoder)</div>{warnings}{ok}"
    )


def gate_meta(balance: PathBalance) -> str:
    if balance.gate is None:
        return '<div class="fx-caption">The fusion gate exists for attention-fusion models only.</div>'
    gate = balance.gate
    word, kind = (
        ("Semantic-led", "info")
        if gate >= 0.66
        else ("Forensic-led", "teal")
        if gate <= 0.33
        else ("Balanced", "slate")
    )
    return (
        f'<div class="fx-gauge-meta">{badge(word, kind)}</div>'
        '<div class="fx-caption">Mean gate before the final fusion projection: values near 1 favor '
        "the semantic state in the gated mixture; values near 0 favor the forensic context. "
        "This is not a percentage of decision importance.</div>"
    )


# ---- captions generated from the data ----------------------------------------------------
def _side(phi: float) -> str:
    return "toward machine" if phi > 0 else "toward human"


def caption_groups(groups: list[GroupAttribution]) -> str:
    top = max(groups, key=lambda g: abs(g.phi))
    machine = sum(g.phi > 0 for g in groups)
    return (
        f"<b>{escape(top.display_name)}</b> contributed most, pushing {_side(top.phi)} "
        f"({top.phi:+.2f} logit). {machine} of {len(groups)} groups push toward machine."
    )


def caption_waterfall(balance: PathBalance) -> str:
    return (
        f"With all raw forensic features set to zero the model would output <b>{balance.semantic_logit:+.2f}</b>; "
        f"the forensic stream adds <b>{balance.forensic_delta:+.2f}</b>, giving a final logit of "
        f"<b>{balance.final_logit:+.2f}</b>."
    )


def caption_features(features: list[FeatureAttribution]) -> str:
    top = min(features, key=lambda f: f.abs_rank)
    return (
        f"Strongest single indicator: <b>{escape(top.name)}</b> ({top.phi:+.3f}, "
        f"{_side(top.phi)}). Arrows show the direction of each feature."
    )


def caption_attention(rho: float | None) -> str:
    if rho is None:
        return "Attention shows what the model read; contribution shows what changed the decision."
    return (
        "Attention shows what the model read; contribution shows what changed the decision. "
        f"Rank agreement (Spearman): <b>{rho:+.2f}</b>."
    )


# ---- tables -------------------------------------------------------------------------------
def indicators_html(
    features: list[FeatureAttribution], groups: list[GroupAttribution], top_k: int = 10
) -> str:
    colors = {g.name: (g.display_name, g.color) for g in groups}
    ranked = sorted(features, key=lambda f: f.abs_rank)[:top_k]
    span = max((abs(f.phi) for f in ranked), default=1.0) or 1.0
    rows = []
    for rank, f in enumerate(ranked, start=1):
        name, color = colors.get(f.group, ("—", TOKENS["muted_series"]))
        width = min(50.0, abs(f.phi) / span * 50.0)
        side = "machine" if f.phi > 0 else "human"
        bar = f'<div class="fx-bar"><span class="{side}" style="width:{width:.1f}%"></span></div>'
        rows.append(
            f'<tr><td class="num">{rank}</td>'
            f'<td><span class="feat">{escape(f.name)}</span><span class="desc">{escape(f.description)}</span></td>'
            f'<td><span class="fx-groupchip"><i style="background:{color}"></i>{escape(name)}</span></td>'
            f'<td class="num">{f.value:.4g}</td><td>{bar}</td><td class="num">{f.phi:+.3f}</td>'
            f'<td><span class="fx-pill {f.direction}">{f.direction}</span></td></tr>'
        )
    head = (
        '<th class="num">#</th><th>Feature</th><th>Group</th><th class="num">Value</th>'
        '<th colspan="2">Contribution</th><th>Direction</th>'
    )
    return f'<div class="fx-tablewrap"><table class="fx-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def features_dataframe(
    features: list[FeatureAttribution], groups: list[GroupAttribution]
) -> pd.DataFrame:
    names = {g.name: g.display_name for g in groups}
    rows = [
        {
            "Rank": f.abs_rank,
            "Feature": f.name,
            "Group": names.get(f.group, "-"),
            "Value": round(f.value, 4),
            "Contribution": round(f.phi, 4),
            "Direction": f.direction,
            "Meaning": f.description,
        }
        for f in sorted(features, key=lambda f: f.abs_rank)
    ]
    return pd.DataFrame(rows)


def empty_features_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Rank",
            "Feature",
            "Group",
            "Value",
            "Contribution",
            "Direction",
            "Meaning",
        ]
    )


def empty_indicators() -> str:
    return '<div class="fx-caption">Run an analysis to see the ranked forensic indicators.</div>'


def example_chip(row: dict) -> str:
    chips = [
        f"Ground truth: {row['label']}",
        f"Generator: {row['generator_label']}",
        f"Attack: {row['attack']}",
        f"Source model: {row['model']}",
    ]
    return "".join(f'<span class="fx-chip">{escape(str(c))}</span>' for c in chips)


METHOD_MARKDOWN = """
### How to read this dashboard

FLAIRD combines two evidence streams: the **semantic** stream (what the text says, from a
ModernBERT encoder) and the **forensic** stream (35 linguistic features in 7 groups, such as
lexical diversity or punctuation habits). The explanation follows the same hierarchy.

**Decision.** The machine score is the sigmoid of the model's logit. It is *prior-adjusted* (the
training objective down-weights the majority class), so it is not a calibrated probability.
Scores between 0.35 and 0.65 are reported as inconclusive.

**Path balance.** The baseline logit is the hybrid model's output with all raw forensic
features set to zero. Learned biases can still produce nonzero feature tokens. The forensic
contribution is the difference from that reference; both add up to the final logit.
The displayed shares compare absolute logit magnitudes, not causal importance percentages.

**Feature groups (Shapley values).** For each group, the contribution is the average change of the
logit when the group is switched on, over all combinations of the other groups. Positive values
push toward *machine*, negative values toward *human*. The values add up exactly to the forensic
contribution.

**Features (Owen values).** The same idea one level deeper: contributions of individual features
inside their group. Exact feature values add up to the group's value. In sampled mode,
features are estimates with sampling error, while group values remain exact.

**Attention versus contribution.** Fusion attention shows where the model *looked*; contribution
shows what *changed* the decision. Attention is not an explanation by itself, which is why both are
shown together.

### Limits

- Contributions are relative to a baseline of zero raw forensic measurements; they explain the model,
  not the ground truth.
- Texts shorter than about 30 words, or very long texts (only the first 512 tokens are read by the
  encoder), are less reliable. Hidden characters and look-alike letters are flagged.
- The generator estimate is a model-family signature, not proof of authorship.
- This is a decision-support tool; it should not be the only evidence in any consequential decision.
"""
