"""Plotly figure builders. Pure functions: explanation data in, ``go.Figure`` out."""

from __future__ import annotations

import plotly.graph_objects as go

from flaird.utils.explain import (
    ConsistencyChecks,
    FeatureAttribution,
    GeneratorAnalysis,
    GroupAttribution,
    PathBalance,
)

from .theme import FONT_STACK, TOKENS

T = TOKENS


def _base(
    fig: go.Figure, height: int, margin: tuple[int, int, int, int] = (8, 8, 8, 8)
) -> go.Figure:
    left, top, right, bottom = margin
    fig.update_layout(
        height=height,
        margin=dict(l=left, t=top, r=right, b=bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=T["text"]),
        hoverlabel=dict(
            font_family=FONT_STACK, bgcolor="#FFFFFF", bordercolor=T["track"]
        ),
        showlegend=False,
        modebar=dict(
            bgcolor="rgba(0,0,0,0)", color=T["muted_series"], activecolor=T["primary"]
        ),
    )
    return fig


def empty_figure(message: str, height: int = 240) -> go.Figure:
    """Placeholder shown before an analysis (or when a component is unavailable)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color=T["text_muted"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height)


def donut(
    value: float, color: str, center: str, caption: str = "", height: int = 190
) -> go.Figure:
    """Single-value donut gauge (value in 0..1)."""
    value = min(1.0, max(0.0, value))
    fig = go.Figure(
        go.Pie(
            values=[value, 1 - value],
            hole=0.74,
            sort=False,
            direction="clockwise",
            marker=dict(colors=[color, T["track"]], line=dict(width=0)),
            textinfo="none",
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        text=f"<b>{center}</b>",
        x=0.5,
        y=0.54,
        showarrow=False,
        font=dict(size=24, color=T["primary_dark"]),
    )
    if caption:
        fig.add_annotation(
            text=caption,
            x=0.5,
            y=0.36,
            showarrow=False,
            font=dict(size=11, color=T["text_muted"]),
        )
    return _base(fig, height)


def balance_donut(balance: PathBalance, height: int = 250) -> go.Figure:
    """Absolute baseline/delta magnitudes; these are not causal decision shares."""
    fig = go.Figure(
        go.Pie(
            labels=["Baseline logit magnitude", "Forensic change magnitude"],
            values=[balance.semantic_share, balance.forensic_share],
            hole=0.68,
            sort=False,
            direction="clockwise",
            marker=dict(
                colors=[T["primary"], T["accent"]], line=dict(color="#FFFFFF", width=3)
            ),
            textinfo="none",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{balance.final_logit:+.2f}</b>",
        x=0.5,
        y=0.56,
        showarrow=False,
        font=dict(size=28, color=T["primary_dark"]),
    )
    fig.add_annotation(
        text="final logit",
        x=0.5,
        y=0.40,
        showarrow=False,
        font=dict(size=12, color=T["text_muted"]),
    )
    return _base(fig, height)


def _group_label(name: str) -> str:
    """Wrap long axis labels so evidence charts remain readable on a phone."""
    return {
        "Function-word style": "Function-word<br>style",
        "Surface composition": "Surface<br>composition",
        "Structural organization": "Structural<br>organization",
        "Punctuation & rhetoric": "Punctuation<br>& rhetoric",
        "Information entropy": "Information<br>entropy",
        "Repetition & burstiness": "Repetition &<br>burstiness",
    }.get(name, name)


def generator_bars(generator: GeneratorAnalysis, height: int = 340) -> go.Figure:
    """Horizontal family scores preserve legible labels at mobile widths."""
    items = sorted(generator.probs.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v * 100 for _, v in items]
    colors = [
        (
            T["primary"]
            if k == generator.predicted
            else T["accent"]
            if k in generator.top3
            else T["muted_series"]
        )
        for k in labels
    ]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:.0f}%" if v >= 1 else "" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))
    fig.update_xaxes(
        title="Model score (%)",
        range=[0, max(values) * 1.3 + 1],
        gridcolor=T["track"],
        ticksuffix="%",
    )
    return _base(fig, height, margin=(92, 8, 30, 48))


def waterfall(
    balance: PathBalance, groups: list[GroupAttribution], height: int = 480
) -> go.Figure:
    """Horizontal additive logit accounting with readable, wrapped group labels."""
    ordered = sorted(groups, key=lambda g: -abs(g.phi))
    names = (
        ["Baseline forensics"]
        + [_group_label(g.display_name) for g in ordered]
        + ["Final logit"]
    )
    values = [balance.semantic_logit] + [g.phi for g in ordered] + [balance.final_logit]
    measure = ["absolute"] + ["relative"] * len(ordered) + ["total"]
    fig = go.Figure(
        go.Waterfall(
            y=names,
            x=values,
            orientation="h",
            measure=measure,
            increasing=dict(marker=dict(color=T["machine"])),
            decreasing=dict(marker=dict(color=T["human"])),
            totals=dict(marker=dict(color=T["primary"])),
            connector=dict(line=dict(color=T["muted_series"], width=1)),
            hovertemplate="%{y}<br>%{x:+.3f} logit<extra></extra>",
        )
    )
    positions = [0.0, balance.semantic_logit]
    for group in ordered:
        positions.append(positions[-1] + group.phi)
    lower, upper = min(positions), max(positions)
    padding = max(upper - lower, 0.1) * 0.15
    fig.update_xaxes(
        title="Machine logit",
        gridcolor=T["track"],
        zerolinecolor=T["muted_series"],
        range=[lower - padding, upper + padding],
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))
    return _base(fig, height, margin=(132, 12, 16, 48))


def _tint(hex_color: str, mix: float = 0.55) -> str:
    """Blend a color with white (used for feature nodes under a group)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * mix) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def sunburst(
    groups: list[GroupAttribution],
    features: list[FeatureAttribution],
    height: int = 480,
) -> go.Figure:
    """Groups to features, sized by absolute contribution, colored by group."""
    by_group = {g.name: g for g in groups}
    ids, labels, parents, values, colors, hover = (
        ["root"],
        ["Forensic evidence"],
        [""],
        [],
        [],
        [],
    )
    leaf_sum: dict[str, float] = {}
    leaves = [f for f in features if f.group in by_group and abs(f.phi) > 1e-12]
    for f in leaves:
        leaf_sum[f.group] = leaf_sum.get(f.group, 0.0) + abs(f.phi)
    total = sum(leaf_sum.values())
    if total == 0:
        return empty_figure("No forensic contribution to display.", height)
    values.append(total)
    colors.append("#FFFFFF")
    hover.append("All forensic features")
    for name, size in leaf_sum.items():
        g = by_group[name]
        ids.append(name), labels.append(g.display_name), parents.append("root")
        values.append(size), colors.append(g.color)
        hover.append(f"{g.display_name}<br>group contribution {g.phi:+.3f}")
    for f in leaves:
        arrow = "▲" if f.phi > 0 else "▼"
        # Only label features that are large enough to read; the rest stay hoverable.
        label = f"{f.name} {arrow}" if abs(f.phi) / total >= 0.03 else ""
        ids.append(f.name), labels.append(label), parents.append(f.group)
        values.append(abs(f.phi)), colors.append(_tint(by_group[f.group].color))
        hover.append(
            f"{f.name}<br>contribution {f.phi:+.3f} ({f.direction})<br>value {f.value:.4g}"
        )
    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1.5)),
            hovertext=hover,
            hoverinfo="text",
            insidetextorientation="radial",
        )
    )
    return _base(fig, height)


def attention_vs_causal(groups: list[GroupAttribution], height: int = 420) -> go.Figure:
    """Fusion routing weights versus absolute baseline-masking contributions."""
    if any(g.attention is None for g in groups):
        return empty_figure(
            "Fusion attention is available for attention-fusion models only.", height
        )
    total = sum(abs(g.phi) for g in groups) or 1.0
    ordered = sorted(groups, key=lambda g: abs(g.phi))
    names = [_group_label(g.display_name) for g in ordered]
    fig = go.Figure()
    fig.add_bar(
        y=names,
        x=[g.attention * 100 for g in ordered],
        orientation="h",
        name="Fusion attention",
        marker=dict(color=T["muted_series"], cornerradius=3),
        hovertemplate="%{y}<br>attention %{x:.1f}%<extra></extra>",
    )
    fig.add_bar(
        y=names,
        x=[abs(g.phi) / total * 100 for g in ordered],
        orientation="h",
        name="Share of contribution",
        marker=dict(color=T["primary"], cornerradius=3),
        hovertemplate="%{y}<br>contribution share %{x:.1f}%<extra></extra>",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.35,
        showlegend=True,
        legend=dict(orientation="h", y=1.25, x=0, yanchor="top", font=dict(size=10)),
    )
    fig.update_xaxes(title="Percent", gridcolor=T["track"], ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return _base(fig, height, margin=(132, 85, 16, 48)).update_layout(showlegend=True)


def group_contributions(groups: list[GroupAttribution], height: int = 340) -> go.Figure:
    """Diverging bars: positive pushes toward machine, negative toward human."""
    ordered = sorted(groups, key=lambda g: abs(g.phi))
    fig = go.Figure(
        go.Bar(
            y=[_group_label(g.display_name) for g in ordered],
            x=[g.phi for g in ordered],
            orientation="h",
            marker=dict(
                color=[T["machine"] if g.phi > 0 else T["human"] for g in ordered],
                cornerradius=3,
            ),
            hovertemplate="%{y}<br>%{x:+.3f} logit<extra></extra>",
        )
    )
    span = max(abs(g.phi) for g in groups) or 1.0
    fig.update_xaxes(
        title="Contribution to the machine logit",
        gridcolor=T["track"],
        zerolinecolor=T["muted_series"],
        zerolinewidth=1.5,
        range=[-span * 1.35, span * 1.35],
    )
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return _base(fig, height, margin=(132, 8, 16, 48))


def consistency_note(checks: ConsistencyChecks) -> str:
    """One-line summary of how well attention and attribution agree."""
    rho = checks.attention_vs_phi_spearman
    if rho is None:
        return "Attention comparison not available."
    return f"Rank agreement between attention and contribution (Spearman): {rho:+.2f}"
