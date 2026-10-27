"""Design tokens and CSS of the FLAIRD dashboard.

A restrained navy and teal palette, light evidence cards, and responsive spacing.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

TOKENS: dict[str, str] = {
    "primary": "#2457A7",
    "primary_dark": "#142D50",
    "accent": "#087F8C",
    "page_bg": "#F3F6FA",
    "card_bg": "#FFFFFF",
    "panel_bg": "#F1F5FA",
    "text": "#172B46",
    "text_muted": "#53657D",
    "muted_series": "#A9B5C1",
    "track": "#E6ECF3",
    "good": "#187449",
    "fair": "#E8B547",
    "poor": "#C62828",
    "machine": "#C54C38",
    "human": "#087F8C",
}

FONT_STACK = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

# Loaded in <head>: Inter from Google Fonts (falls back to the system font offline) and a
# script that pins the light color scheme so the dashboard looks the same everywhere.
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
  (function () {
    var url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
      url.searchParams.set('__theme', 'light');
      window.location.replace(url);
    }
  })();
  // Gradio can mount Plotly charts in hidden tabs. Resize when their real
  // container becomes visible or changes width (including mobile rotation).
  (() => {
    const observed = new WeakSet();
    const sizes = new ResizeObserver(entries => {
      for (const {target, contentRect} of entries) {
        if (contentRect.width > 0 && contentRect.height > 0 && target._fullLayout) {
          requestAnimationFrame(() => window.Plotly?.Plots.resize(target));
        }
      }
    });
    let pending = false;
    const attach = () => {
      pending = false;
      document.querySelectorAll('.fx-card .js-plotly-plot').forEach(plot => {
        if (!observed.has(plot)) {
          observed.add(plot);
          sizes.observe(plot);
        }
      });
    };
    const start = () => {
      new MutationObserver(() => {
        if (!pending) { pending = true; requestAnimationFrame(attach); }
      }).observe(document.body, {childList: true, subtree: true});
      attach();
    };
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', start, {once: true});
    } else { start(); }
  })();
</script>
"""


def build_css() -> str:
    """Return the dashboard CSS with the design tokens injected as CSS variables."""
    variables = "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in TOKENS.items())
    body = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    return f":root {{\n{variables}\n  --font: {FONT_STACK};\n}}\n{body}"


def build_theme() -> gr.themes.Base:
    """Gradio base theme aligned with the tokens (component-level details live in the CSS)."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#EAF1FB",
            c100="#D5E3F6",
            c200="#ABC7ED",
            c300="#7EA7DD",
            c400="#4F83C8",
            c500="#2F66AD",
            c600="#2457A7",
            c700="#1A4580",
            c800="#153866",
            c900="#102B4F",
            c950="#0B1E38",
        ),
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill=TOKENS["page_bg"],
        body_text_color=TOKENS["text"],
        body_text_color_subdued=TOKENS["text_muted"],
        block_info_text_color=TOKENS["text_muted"],
        block_label_text_color=TOKENS["text"],
        button_primary_background_fill=TOKENS["primary"],
        button_primary_background_fill_hover=TOKENS["primary_dark"],
        button_primary_text_color="#FFFFFF",
        block_radius="12px",
        input_radius="10px",
    )
