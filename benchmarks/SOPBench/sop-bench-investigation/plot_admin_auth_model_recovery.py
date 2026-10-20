#!/usr/bin/env python3
"""Regenerate the clean model-recovery heatmap SVG."""
from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).with_name("admin-auth-order-model-comparison-heatmap.svg")
W = 1700
ROWS = [
    ("GPT-5.4 Nano", "GMI · v3 · 20+20", "10/20 · 50%", "red", "yellow", "Missed all 10 violations; inactive controls passed vacuously. Matched order pair failed."),
    ("Qwen3.5 27B", "GMI · v3 · 20+20", "10/20 · 50%", "red", "yellow", "Missed all 10 violations; inactive controls passed vacuously. Matched order pair failed."),
    ("Gemini 3.5 Flash-Lite", "GMI · v3 · 20+20", "11/20 · 55%", "yellow", "yellow", "Matched order pair, but missed 9/10 violations. Inactive controls passed; overall rule is too weak."),
    ("GPT-5.6 Terra", "GMI · v3 · 20+20", "20/20 · 100%", "green", "green", "Recovered active ordering; accepted all 4 inactive controls and all active controls."),
    ("GPT-6 Astra", "GMI · v3 · 20+20", "20/20 · 100%", "green", "green", "Recovered active ordering; accepted all 4 inactive controls and all active controls."),
    ("Grok 4.5", "GMI · v3 · 20+20", "20/20 · 100%", "green", "green", "Recovered active ordering; accepted all 4 inactive controls and all active controls."),
    ("Qwen3.5 397B A17B", "GMI · v3 · 20+20", "20/20 · 100%", "green", "red", "Recovered active ordering but ignored gate scope; rejected all 4 inactive controls."),
    ("Qwen3.8 Max", "GMI · v3 · 20+20", "20/20 · 100%", "green", "green", "Retry returned a parseable scoped rule; accepted all active and inactive controls."),
    ("Step-3.7 Flash", "GMI · v3 · 20+20", "Unparseable", "gray", "gray", "Returned text, but it did not contain a parseable formula."),
    ("GLM-5.3 Flash", "GMI · v3 · 20+20", "Unparseable", "gray", "gray", "Returned text, but it did not contain a parseable formula."),
    ("DeepSeek V4 Flash", "GMI · v3 · 20+20", "Unparseable", "gray", "gray", "Returned no parseable formula."),
    ("DeepSeek V4 Pro", "GMI · v3 · 20+20", "Unparseable", "gray", "gray", "Returned no parseable formula."),
]
COLORS = {
    "green": ("#dcefe3", "#176b3a", "RECOVERED"),
    "yellow": ("#fff0c9", "#815600", "UNCLEAR"),
    "red": ("#f8dfdc", "#9e2c2c", "NOT RECOVERED"),
    "gray": ("#e5e7e9", "#4b5560", "UNPARSEABLE"),
}

def text(x: float, y: float, value: str, size: int = 16, color: str = "#243746", weight: int = 400, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}px" fill="{color}" font-weight="{weight}">{escape(value)}</text>'

def wrap(value: str, limit: int) -> list[str]:
    result: list[str] = []
    current = ""
    for word in value.split():
        candidate = (current + " " + word).strip()
        if current and len(candidate) > limit:
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)
    return result

header_y = 42
row_height = 136
height = header_y + 72 + len(ROWS) * row_height + 18
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">',
    '<title>Admin-auth rule recovery by model</title>',
    '<desc>Comparison of held-out ordering recovery and gate-scope recovery.</desc>',
    f'<rect width="{W}" height="{height}" fill="#ffffff"/>',
]

# Table headings on white with crisp horizontal rules, no header panel.
cols = {"model": 68, "run": 350, "score": 550, "order": 750, "scope": 990, "notes": 1230}
svg.append('<line x1="50" y1="34" x2="1650" y2="34" stroke="#667580" stroke-width="1.5"/>')
for key, label in [("model", "Model"), ("run", "Training"), ("score", "Holdout"), ("order", "Order recovered?"), ("scope", "Gate scope?"), ("notes", "Notes / misses")]:
    svg.append(text(cols[key], header_y + 36, label, 24, "#263746", 700))
svg.append(f'<line x1="50" y1="{header_y + 72}" x2="1650" y2="{header_y + 72}" stroke="#667580" stroke-width="1.5"/>')

for i, row in enumerate(ROWS):
    model, run, score, order, scope, note = row
    y = header_y + 72 + i * row_height
    # Model labels are allowed to wrap instead of colliding with the next column.
    model_lines = wrap(model, 17)
    for n, line in enumerate(model_lines):
        svg.append(text(cols["model"], y + 42 + n * 33, line, 27, "#172b3a", 700))
    svg.append(text(cols["run"], y + 49, run, 24, "#526170", 500))
    svg.append(text(cols["score"], y + 49, score, 27, "#263746", 700))
    for key, status in [("order", order), ("scope", scope)]:
        fill, ink, label = COLORS[status]
        x = cols[key]
        svg.append(f'<rect x="{x}" y="{y + 12}" width="210" height="100" fill="{fill}"/>')
        svg.append(f'<text x="{x + 105}" y="{y + 62}" text-anchor="middle" dominant-baseline="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="25px" fill="{ink}" font-weight="700">{escape(label)}</text>')
    for j, line in enumerate(wrap(note, 35)[:3]):
        svg.append(text(cols["notes"], y + 37 + 32 * j, line, 25, "#344554", 400))
    svg.append(f'<line x1="50" y1="{y + row_height}" x2="1650" y2="{y + row_height}" stroke="#d2d9de" stroke-width="1"/>')

svg.append("</svg>")
OUT.write_text("\n".join(svg), encoding="utf-8")
print(OUT)
