#!/usr/bin/env python3
"""Render the healthcare model-recovery table as a status heatmap SVG."""
from __future__ import annotations
import html, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "healthcare-login-order-results.json"
OUTPUT = HERE / "healthcare-login-order-model-recovery.svg"
INK, MUTED, RULE = "#253746", "#526779", "#dce2e6"
GREEN_BG, GREEN_FG = "#dcefe4", "#176b3b"
AMBER_BG, AMBER_FG = "#fff0c5", "#805600"
GRAY_BG, GRAY_FG = "#e5e7e9", "#505b63"
RED_BG, RED_FG = "#f8dfdd", "#a3322d"
def text(x, y, value, size, color=INK, weight=400, anchor=None):
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x}" y="{y}"{anchor_attr} font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'
def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))["results"]
    by_name = {item["display_name"]: item for item in results}
    models = ["Gemini 3.8 Flash", "Qwen3.5 27B", "GPT-6-Astra"]
    headers = ["Model", "Training", "Holdout", "Order recovered?", "Gate scope?", "Notes / misses"]
    widths = [245, 220, 220, 250, 220, 545]
    x0, header_y, row_y, row_h = 50, 35, 100, 138
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="560" viewBox="0 0 1800 560" role="img" aria-labelledby="title desc">', '<title id="title">Healthcare SOPBench model recovery results</title>', '<desc id="desc">Three models mine login-before-update ordering. Two are partial; GPT-6 Astra recovers the rule on holdout. Gate scope is untested for all models.</desc>', '<rect width="1800" height="560" fill="#ffffff"/>', '<line x1="50" y1="34" x2="1750" y2="34" stroke="#9aa6ad" stroke-width="2"/>']
    x = x0
    for width, heading in zip(widths, headers):
        svg.append(text(x + 15, header_y + 36, heading, 23, weight=700)); x += width
    svg.append('<line x1="50" y1="96" x2="1750" y2="96" stroke="#9aa6ad" stroke-width="2"/>')
    for index, name in enumerate(models):
        item = by_name[name]; metric = item["evaluation"]["evaluated_sets"]["heldout_active_gate"]["metrics"]
        top = row_y + index * row_h; score = metric["correct"]
        holdout = f"{score}/20 · {round(metric['accuracy'] * 100)}%"
        if score == 20: order, order_bg, order_fg = "RECOVERED", GREEN_BG, GREEN_FG
        elif metric["true_violations"] == 10 and metric["missed_violations"] == 0: order, order_bg, order_fg = "PARTIAL", AMBER_BG, AMBER_FG
        else: order, order_bg, order_fg = "NOT RECOVERED", RED_BG, RED_FG
        gate_x = x0 + sum(widths[:4]); order_x = x0 + sum(widths[:3])
        svg.append(f'<rect x="{order_x+6}" y="{top+10}" width="{widths[3]-14}" height="{row_h-20}" fill="{order_bg}"/>')
        svg.append(f'<rect x="{gate_x+6}" y="{top+10}" width="{widths[4]-14}" height="{row_h-20}" fill="{GRAY_BG}"/>')
        svg.append(text(x0+15, top+54, name, 22, weight=700))
        svg.append(text(x0+widths[0]+15, top+54, "GMI · 20 + 20", 20, color=MUTED))
        svg.append(text(x0+widths[0]+widths[1]+15, top+54, holdout, 21, weight=700))
        svg.append(text(order_x+widths[3]//2, top+67, order, 20, color=order_fg, weight=700, anchor="middle"))
        svg.append(text(gate_x+widths[4]//2, top+67, "NOT TESTED", 20, color=GRAY_FG, weight=700, anchor="middle"))
        note_x = gate_x+widths[4]+15
        lines = (["Detected all 10 ordering violations;", "accepted no-success traces without", "false rejections on this holdout."] if name == "GPT-6-Astra" else ["Detected all 10 violations, but rejected 2", "conforming traces: required login even", "when no update succeeded."])
        for line_index, line in enumerate(lines): svg.append(text(note_x, top+37+line_index*27, line, 19, color=MUTED))
        svg.append(f'<line x1="50" y1="{top+row_h}" x2="1750" y2="{top+row_h}" stroke="{RULE}" stroke-width="1.5"/>')
    svg.append('</svg>'); OUTPUT.write_text("\n".join(svg)+"\n", encoding="utf-8"); print(OUTPUT)
if __name__ == "__main__": main()
