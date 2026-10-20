#!/usr/bin/env python3
"""Render the healthcare trace-mining setup and pilot results as a clean SVG."""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "healthcare-login-order-dataset.json"
RESULTS = HERE / "healthcare-login-order-results.json"
OUTPUT = HERE / "healthcare-login-order-experimental-setup.svg"


def text(x: int, y: int, value: str, size: int, *, weight: str = "400", color: str = "#17252c") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'


def rect(x: int, y: int, width: int, height: int, *, fill: str = "#f5f7f7", stroke: str = "#26343b", sw: int = 2) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def main() -> None:
    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))["results"]
    by_name = {item["display_name"]: item for item in results}
    counts = dataset["counts"]
    filtered = dataset["filtered_run_counts"]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="940" viewBox="0 0 1600 940" role="img" aria-labelledby="title desc">',
        '<title id="title">Healthcare SOPBench login-before-update trace mining setup and results</title>',
        '<desc id="desc">Four-step flow from recorded healthcare SOPBench traces through sanitization, LLM contract proposal, and evaluation, followed by three model holdout scores.</desc>',
        '<rect width="1600" height="940" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        text(52, 64, "Healthcare SOPBench: mining login-before-update order", 31, weight="700"),
        text(52, 103, "Recorded traces → sanitized examples → LTLf candidates → finite-trace evaluation", 20, color="#40515a"),
    ]

    # Main pipeline: straight arrows and square-corner boxes, matching the bank figure.
    cards = [
        (52, 145, 350, "1  Source traces", [
            f"{dataset['source_records_seen']:,} healthcare records scanned",
            f"{filtered['eligible_update_policy_runs']:,} eligible update_policy runs",
            f"{filtered['eligible_violates']:,} violate login-before-update",
        ]),
        (456, 145, 326, "2  Sanitize and split", [
            "Keep tool, success, and SOP gate",
            "20 + 20 training; 10 + 10 holdout",
            "Split by task configuration",
        ]),
        (836, 145, 326, "3  LLM proposes LTLf", [
            "Receives labeled training traces only",
            "Candidate formula is saved",
            "Holdout remains hidden",
        ]),
        (1216, 145, 332, "4  Evaluate candidate", [
            "ContrAgent finite-trace semantics",
            "Compare with ordering labels",
            "Also check 8 no-success controls",
        ]),
    ]
    for x, y, width, heading, lines in cards:
        svg.append(rect(x, y, width, 176))
        svg.append(text(x + 22, y + 40, heading, 21, weight="700"))
        for index, line in enumerate(lines):
            svg.append(text(x + 22, y + 84 + index * 31, line, 17, color="#40515a"))
    for x1, x2 in [(407, 447), (787, 827), (1167, 1207)]:
        svg.append(f'<line x1="{x1}" y1="233" x2="{x2}" y2="233" stroke="#26343b" stroke-width="2.5" marker-end="url(#arrow)"/>')

    svg += [
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M1,1 L11,6 L1,11" fill="none" stroke="#26343b" stroke-width="1.8"/></marker></defs>',
        text(52, 385, "Active-gate held-out results", 24, weight="700"),
        text(52, 418, "20 traces: 10 conforming · 10 violating", 18, color="#40515a"),
    ]

    # Results table.
    cols = [(52, 315, "Model"), (367, 175, "Score"), (542, 1006, "What the candidate captured / missed")]
    header_y, header_h, row_h = 446, 55, 61
    for x, width, heading in cols:
        svg.append(rect(x, header_y, width, header_h, fill="#e9edef", sw=1.5))
        svg.append(text(x + 15, header_y + 35, heading, 18, weight="700"))

    notes = {
        "Gemini 3.8 Flash": "Recovered ordering; falsely requires login when no update succeeds.",
        "Qwen3.5 27B": "Recovered successful-update order; same false rejection of no-update traces.",
        "GPT-6-Astra": "Recovered ordering and accepted traces with no successful update.",
    }
    for index, name in enumerate(["Gemini 3.8 Flash", "Qwen3.5 27B", "GPT-6-Astra"]):
        item = by_name[name]
        metric = item["evaluation"]["evaluated_sets"]["heldout_active_gate"]["metrics"]
        y = header_y + header_h + index * row_h
        values = [name, f"{metric['correct']} / {metric['n']}", notes[name]]
        for column, ((x, width, _), value) in enumerate(zip(cols, values)):
            svg.append(rect(x, y, width, row_h, fill="#eef5f1", stroke="#34765a", sw=1.4))
            svg.append(text(x + 15, y + 38, value, 17, weight="700" if column < 2 else "400"))

    svg += [
        '<line x1="52" y1="720" x2="1548" y2="720" stroke="#26343b" stroke-width="1.6"/>',
        text(52, 758, "Interpretation", 19, weight="700"),
        text(255, 758, "All eligible update_policy traces had the login gate active; inactive-gate scope was not tested.", 17, color="#40515a"),
        text(52, 803, "Labels measure this procedural ordering only—not overall task safety, permission, or completion.", 16, color="#40515a"),
        text(52, 838, "Successful logout traces were excluded to focus on ordering without session-reset semantics.", 16, color="#40515a"),
        text(52, 895, "The run split is by task configuration; prompts, arguments, database values, and benchmark safety labels were discarded.", 15, color="#40515a"),
        '</g></svg>',
    ]
    OUTPUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
