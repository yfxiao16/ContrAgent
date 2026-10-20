#!/usr/bin/env python3
"""Draw a flowchart separating bank-trace labels from synthetic ordering checks."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "admin-auth-order-summary.json").read_text())
RESULTS = json.loads((HERE / "controlled-admin-auth-order-results.json").read_text())
MINER_DATA = json.loads((HERE / "admin-auth-order-mining-dataset-v2-20-per-class.json").read_text())
OUT = HERE / "admin-auth-order-evidence.svg"
W, H = 1540, 1370
S: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 Z" fill="#7b8994"/></marker></defs>',
    '<title>Admin-auth ordering: bank traces and synthetic checks</title>',
    '<desc>Branching flowchart showing task permission, transfer calls, and authentication order in bank traces, followed by controlled synthetic examples.</desc>',
    '<rect width="100%" height="100%" fill="#ffffff"/>' ,
]


def txt(x: float, y: float, value: str, size: int = 16, color: str = "#243746", weight: int = 400, anchor: str = "start") -> None:
    S.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}px" fill="{color}" font-weight="{weight}">{escape(value)}</text>')


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str = "#cbd4db", radius: int = 0, sw: float = 1.5) -> None:
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def path(d: str, arrow: bool = True, color: str = "#7b8994") -> None:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    S.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{marker}/>')


def node(x: int, y: int, w: int, h: int, title: str, detail: str, fill: str, accent: str, title_size: int = 20) -> None:
    rect(x, y, w, h, fill)
    txt(x + 16, y + 31, title, title_size, accent, 700)
    txt(x + 16, y + 61, detail, 17, "#405260", 500)

# Header

txt(52, 57, "Admin-auth ordering: bank traces and synthetic checks", 34, "#142433", 700)
txt(52, 91, "Read the bank data from left to right. Permission labels and observed tool behavior are separate questions.", 19, "#526170")

# Main flow chart labels.
txt(58, 151, "BANK TRACES · 794 run records · 12 unique task scenarios", 21, "#142433", 700)
txt(58, 187, "TASK PERMISSION", 17, "#526170", 700)
txt(408, 187, "TRANSFER CALL", 17, "#526170", 700)
txt(900, 187, "AUTHENTICATION BEFORE FIRST TRANSFER", 17, "#526170", 700)

counts = SUMMARY["run_patterns_by_task_permission"]
all_counts = SUMMARY["run_patterns_all"]
p_total = sum(counts["Permitted"].values())
f_total = sum(counts["Forbidden"].values())
p_no_transfer = counts["Permitted"]["No transfer attempted"]
f_no_transfer = counts["Forbidden"]["No transfer attempted"]
p_attempt = p_total - p_no_transfer
f_attempt = f_total - f_no_transfer

# Draw connectors first, as two permission-specific trees. Branch labels identify
# the observed call/no-call split; the boxes carry the exact counts.
# Permitted branch.
path("M280 302 H338 V250 H405")
path("M338 302 V402 H405")
txt(346, 244, "No", 14, "#526170", 700)
txt(346, 421, "Yes", 14, "#526170", 700)
# Permitted transfer attempt to auth-pattern leaves.
path("M690 402 H790 V357 H900")
path("M790 402 V462 H900")
# Forbidden branch.
path("M280 659 H338 V590 H405")
path("M338 659 V744 H405")
txt(346, 584, "No", 14, "#526170", 700)
txt(346, 763, "Yes", 14, "#526170", 700)
# Forbidden transfer attempt to auth-pattern leaves.
path("M690 744 H790 V672 H900")
path("M790 744 V777 H900")
path("M790 777 V882 H900")

# Permission source nodes.
node(58, 250, 222, 104, f"Permitted · {p_total}", "4 task scenarios", "#eef7f1", "#176b3a")
node(58, 607, 222, 104, f"Forbidden · {f_total}", "8 task scenarios", "#fff5e9", "#8b5a16")

# Transfer-call split nodes.
node(405, 207, 285, 86, f"No transfer call · {p_no_transfer}", "52 permitted runs", "#f1f3f5", "#405260", 18)
node(405, 359, 285, 86, f"Transfer attempted · {p_attempt}", "188 permitted runs", "#edf2ff", "#294ca3", 18)
node(405, 547, 285, 86, f"No transfer call · {f_no_transfer}", "492 forbidden runs", "#f1f3f5", "#405260", 18)
node(405, 701, 285, 86, f"Transfer attempted · {f_attempt}", "62 forbidden runs", "#edf2ff", "#294ca3", 18)

# Authentication pattern leaves.
node(900, 315, 575, 84, f"Successful admin auth before transfer · {counts['Permitted']['Successful admin auth before transfer']}", "184 permitted runs", "#e5f4ea", "#176b3a", 19)
node(900, 420, 575, 84, f"No admin-auth call · {counts['Permitted']['No admin-auth call']}", "4 permitted runs", "#fff4d7", "#815600", 19)
node(900, 630, 575, 84, f"Successful admin auth before transfer · {counts['Forbidden']['Successful admin auth before transfer']}", "15 forbidden runs", "#e5f4ea", "#176b3a", 19)
node(900, 735, 575, 84, f"Failed admin auth before transfer · {counts['Forbidden']['Failed admin auth before transfer']}", "19 forbidden runs", "#fdebe8", "#9e3c32", 19)
node(900, 840, 575, 84, f"No admin-auth call · {counts['Forbidden']['No admin-auth call']}", "28 forbidden runs", "#fff4d7", "#815600", 19)

# Interpretive callout; sits beneath the tree and defines the disputed labels.
rect(58, 950, 1417, 82, "#f7f8fa", "#d7dfe5", 0)
txt(78, 981, "Key: ‘forbidden’ describes the task, not the trace.", 17, "#243746", 700)
txt(78, 1009, "‘No transfer call’ means no transfer_funds event, not no tool calls at all. Forbidden tasks can still show successful auth (15 do).", 15, "#405260")
txt(58, 1057, f"No auth-after-transfer counterexamples were observed ({all_counts['Admin auth only after transfer']} of {SUMMARY['scoped_run_records']}).", 16, "#9e3c32", 700)

# Synthetic section, visually separated from empirical tree.
S.append('<line x1="58" y1="1090" x2="1475" y2="1090" stroke="#cbd4db" stroke-width="1.5"/>')
txt(58, 1125, "SYNTHETIC CHECKS · constructed traces, not observed bank runs", 19, "#65449a", 700)
txt(58, 1153, "Same task and active gate; only event order changes.", 14, "#526170")

# Sequence rows from the actual fixture/result artifacts.
rows = RESULTS["results"]
sequence_x = [215, 500, 785]
sequence_w = [245, 245, 125]
sequence_labels = {"authenticate_admin_password": "Admin auth succeeded", "transfer_funds": "Transfer succeeded", "exit_conversation": "End"}
for idx, result in enumerate(rows):
    y = 1170 + idx * 73
    expected_ok = result["expected"] == "conforms"
    bg = "#e8f5ec" if expected_ok else "#fdebe8"
    accent = "#176b3a" if expected_ok else "#9e3c32"
    txt(58, y + 28, "A" if idx == 0 else "B", 18, "#243746", 700)
    for j, event in enumerate(result["event_order"]):
        x = sequence_x[j]
        rect(x, y, sequence_w[j], 42, bg, "#d7dfe5", 0)
        txt(x + sequence_w[j] / 2, y + 27, sequence_labels.get(event, event), 16, "#344554", 600, "middle")
        if j < len(sequence_x) - 1:
            path(f"M{x + sequence_w[j] + 6} {y + 21} H{sequence_x[j + 1] - 8}", color="#7b8994")
    txt(1012, y + 27, f"Expected: {result['expected']}", 14, "#526170", 600)
    txt(1240, y + 27, f"Contract: {result['contragent_verdict']}", 15, accent, 700)

# Miner task counts, explicitly a second synthetic dataset.
train = MINER_DATA["counts"]["training"]
holdout = MINER_DATA["counts"]["heldout_active_gate"]
controls = MINER_DATA["counts"]["heldout_controls"]
txt(58, 1335, f"Separate LLM-mining set: {train['conforms']} conforming + {train['violates']} violating train · {holdout['conforms']} + {holdout['violates']} active-gate holdout · {controls} gate-scope controls. Both v2 models: 20/20 active-gate; 0/4 inactive-gate controls.", 15, "#526170")
S.append("</svg>")
OUT.write_text("\n".join(S), encoding="utf-8")
print(OUT)
