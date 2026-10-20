#!/usr/bin/env python3
"""Render the controlled admin-auth ordering check as a standalone SVG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "controlled-admin-auth-order-results.json"
OUTPUT = HERE / "controlled-admin-auth-order-figure.svg"


def add_card(ax, x: float, y: float, width: float, height: float, title: str, detail: str, fill: str, edge: str) -> None:
    card = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.11",
        facecolor=fill, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(card)
    title_size = 8.3 if len(title) > 20 else 10.1
    ax.text(x + width / 2, y + height * 0.61, title, ha="center", va="center", fontsize=title_size, weight="bold", color="#24343b")
    ax.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center", fontsize=8.5, color="#51656e")


def add_arrow(ax, x1: float, x2: float, y: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#70838a"))


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_id = {item["trace_id"]: item for item in results["results"]}
    first = by_id["auth_before_transfer"]
    second = by_id["transfer_before_auth"]
    gate = first["gate_active"] and second["gate_active"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    ax.text(0.42, 5.05, "A controlled check of admin-auth ordering", fontsize=18, weight="bold", color="#1e3038", va="center")
    ax.text(0.42, 4.68, "Same successful events and task scope; only their order changes", fontsize=10.5, color="#52666e", va="center")

    gate_box = FancyBboxPatch((0.43, 4.05), 10.74, 0.4, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#edf3f2", edgecolor="#bdd0ca", linewidth=1)
    ax.add_patch(gate_box)
    ax.text(0.68, 4.25, f"Task context held fixed:  authenticated_admin_password gate = {'ACTIVE' if gate else 'INACTIVE'}", fontsize=10, weight="bold", color="#2e5e4a", va="center")

    row_y = [2.95, 1.65]
    labels = ["Conforming order", "Order violation"]
    trace_ids = ["auth_before_transfer", "transfer_before_auth"]
    event_pairs = [
        [("authenticate_admin_password", "succeeded = true"), ("transfer_funds", "succeeded = true")],
        [("transfer_funds", "succeeded = true"), ("authenticate_admin_password", "succeeded = true")],
    ]
    verdict_colors = {"PASS": ("#e8f4ec", "#3f8a5d", "#246541"), "FAIL": ("#fae9e7", "#cb6658", "#963c32")}
    for idx, y in enumerate(row_y):
        item = by_id[trace_ids[idx]]
        ax.text(0.48, y + 0.25, labels[idx], fontsize=10, weight="bold", color="#35474e", va="center")
        y_card = y - 0.14
        xs = [3.03, 5.39, 7.75]
        for card_idx, x in enumerate(xs[:2]):
            name, detail = event_pairs[idx][card_idx]
            add_card(ax, x, y_card, 1.88, 0.78, name, detail, "#f4f7f8", "#bdcbd0")
        add_card(ax, xs[2], y_card, 1.62, 0.78, "exit_conversation", "end of trace", "#f4f5f5", "#c7ced0")
        add_arrow(ax, 4.95, 5.28, y + 0.25)
        add_arrow(ax, 7.31, 7.64, y + 0.25)
        fill, edge, text_color = verdict_colors[item["contragent_verdict"]]
        verdict = FancyBboxPatch((9.75, y_card + 0.11), 1.0, 0.56, boxstyle="round,pad=0.03,rounding_size=0.14", facecolor=fill, edgecolor=edge, linewidth=1.7)
        ax.add_patch(verdict)
        ax.text(10.25, y + 0.25, item["contragent_verdict"], ha="center", va="center", fontsize=11, weight="bold", color=text_color)

    ax.text(0.45, 0.93, "Contract condition", fontsize=9, weight="bold", color="#35474e")
    ax.text(2.02, 0.93, "When the gate is active, a successful transfer requires prior successful admin authentication.", fontsize=9.0, color="#35474e")
    ax.text(0.45, 0.52, "Synthetic fixtures evaluated with ContrAgent’s shipped bank contract. This checks contract/trace semantics; it is not a miner result.", fontsize=8.6, color="#64777e")

    fig.savefig(OUTPUT, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
