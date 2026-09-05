"""Command-line entry points.

``contragent eval TRACES [FORMULA...] --config LIB``
    Offline evaluation: replay recorded traces (``safe_*.json`` /
    ``unsafe_*.json``) against a contract library and report detection
    metrics per contract.
``contragent replay TRACE --config LIB``
    Replay one trace and print its end-of-trace verdict, first violation,
    and violating contracts.
``contragent conflicts --config LIB``
    Load-time conflict check of a library (minimal unsatisfiable core, then
    joint satisfiability of the core's assumptions).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from contragent import __version__


@click.group()
@click.version_option(version=__version__, prog_name="contragent")
def cli() -> None:
    """ContrAgent: contract-based supervision of LLM agent trajectories."""


def _collect_entries(config_path: str, agent_id: str | None) -> tuple[str, list]:
    from contragent.config import load_config

    cfg = load_config(config_path)
    if not agent_id:
        if len(cfg.agents) == 1:
            agent_id = next(iter(cfg.agents))
        else:
            raise click.ClickException(
                f"library has several agents ({list(cfg.agents)}); pass --agent"
            )
    if agent_id not in cfg.agents:
        raise click.ClickException(f"agent {agent_id!r} not in library {config_path}")
    entries: list = []
    for ce in cfg.agents[agent_id].contracts:
        for value in (ce.assumption, ce.guarantee):
            if value is None:
                continue
            entries.extend(value if isinstance(value, list) else [value])
    return agent_id, entries


@cli.command(name="eval")
@click.argument("trace_path", type=click.Path(exists=True, path_type=Path))
@click.argument("contracts", nargs=-1)
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), help="Library file.")
@click.option("--agent", "-a", "agent_id", help="Agent block to use (with --config).")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable report.")
def eval_cmd(trace_path: Path, contracts, config_path, agent_id, as_json) -> None:
    """Evaluate recorded traces against contracts and report detection metrics."""
    from contragent.eval_runner import discover_cases, format_report, run_eval

    if config_path and contracts:
        raise click.ClickException("cannot use both --config and inline formulas")
    if agent_id and not config_path:
        raise click.ClickException("--agent requires --config")
    if not config_path and not contracts:
        raise click.ClickException("give a --config library or at least one formula")
    if config_path:
        _, entries = _collect_entries(config_path, agent_id)
    else:
        entries = list(contracts)
    cases = discover_cases(trace_path)
    if not cases:
        click.echo(click.style(f"No trace files found at {trace_path}", fg="yellow"))
        sys.exit(0)
    report = run_eval(cases, entries)
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(format_report(report))


@cli.command(name="replay")
@click.argument("trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True))
@click.option("--agent", "-a", "agent_id", default=None)
@click.option("--json", "as_json", is_flag=True)
def replay_cmd(trace_path: Path, config_path, agent_id, as_json) -> None:
    """Replay one recorded trace and print the end-of-trace verdict."""
    from contragent.core import ContrAgent
    from contragent.models.trace import Trace

    data = json.loads(trace_path.read_text())
    trace = Trace.from_dict(data)
    guard = ContrAgent(agent_id=agent_id or "agent", config=config_path)
    result = guard.evaluate_trace(trace)
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    color = "red" if result["verdict"] == "FAIL" else "green"
    click.echo(click.style(f"verdict: {result['verdict']}", fg=color, bold=True))
    if result["first_violation"] is not None:
        ev = trace.events[result["first_violation"]]
        click.echo(
            f"first violation: event {result['first_violation']} ({ev.event_type} {ev.tool or ''})"
        )
    for v in result["violations"]:
        click.echo(f"  - {v['contract'] or v['guarantee']}")
    sys.exit(1 if result["verdict"] == "FAIL" else 0)


@cli.command(name="conflicts")
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True))
@click.option("--agent", "-a", "agent_id", default=None)
@click.option("--backend", default="auto", show_default=True, help="auto | builtin | mus2muc")
@click.option("--json", "as_json", is_flag=True)
def conflicts_cmd(config_path, agent_id, backend, as_json) -> None:
    """Check that a contract library is conflict-free."""
    from contragent.analysis import check_conflicts
    from contragent.config import load_system

    system = load_system(config_path)
    contracts = system.contracts
    if agent_id:
        contracts = [c for c in contracts if c.agent.id == agent_id]
    report = check_conflicts(contracts, backend=backend)
    if as_json:
        click.echo(json.dumps({"conflict_free": report.ok, "report": report.render()}, indent=2))
    else:
        click.echo(report.render())
    sys.exit(0 if report.ok else 1)


@cli.command(name="export-chase")
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Destination .logics file (default: print to stdout).",
)
@click.option("--agent", "-a", "agent_id", default=None)
@click.option(
    "--semantics",
    type=click.Choice(["finite", "infinite"]),
    default="finite",
    show_default=True,
    help="finite: LTLf-to-LTL translation with an 'alive' proposition.",
)
def export_chase_cmd(config_path, out_path, agent_id, semantics) -> None:
    """Export a library to CHASE's logics specification language."""
    from contragent.analysis.chase import contract_identifiers, export_library

    text = export_library(config_path, out_path, agent_id=agent_id, semantics=semantics)
    if out_path:
        n = len(contract_identifiers(text))
        click.echo(f"wrote {out_path}: {n} contracts ({semantics}-trace semantics)")
    else:
        click.echo(text, nl=False)


main = cli

if __name__ == "__main__":  # pragma: no cover
    main()
