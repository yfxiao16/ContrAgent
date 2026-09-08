"""Contract library files.

A library is a YAML file with one block per agent, each holding a list
of contracts. A contract has a guarantee (required) and an assumption
(optional), written either as an ALTL\\ :sub:`f` formula over the
interaction predicates or, when an extractor is configured, as a
natural-language requirement that the formulation pipeline lifts and
grounds to a formula::

    version: "1"
    agents:
      "*":
        include:
          - contragent:sopbench/bank        # another library file
        contracts:
          - desc: "identity must be verified before a transfer"
            A: {ltl: "F(called(transfer_funds))"}
            G: {ltl: "(!(called(transfer_funds)) U called(verify_identity)) | G(!(called(transfer_funds)))"}
          - desc: "at most three payments per session"
            G: {ltl: "G((Var('count', 'pay_bill') <= 3))"}

``A``/``G`` may be spelled ``assumption``/``guarantee``; a list value is
a conjunction. The formula syntax is the infix form printed by
``repr(formula)`` and parsed by :func:`contragent.formulas.parser.parse_repr`.
``${VAR}`` and ``${VAR:-default}`` in string values are interpolated
from the environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when a library file is malformed."""


@dataclass
class ToolEntry:
    """A tool the agent may call, given to the formulation pipeline as context."""

    name: str
    description: str = ""
    params: str = ""


@dataclass
class ConstraintEntry:
    """One formula (or natural-language requirement) inside a contract."""

    ltl: str | None = None
    nl: str | None = None
    source: str | None = None
    desc: str | None = None

    @property
    def is_ltl(self) -> bool:
        return self.ltl is not None


@dataclass
class ContractEntry:
    guarantee: ConstraintEntry | list[ConstraintEntry] = None  # type: ignore[assignment]
    assumption: ConstraintEntry | list[ConstraintEntry] | None = None
    desc: str | None = None
    activate_at: str | None = None
    assumption_mode: str | None = None
    library_source: str | None = None


@dataclass
class AgentConfig:
    agent_id: str
    contracts: list[ContractEntry] = field(default_factory=list)


@dataclass
class ExtractorSection:
    """Model used by the formulation pipeline for natural-language entries."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    @property
    def configured(self) -> bool:
        return any((self.provider, self.model, self.api_key))


@dataclass
class ContrAgentConfig:
    version: str = "1"
    defaults: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolEntry] = field(default_factory=list)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    extractor: ExtractorSection = field(default_factory=ExtractorSection)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    if isinstance(value, str):

        def _sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_VAR_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _parse_extractor_section(raw: Any) -> ExtractorSection:
    if raw is None:
        return ExtractorSection()
    if not isinstance(raw, dict):
        raise ConfigError(f"'extractor' must be a mapping, got {type(raw).__name__}")
    return ExtractorSection(
        provider=raw.get("provider"),
        model=raw.get("model"),
        api_key=raw.get("api_key") or None,
        base_url=raw.get("base_url") or None,
    )


def _parse_constraint_entry(item: Any) -> ConstraintEntry:
    if isinstance(item, str):
        return ConstraintEntry(nl=item)
    if not isinstance(item, dict):
        raise ConfigError(f"Constraint must be a string or mapping, got: {type(item).__name__}")
    if "ltl" in item:
        text = item["ltl"]
        if not isinstance(text, str) or not text.strip():
            raise ConfigError(f"Constraint 'ltl' must be a non-empty string, got: {text!r}")
        return ConstraintEntry(ltl=text, source=item.get("source"), desc=item.get("desc"))
    if "nl" in item:
        return ConstraintEntry(nl=item["nl"], source=item.get("source"), desc=item.get("desc"))
    raise ConfigError(
        f"Constraint mapping must have an 'ltl' or 'nl' key, got: {list(item.keys())}"
    )


def _parse_constraint_field(value: Any) -> ConstraintEntry | list[ConstraintEntry] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [_parse_constraint_entry(v) for v in value]
    return _parse_constraint_entry(value)


def _parse_contract_entry(item: Any, agent_id: str) -> ContractEntry:
    if not isinstance(item, dict):
        raise ConfigError(
            f"Agent '{agent_id}': each contract must be a mapping with "
            f"'A'/'assumption' (optional) and 'G'/'guarantee' (required); "
            f"got {type(item).__name__}"
        )
    if "A" in item and "assumption" in item:
        raise ConfigError(f"Agent '{agent_id}': contract has both 'A' and 'assumption': {item!r}")
    if "G" in item and "guarantee" in item:
        raise ConfigError(f"Agent '{agent_id}': contract has both 'G' and 'guarantee': {item!r}")
    g_raw = item.get("G", item.get("guarantee"))
    if g_raw is None:
        raise ConfigError(f"Agent '{agent_id}': contract is missing 'G' / 'guarantee': {item!r}")
    a_raw = item.get("A", item.get("assumption"))
    activate_at = item.get("activate_at")
    if activate_at is not None and activate_at != "first_match":
        raise ConfigError(
            f"Agent '{agent_id}': unknown activate_at value {activate_at!r}; "
            f"the only supported value is 'first_match'."
        )
    assumption_mode = item.get("assumption_mode")
    if assumption_mode is not None and assumption_mode not in ("monitored", "enforced"):
        raise ConfigError(
            f"Agent '{agent_id}': unknown assumption_mode value {assumption_mode!r}; "
            f"supported values are 'monitored' and 'enforced'."
        )
    if assumption_mode == "enforced" and a_raw is None:
        raise ConfigError(
            f"Agent '{agent_id}': assumption_mode='enforced' requires an "
            f"'A' / 'assumption' to enforce: {item!r}"
        )
    return ContractEntry(
        guarantee=_parse_constraint_field(g_raw),  # type: ignore[arg-type]
        assumption=_parse_constraint_field(a_raw),
        desc=item.get("desc"),
        activate_at=activate_at,
        assumption_mode=assumption_mode,
    )


# ----------------------------------------------------------------------
# Includes
# ----------------------------------------------------------------------

_BUNDLED_PREFIX = "contragent:"


def bundled_libraries_root() -> Path:
    import contragent

    return (Path(contragent.__file__).parent / "contracts").resolve()


def _resolve_include_spec(spec: str, base_dir: Path) -> Path:
    """Resolve ``contragent:<category>/<name>`` (shipped) or a file path."""
    if not isinstance(spec, str) or not spec.strip():
        raise ConfigError(f"include: entry must be a non-empty string, got {spec!r}")
    if spec.startswith(_BUNDLED_PREFIX):
        rel = spec[len(_BUNDLED_PREFIX) :].strip()
        if not rel:
            raise ConfigError(f"include: empty library name in {spec!r}")
        if not rel.endswith(".yaml"):
            rel += ".yaml"
        root = bundled_libraries_root()
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as e:
            raise ConfigError(f"include: {spec!r} resolves outside the shipped libraries") from e
        if not candidate.exists():
            available = sorted(
                f"{_BUNDLED_PREFIX}{p.relative_to(root).with_suffix('')}"
                for p in root.rglob("*.yaml")
            )
            raise ConfigError(f"include: library not found: {spec!r}. Available: {available}")
        return candidate
    raw = Path(spec).expanduser()
    p = raw.resolve() if raw.is_absolute() else (base_dir / raw).resolve()
    if not p.exists():
        raise ConfigError(f"include: file not found: {p}")
    return p


def _load_included_contracts(
    spec: str, base_dir: Path, agent_id: str, _seen: set[str]
) -> list[ContractEntry]:
    import yaml

    if spec in _seen:
        raise ConfigError(f"include: cycle detected: {' -> '.join([*_seen, spec])}")
    path = _resolve_include_spec(spec, base_dir)
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"include {spec!r}: invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"include {spec!r}: root must be a mapping")
    raw = _interpolate_env(raw)
    agents = raw.get("agents")
    if not isinstance(agents, dict) or list(agents.keys()) != ["*"]:
        raise ConfigError(
            f"include {spec!r}: an included library must define exactly one "
            f"agent named '*', got {list(agents.keys()) if isinstance(agents, dict) else agents!r}"
        )
    template = agents["*"]
    if not isinstance(template, dict):
        raise ConfigError(f"include {spec!r}: '*' must be a mapping")
    contracts_raw = template.get("contracts", [])
    if not isinstance(contracts_raw, list):
        raise ConfigError(f"include {spec!r}: 'contracts' must be a list")
    pulled: list[ContractEntry] = []
    nested = template.get("include", [])
    if nested:
        if not isinstance(nested, list):
            raise ConfigError(f"include {spec!r}: nested 'include' must be a list")
        _seen.add(spec)
        try:
            for n in nested:
                pulled.extend(_load_included_contracts(n, path.parent, agent_id, _seen))
        finally:
            _seen.discard(spec)
    for item in contracts_raw:
        ce = _parse_contract_entry(item, agent_id)
        ce.library_source = spec
        pulled.append(ce)
    return pulled


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_config(path: str | Path) -> ContrAgentConfig:
    """Parse a library file. Formulas are compiled later by :func:`config_to_system`."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Library file not found: {path}")
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("Library file must be a YAML mapping")
    raw = _interpolate_env(raw)

    config = ContrAgentConfig(
        version=str(raw.get("version", "1")),
        defaults=raw.get("defaults", {}) or {},
        extractor=_parse_extractor_section(raw.get("extractor")),
    )
    for t in raw.get("tools", []) or []:
        if isinstance(t, dict):
            config.tools.append(
                ToolEntry(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    params=t.get("params", ""),
                )
            )
        elif isinstance(t, str):
            config.tools.append(ToolEntry(name=t))

    agents_raw = raw.get("agents", {}) or {}
    if not isinstance(agents_raw, dict):
        raise ConfigError("'agents' must be a mapping of agent_id -> block")
    base_dir = path.parent.resolve()
    for agent_id, block in agents_raw.items():
        ac = AgentConfig(agent_id=agent_id)
        if isinstance(block, list):
            for item in block:
                ac.contracts.append(ContractEntry(guarantee=_parse_constraint_entry(item)))
        elif isinstance(block, dict):
            includes = block.get("include", []) or []
            if not isinstance(includes, list):
                raise ConfigError(f"Agent '{agent_id}': 'include' must be a list")
            for spec in includes:
                ac.contracts.extend(_load_included_contracts(spec, base_dir, agent_id, set()))
            contracts_raw = block.get("contracts", []) or []
            if not isinstance(contracts_raw, list):
                raise ConfigError(f"Agent '{agent_id}': 'contracts' must be a list")
            for item in contracts_raw:
                ac.contracts.append(_parse_contract_entry(item, agent_id))
        else:
            raise ConfigError(f"Agent '{agent_id}': value must be a mapping or list")
        config.agents[agent_id] = ac
    return config


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------


def _compile_ltl(entry: ConstraintEntry) -> Any:
    """Parse a formula string into a :class:`DetFormula`."""
    from contragent.formulas.det import DetFormula
    from contragent.formulas.parser import ParseError, parse_formula, parse_repr
    from contragent.formulas.regex_check import RegexValidationError, check_regexes

    # Two spellings are accepted: the infix form printed by ``repr`` and the
    # prefix form ``G(Implies(called(a), ...))``.
    try:
        formula = parse_repr(entry.ltl)
    except ParseError as infix_error:
        try:
            formula = parse_formula(entry.ltl)
        except ParseError:
            raise ConfigError(f"Failed to parse formula {entry.ltl!r}: {infix_error}") from infix_error
    try:
        check_regexes(formula)
    except RegexValidationError as e:
        raise ConfigError(f"Invalid regex in formula {entry.ltl!r}: {e}") from e
    return DetFormula(
        formula=formula,
        desc=entry.desc or entry.ltl,
        kind="ltl",
        liveness=has_pending_obligation(formula),
    )


def has_pending_obligation(formula: Any) -> bool:
    """True when the formula carries an unbounded eventuality.

    Such a formula (``F p``, ``G(a -> F b)``, a strong ``p U q``) cannot be
    refuted by a finite prefix: it is undecided while the obligation is
    pending and is settled only at session end. A weak until
    ``(!q U p) | G(!q)`` is a safety property and is not counted.
    """
    from contragent.formulas.formula import F, G, Not, Or, U

    def weak_until(node: Any) -> bool:
        # Or(U(Not(q), p), G(Not(q))) in either order
        if not isinstance(node, Or):
            return False
        a, b = node.left, node.right
        for u, g in ((a, b), (b, a)):
            if isinstance(u, U) and isinstance(g, G) and isinstance(u.left, Not) and isinstance(g.child, Not):
                if u.left == g.child:
                    return True
        return False

    def walk(node: Any) -> bool:
        if isinstance(node, F):
            return True
        if weak_until(node):
            u = node.left if isinstance(node.left, U) else node.right
            return walk(u.right)
        if isinstance(node, U):
            return True
        for attr in ("child", "left", "right"):
            sub = getattr(node, attr, None)
            if sub is not None and walk(sub):
                return True
        return False

    return walk(formula)


def _compile_nl(entry: ConstraintEntry, llm_extractor: Any, tool_inventory: list[dict] | None) -> Any:
    """Lift a natural-language requirement to a formula through the extractor."""
    if llm_extractor is None:
        raise ConfigError(
            f"Natural-language contract {entry.nl!r} needs an 'extractor:' section "
            "(or write the formula directly with 'ltl:')."
        )
    results = llm_extractor.extract_from_nl(entry.nl, tool_inventory=tool_inventory)
    ok = [r for r in results if r.ok]
    if not ok:
        errors = "; ".join(r.error for r in results if r.error) or "no constraint produced"
        raise ConfigError(f"Could not formulate {entry.nl!r}: {errors}")
    compiled = ok[0].compiled
    if entry.desc:
        from dataclasses import replace

        compiled = replace(compiled, desc=entry.desc)
    return compiled


def _compile_single(
    entry: ConstraintEntry,
    llm_extractor: Any = None,
    tool_inventory: list[dict] | None = None,
) -> Any:
    if entry.is_ltl:
        return _compile_ltl(entry)
    return _compile_nl(entry, llm_extractor, tool_inventory)


def _compile_field(
    value: ConstraintEntry | list[ConstraintEntry] | None,
    llm_extractor: Any = None,
    tool_inventory: list[dict] | None = None,
) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_compile_single(v, llm_extractor, tool_inventory) for v in value]
    return _compile_single(value, llm_extractor, tool_inventory)


def build_extractor(section: ExtractorSection) -> Any:
    """Construct the formulation pipeline's extractor, or ``None`` if unconfigured."""
    if not section.configured:
        return None
    from contragent.generation.llm_extraction import UnifiedExtractor

    return UnifiedExtractor(
        provider=section.provider,
        model=section.model,
        api_key=section.api_key,
        base_url=section.base_url,
    )


def config_to_system(
    config: ContrAgentConfig,
    llm_extractor: Any = None,
    tool_inventory: list[dict] | None = None,
) -> Any:
    """Compile every contract entry into a :class:`System` of :class:`Contract` objects."""
    from contragent.models.agent import Agent
    from contragent.models.contract import Contract
    from contragent.models.system import System

    if tool_inventory is None and config.tools:
        tool_inventory = [
            {"name": t.name, "description": t.description, "params": t.params}
            for t in config.tools
        ]
    if llm_extractor is None:
        llm_extractor = build_extractor(config.extractor)

    contracts: list[Contract] = []
    for agent_id, ac in config.agents.items():
        agent = Agent(id=agent_id)
        for ce in ac.contracts:
            g = _compile_field(ce.guarantee, llm_extractor, tool_inventory)
            a = _compile_field(ce.assumption, llm_extractor, tool_inventory)
            contracts.append(
                Contract(
                    agent=agent,
                    guarantee=g,
                    assumption=a,
                    desc=ce.desc,
                    activate_at=ce.activate_at,
                    assumption_mode=ce.assumption_mode or "monitored",
                )
            )
    system = System(name="config")
    system._contracts = contracts
    return system


def load_system(path: str | Path) -> Any:
    """``load_config`` followed by ``config_to_system``."""
    return config_to_system(load_config(path))


def config_to_guard_kwargs(config: ContrAgentConfig, agent_id: str) -> dict[str, Any]:
    """Keyword arguments for :class:`~contragent.core.ContrAgent` from one agent block.

    Returns ``{"agent_id": ..., "contracts": [...]}`` where each contract is a
    mapping with the compiled ``guarantee``, the compiled ``assumption`` (when
    present), ``desc``, and ``activate_at`` / ``assumption_mode`` (when set),
    ready to pass as ``ContrAgent(**kwargs)``. A ``"*"`` block applies to any
    agent.
    """
    block = config.agents.get(agent_id) or config.agents.get("*")
    if block is None:
        raise ConfigError(f"agent {agent_id!r} not found in library; have {list(config.agents)}")
    llm_extractor = build_extractor(config.extractor)
    tool_inventory = [
        {"name": t.name, "description": t.description, "params": t.params} for t in config.tools
    ] or None
    contracts: list[dict[str, Any]] = []
    for ce in block.contracts:
        entry: dict[str, Any] = {
            "guarantee": _compile_field(ce.guarantee, llm_extractor, tool_inventory),
            "desc": ce.desc,
        }
        assumption = _compile_field(ce.assumption, llm_extractor, tool_inventory)
        if assumption is not None:
            entry["assumption"] = assumption
        if ce.activate_at is not None:
            entry["activate_at"] = ce.activate_at
        if ce.assumption_mode is not None:
            entry["assumption_mode"] = ce.assumption_mode
        contracts.append(entry)
    return {"agent_id": agent_id, "contracts": contracts}


__all__ = [
    "ConfigError",
    "ToolEntry",
    "ConstraintEntry",
    "ContractEntry",
    "AgentConfig",
    "ExtractorSection",
    "ContrAgentConfig",
    "load_config",
    "load_system",
    "config_to_system",
    "build_extractor",
    "bundled_libraries_root",
]
