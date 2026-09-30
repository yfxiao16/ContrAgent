"""ContrAgent: contract-based deterministic trajectory supervision of LLM agents.

An agent's behaviour is specified as assume-guarantee contracts whose
assumptions and guarantees are ALTLf formulas over interaction
predicates evaluated on the tool-call trace. Each contract compiles to a
DFA monitor; the same monitors gate tool calls online (``ContrAgent``)
and grade recorded traces offline (``evaluate_trace``, ``contragent eval``).
A contract's assumption is maintained from the environment side: an
environment event that would falsify it is suppressed before it reaches
the agent.
"""

from contragent.analysis import ConflictReport, check_conflicts
from contragent.config import ConfigError, ContrAgentConfig, load_config, load_system
from contragent.contract import ContractBuilder, contract
from contragent.core import CheckResult, ContractViolation, ContrAgent
from contragent.formulas.det import DetFormula
from contragent.formulas.parser import parse_formula, parse_repr
from contragent.models.agent import Agent
from contragent.models.contract import Contract
from contragent.models.system import System
from contragent.models.trace import Event, Trace
from contragent.runtime import Block, Escalate, Redirect, Supervisor, TraceVerifier

__version__ = "1.0.0"

__all__ = [
    "ContrAgent",
    "CheckResult",
    "ContractViolation",
    "Supervisor",
    "TraceVerifier",
    "Block",
    "Escalate",
    "Redirect",
    "Contract",
    "ContractBuilder",
    "contract",
    "DetFormula",
    "parse_formula",
    "parse_repr",
    "Agent",
    "System",
    "Event",
    "Trace",
    "load_config",
    "load_system",
    "ContrAgentConfig",
    "ConfigError",
    "check_conflicts",
    "ConflictReport",
    "__version__",
]
