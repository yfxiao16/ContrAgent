# Authoring a contract library

This page is for whoever turns a policy into contracts: a person, or a
coding agent pointed at this file. It covers what a contract can say,
how to say it so the supervisor reads it the way you meant, and how to
check the result before an agent runs under it. The vocabulary of
predicates is in [predicates.md](predicates.md); this page is about
putting them together.

## 1. What a contract is

A contract is a pair (A, G) over the agent's interaction trace:

- **G**, the guarantee, is what the agent must keep. It is checked on
  every tool call before the call runs; a call that would break it is
  refused (or redirected, or escalated), and never reaches the tool.
- **A**, the assumption, is what the environment must keep. It is
  optional. A tool result that would falsify it is suppressed: the
  result is withheld from the agent and the session state does not
  advance on it. While A holds, G is in force.

Both are ALTLf formulas: linear temporal logic on the finite trace of
the session, over interaction predicates such as `called('t')`,
`arg_field_has('t', 'field', 'regex')`, `output_has('t', 'regex')`, and
counters such as `Var('count', 't')`.

A library is a YAML file. Each contract has a `desc`, one or more `G`
formulas, and zero or more `A` formulas. Lists mean AND.

```yaml
version: "1"
agents:
  "*":                      # every agent, or a specific agent id
    contracts:
      - desc: "identity must be verified before funds move"
        G: {ltl: "(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))"}
      - desc: "file reads carry no credential"
        A: {ltl: "G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"}
        G: {ltl: "G((called('send_email') -> called('read_file')))"}
```

The `desc` is what a refused agent is told ("`transfer_funds` was
rejected by policy: identity must be verified before funds move"), so
write it as the rule a person would state, not as a label. It is
optional: a contract without one is reported under its formula read back
in words ("`verify_identity` must precede `transfer_funds`, or
`transfer_funds` must never be called"), which is accurate but says
nothing about why the rule exists.

Two more keys a library file accepts. `include: [contragent:sopbench/bank]`
pulls in a shipped library. A constraint written as `nl: "..."` instead
of `ltl:` is lifted to a formula at load time by the formulation
pipeline, which needs an `extractor:` section naming a model and the
`.[llm]` extra; the result is reported back in words for review, and
this guide's checks (section 6) apply to it as to any hand-written
formula.

## 2. Reading the policy

Go through the policy document sentence by sentence and sort each rule
into one of four shapes. Nearly every procedural rule is one of them.

| The rule says | Shape | Formula |
|---|---|---|
| "B only after A" (a precondition) | precedence | `(!(called('B')) U called('A')) \| G(!(called('B')))` |
| "never do X" / "never do X with such arguments" | prohibition | `G(!(called('X')))` / `G((called('X') -> !(arg_field_has('X', 'f', 'regex'))))` |
| "at most N times" | bound | `G((Var('count', 'X') <= N))` |
| "after A, must eventually do B" (an obligation) | obligation | `G((called('A') -> F(called('B'))))` |

Rules that do not fit are usually one of these in disguise. "Confirm
with the customer before cancelling" is a precedence on `confirm` and
`cancel`. "Refunds above $100 need approval" is a precedence whose
trigger carries an argument: `G((called('refund') &
Var('arg_numeric', 'refund', 'amount') > 100) -> ...)`, or a
prohibition on the unapproved case. "Do not leak the key" is a
prohibition on an output pattern, which belongs on the assumption side
(section 4).

Write one contract per rule. A contract that bundles several rules
gives one refusal message for all of them and one line in the conflict
report; the agent and the reviewer both lose the information of which
rule fired.

## 3. Weak and strong: the two mistakes that matter

**Precedence is a weak until.** "B only after A" must allow a session in
which B never happens. The formula `(!(called('B')) U called('A'))` on
its own is a *strong* until: it also demands that A eventually happens,
so a session that ends without any A is a violation at `finish_session`,
even if B was never attempted. The disjunction `| G(!(called('B')))`
makes it weak. Use the weak form for preconditions; use the strong form
only when you mean "A must happen".

**Obligations are decided at session end.** `F` and a strong `U` cannot
be refuted by a finite prefix: while the obligation is pending the
contract is undecided, and the supervisor does not refuse anything on
its account. It is reported by `finish_session()` (or at the end of a
replayed trace) if still owed. So an obligation never blocks a call; if
you need "B must be the very next call after A", write `X`:
`G((called('A') -> X(called('B'))))`. `X` does refuse: the call after A
that is not B is rejected.

## 4. Which side a predicate goes on

A guarantee constrains the agent, so it is written over what the agent
does: `called`, argument predicates, counters. An assumption constrains
the environment, so it is written over what the agent receives: tool
outputs (`output_has`), user input (`prompt_contains`), context
(`ctx`), time.

The supervisor keeps an assumption by *suppressing* the event that
would break it. It can withhold a tool result; it cannot withhold an
action the agent itself took. An assumption written over `called(...)`
therefore has nothing to suppress, and loading it raises a warning. To
scope a guarantee to something the agent does, put the trigger inside
the guarantee as `G(trigger -> ...)`.

```yaml
# wrong: the trigger is an agent action, so it cannot be assumed
A: {ltl: "G(called('transfer_funds'))"}
G: {ltl: "G(called('verify_identity'))"}

# right
G: {ltl: "G((called('transfer_funds') -> called('verify_identity')))"}
```

## 5. Arguments

Predicates over arguments make a rule precise ("no `rm -rf`", "no
refund above the limit", "only these recipients"):

```yaml
- desc: "no recursive deletion from the shell"
  G: {ltl: "G((called('bash') -> !(arg_field_has('bash', 'command', 'rm -rf'))))"}
- desc: "refunds stay under the limit"
  G: {ltl: "G((called('refund') -> Var('arg_numeric', 'refund', 'amount') <= 100))"}
```

Patterns are regular expressions matched against the field's string
value; anchor them (`'^13$'`) when you mean equality. Two consequences
of reading arguments:

- A call that arrives without the argument a contract reads is
  **refused**, not passed: the predicate has no value, and reading it as
  false would let `G(call -> !bad)` pass unchecked. Make sure the
  integration forwards the arguments. (`CONTRAGENT_ALLOW_MISSING_ARGS=1`
  restores the permissive behaviour if you accept that risk.)
- Tool names are compared in a canonical spelling, and an MCP wire name
  `mcp__server__tool` answers to `tool`, so write the bare name.

## 6. Check the library before an agent runs under it

Three checks, all deterministic, all without a model.

**Conflicts.** A library where no session can satisfy every contract at
once refuses everything or something arbitrary. The check treats the
library as one conjunction and extracts a minimal unsatisfiable core:

```bash
contragent conflicts --config policy.yaml
```

Fix the contracts the core names; the rest are not involved.

**Replay one trace per rule.** For each contract write the shortest
trace that satisfies it and the shortest that violates it, and replay
both. This is the test suite of the library, and it catches the weak /
strong mistakes of section 3 immediately:

```bash
contragent replay traces/refund_after_check.json    --config policy.yaml   # expect PASS
contragent replay traces/refund_without_check.json  --config policy.yaml   # expect FAIL
```

A trace is `{"events": [{"ts": 0, "agent": "agent", "type":
"tool_call", "tool": "check_policy", "args": {...}}, ...]}`; see
[../examples/traces/](../examples/traces/). `contragent eval DIR
--config policy.yaml` scores a whole directory of `safe_*.json` /
`unsafe_*.json` traces and reports precision and recall per contract.

**Dead ends.** Two shapes let a library reach a state that every
contract still permits but no continuation can complete. An obligation
whose obligated tool also carries a count bound: once the bound is
spent, the trigger leaves the obligation undischargeable. And two `X`
obligations sharing a trigger but naming different tools: `X` admits
one next event. `contragent.analysis.dead_ends.check_dead_ends` reports
both from the formulas alone; avoid them when authoring.

## 7. Reviewing what you wrote

Read each formula back in words. The library ships a back-translator
(`contragent.formulas.nl_gen.formula_to_nl`) that renders a formula as
a sentence; if the sentence is not the rule you started from, the
formula is wrong, whoever wrote it. The most common findings on review:

- a precedence written strong (section 3), which makes `finish_session`
  report sessions that never needed the action;
- a prohibition that should be conditioned on an argument but is not,
  which refuses every call of the tool;
- an assumption over `called(...)`, which suppresses nothing;
- several rules in one contract, which hides which rule fired.

## 8. Choosing the enforcement action

`Block` is the default and right for almost every rule: the call is
refused and the `desc` is returned to the agent as the tool result.
`Redirect("safe_tool")` substitutes a pre-approved call, for rules of the
form "instead of X do Y". `Escalate` holds the call for a human. Set
them per contract when constructing the supervisor:

```python
from contragent import ContrAgent, Redirect

guard = ContrAgent(agent_id="assistant", config="policy.yaml",
                   policy={"no recursive deletion from the shell": Redirect("safe_rm")})
```

`ContrAgent(mode="flag")` records every decision without gating the
agent: run a new library in this mode over real sessions first, read
what it would have refused, and only then switch to the default `gate`.

## 9. Writing contracts in Python

The fluent helper builds the same contracts in code, useful when the
library is generated or lives next to the tools:

```python
from contragent import ContrAgent, contract, parse_repr

library = [
    contract("identity must be verified before funds move").guarantees(
        parse_repr("(!(called('transfer_funds')) U called('verify_identity')) | G(!(called('transfer_funds')))")
    ),
    contract("file reads carry no credential")
        .assume(parse_repr("G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"))
        .guarantees(parse_repr("G((called('send_email') -> called('read_file')))")),
]
guard = ContrAgent(agent_id="assistant", contracts=library)
```

Both forms compile to the same monitors and are reported under the same
`desc`.

To put the contracts around your tools, wrap them: `@guard.wrap` on a
function, or `guard.wrap({"name": fn, ...})` on a tool table. Every call
then passes `guard_before` on the way in (a refused call returns the
refusal text without running, or raises `ContractViolation` with
`on_block="raise"`) and `guard_after` on the way out (a withheld result
is replaced the same way). A loop that executes tools elsewhere calls
the two hooks itself.

## 10. A worked example

Policy text, from a refund desk SOP:

> Verify the customer's identity before any refund. A refund needs a
> policy check on the order. Refund at most once per order. Every refund
> is followed by a receipt. Never refund more than $500.

Library:

```yaml
version: "1"
agents:
  "*":
    contracts:
      - desc: "identity must be verified before a refund"
        G: {ltl: "(!(called('issue_refund')) U called('verify_identity')) | G(!(called('issue_refund')))"}
      - desc: "a refund is issued only after the policy check"
        G: {ltl: "(!(called('issue_refund')) U called('check_policy')) | G(!(called('issue_refund')))"}
      - desc: "at most one refund per session"
        G: {ltl: "G((Var('count', 'issue_refund') <= 1))"}
      - desc: "every refund is followed by a receipt"
        G: {ltl: "G((called('issue_refund') -> F(called('send_receipt'))))"}
      - desc: "refunds stay under the limit"
        G: {ltl: "G((called('issue_refund') -> Var('arg_numeric', 'issue_refund', 'amount') <= 500))"}
```

Notes a reviewer would make: "at most once per order" was written as
"once per session", because the library has no notion of order identity
across calls; if the same session may refund two orders, drop the
bound or scope it by argument. The two preconditions are separate
contracts so a refusal names the missing step. The receipt is an
obligation: it never refuses a call, and a replay reports it against
the refund that incurred it. Then:

```bash
contragent conflicts --config refund.yaml
contragent replay traces/happy.json --config refund.yaml        # PASS
contragent replay traces/no_check.json --config refund.yaml     # FAIL at issue_refund
contragent replay traces/no_receipt.json --config refund.yaml   # FAIL: the receipt owed since issue_refund
```
