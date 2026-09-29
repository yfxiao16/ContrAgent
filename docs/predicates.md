# Interaction predicates

The fixed vocabulary that ALTL<sub>f</sub> formulas are written over. The
left column is the notation of the ContrAgent paper, the middle column the
spelling accepted by the parser.

| Paper | Formula spelling | Meaning |
|---|---|---|
| Call(T) | `called(T)`, `called_with(T, p)` | tool T is invoked (with arguments matching p) |
| ArgHas(T,f,p) | `arg_field_has(T, f, p)` | argument f of T matches pattern p |
| Path(T,P) | `arg_paths_within(T, P, ...)` | T's file paths lie within P |
| Subset(f,S) | `Subset(ArgValue(T, f), S)` | values in field f lie within set S |
| OutHas(T,p) | `output_has(T, p)` | result of T matches pattern p |
| Said(p), In(p) | `llm_said(p)`, `prompt_contains(p)` | model output / input matches p |
| Match(f,k) | `Eq(ArgValue(T, f), CtxValue(k))` | argument field f equals context value k |
| Ctx(k,v) | `ctx(k, v)` | context key k holds value v |
| Flow(s,d) | `flow(s, d)` | data from source s reaches sink d |
| Has(f) | `contains(f)` | a produced value contains field f |
| Perm(P) | `perm(P)` | caller holds permission P |
| Cnt(T) | `Var('count', T)` | number of T calls so far |
| Run(T) | `Var('consecutive_count', T)` | length of the current run of T |
| Num(T,f) | `Var('arg_numeric', T, f)` | numeric value of argument field f |
| Len(T,f) | `ArgLength(T, f)` | character length of argument field f |
| InLen, Chars | `Var('context_length')`, `Var('char_count')` | length of the model input / response |
| Tok | `Var('token_count')` | cumulative tokens consumed |
| Depth | `Var('delegation_depth')` | agent-delegation depth |
| Since(e) | `Var('time_since', e)` | time elapsed since predicate e held |

Numeric quantities are compared with `<=`, `<`, `>=`, `>`, and `==`.

## Operators

Temporal operators are `G` (always), `F` (eventually), `X` (next), and `U`
(until); connectives are `&`, `|`, `!`, and `->`. The prefix spelling
`G(Implies(called(a), F(called(b))))` is accepted as well.

## Which side a predicate belongs on

An assumption must be written over environment predicates only: tool
results, user input, context, and time. A condition on the agent's own
actions (`called`, `arg_field_has`, `count`, ...) belongs in the guarantee
instead, as the premise of an implication:

```yaml
# wrong: the trigger is an agent action, so it cannot be assumed
A: {ltl: "G(called('transfer_funds'))"}

# right
G: {ltl: "G((called('transfer_funds') -> called('verify_identity')))"}
```

The supervisor maintains an assumption by suppressing an environment event,
which it cannot do for an event the agent itself produced. Writing an agent
predicate in an assumption raises a `DeprecationWarning` at load time.

## Feeding the predicates

Data-flow and context predicates are supplied through the observer methods
on the supervisor: `observe_data_write`, `observe_data_read`,
`observe_delegation`, `observe_context`, and `observe_llm_call`. Predicates
over tool calls and results need no extra wiring; `guard_before` and
`guard_after` ground them.

## When a predicate has no value

ContrAgent defines an interaction predicate as a total function of the
session state, the event, and its parameter, with the values true and
false. The implementation keeps that reading. This section states what it
does with input from which no value can be read, since a contract that
reads such a predicate as false is satisfied without having checked
anything: every shipped guarantee has the shape `G(premise -> conclusion)`
or `(!after U before) | G(!after)`, and both read as satisfied when a
predicate in them never fires.

**Tool names.** `called(T)` and every other predicate keyed by a tool
compare names in a canonical spelling: surrounding whitespace is dropped
and case is folded. A call that arrives under the MCP wire name
`mcp__server__T` also answers to `T`. A contract written against `T`
therefore applies to `T`, `T `, `t`, and `mcp__server__T`. A contract
written against `mcp__server__T` applies to that server's tool only.

**Numeric arguments.** `Var('arg_numeric', T, f)` and an ordered comparison
on `ArgValue(T, f)` read a string argument as a number when it
unambiguously denotes one: `"5000"`, `"$5,000"`, `"5,000"`, `"5000 USD"`.
`"5,50"` is not read, since the comma may be a decimal separator. A value
beyond the float range is infinity, so a cap on it fires. Both paths use
the same reader, so they agree on every format.

**A pattern predicate over an absent field.** `arg_field_has(T, f, p)` is
false when the call has no field `f`, and `arg_length_exceeds(T, f, N)` is
false. These are values, not gaps: no field `f` matches nothing.

**Arguments the supervisor cannot value.** When some loaded contract
reads a tool's arguments and a call to that tool arrives with no
arguments at all, without a field a contract reads, or with a value a
numeric predicate cannot read as a number, the predicates over that event
have no value. The supervisor does not guess one. `guard_before` refuses
the call and tells the agent why, as it would refuse a violating call, so
the event never enters the trace. A tool no contract reads the arguments
of is unaffected. Setting `CONTRAGENT_ALLOW_MISSING_ARGS=1` restores the
earlier behaviour, under which the predicate read as false. Offline
replay does not refuse; it counts these events
(`contragent.tracer.grounding.grounding_misses`) and warns once per
predicate.

The formalism stays two-valued. A comparison whose operand has no value
evaluates to false in both evaluators, and that value alone is not
fail-closed for every formula shape: `G(c -> !(x > n))` is satisfied by
it while `G(c -> x <= n)` is violated. The refusal is therefore placed
before evaluation rather than inside it.
