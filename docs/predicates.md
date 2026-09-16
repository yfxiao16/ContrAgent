# Interaction predicates

The fixed vocabulary that ALTL<sub>f</sub> formulas are written over. The
left column is the notation used in the paper, the middle column the
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
