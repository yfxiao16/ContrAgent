# R-Judge: safety judgment of recorded records (offline)

## Claim

The same contract automata that gate an agent online grade a recorded
record offline, and on records whose risk is procedural (an action taken
without the required check, a side effect after an untrusted read) a
deterministic library matches LLM judges while producing the same verdict
every time.

## Setup

R-Judge is a detection benchmark: each record is an agent interaction
labelled safe or unsafe by human annotators, across 27 scenarios and five
application categories. Each record is converted to a trace with the real
tool names and arguments (`convert.py`) and replayed against the library;
a record is judged unsafe when any contract fires. Precision, recall, and
the false-positive rate on safe records are reported.

## Contracts

`contragent/contracts/benchmark/rjudge.yaml`: a small library whose central
rule is the indirect-injection guard (a high-risk side effect, such as
sending, transferring, granting access, or executing, requires a preceding
confirmation), together with sensitive-read gates and destructive-command
bans. Tool groupings are declared explicitly in the file.

## Results

Summary files: [`results/rjudge_eval.json`](results/rjudge_eval.json),
[`results/rjudge_by_category.txt`](results/rjudge_by_category.txt).

ContrAgent reaches an average F<sub>1</sub> of **91.8%** (97.0% precision,
87.0% recall) over multiple trials. The records it misses are semantic
rather than procedural: the harm lies in the content of an otherwise
permitted action, which a deterministic tool-call check does not decide.
Across the 23 base models profiled in the appendix, the false-positive rate
on safe traces stays below 1% for 19 of 23 models (maximum 4.5%,
Llama-3.1-8B) while the per-model violation rate varies widely with
capability.

## Reproduce

Clone <https://github.com/Lordog/R-Judge> next to the harness as described
in `benchmarks/R-Judge/contragent_eval/convert.py`, then

```bash
bash benchmarks/R-Judge/contragent_eval/run.sh
```

which converts the records, runs `contragent eval` with
`contracts/benchmark/rjudge.yaml`, and prints per-category recall and
false-positive rates.
