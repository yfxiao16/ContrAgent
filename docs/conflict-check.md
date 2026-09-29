# Conflict check: optional tooling

`contragent conflicts --config library.yaml` treats the library as the
conjunction of its contracts, extracts a minimal unsatisfiable core, and
tests whether the assumptions of that core are jointly satisfiable. The
built-in search needs nothing beyond the runtime. Two optional tools
refine it.

## Z3

```bash
pip install -e ".[smt]"
```

Makes the numeric consistency filter exact. Without it a built-in
interval checker is used, which is sound but may report a numeric
conflict it cannot decide as unknown.

## mus2muc

Enumerates every minimal unsatisfiable core instead of the disjoint
cores the built-in search finds.

1. `pip install git+https://github.com/ainnoot/mus2muc`
2. Build its patched `wasp` solver and an LTL<sub>f</sub> solver
   (`aaltaf` or `black`) as described in the mus2muc README.
3. Point `CONTRAGENT_MUS2MUC_BIN` at the folder holding the binaries.

`contragent conflicts --backend mus2muc` then uses it; the default
`--backend auto` uses it whenever it is available.
