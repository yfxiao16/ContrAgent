# LLM-as-judge vs ContrAgent on SOPBench (paired, balanced pilot)

**Sample:** 40 unsafe + 40 safe per domain × 7 domains = **560 instances**, drawn
with a fixed seed (`SEED=1234`) from `traces/<domain>/`. The LLM judge and
ContrAgent are scored on the **identical** files (see `sample_manifest.json`).
Positive class = `unsafe` (the agent completed a goal the SOP forbids).

**Judge:** `gemini-2.5-flash` via REST, `temperature=0`, `maxOutputTokens=4000`
(thinking model). Each call is given the *same facts ContrAgent reads*: the
per-domain SOP (the human-readable rule set extracted from
`contragent/contracts/sopbench/<domain>.yaml`, which is authored faithfully from
`env/domains/<domain>/<domain>.py`), the world-state facts on the goal call
(`state_*`, `param_*`, `gate_*`, `prior_*`), and the ordered tool-call sequence.
Verdict parsed as VIOLATION→unsafe / COMPLIANT→safe; unparseable→safe (recorded).

**ContrAgent:** deterministic LTL contracts, **0 LLM calls**, microseconds/case.

---

## (a) Head-to-head: per-domain + aggregate (recall / FPR / precision / F1)

| Domain | Judge R | CA R | Judge FPR | CA FPR | Judge P | CA P | Judge F1 | CA F1 |
|---|---|---|---|---|---|---|---|---|
| bank          | 55.0 | **72.5** | 12.5 | **2.5** | 81.5 | **96.7** | 65.7 | **82.9** |
| dmv           | 37.5 | **50.0** | 5.0  | **2.5** | 88.2 | **95.2** | 52.6 | **65.6** |
| healthcare    | **67.5** | 60.0 | 20.0 | **2.5** | 77.1 | **96.0** | **72.0** | 73.8* |
| hotel         | 22.5 | **30.0** | 10.0 | **0.0** | 69.2 | **100.0** | 34.0 | **46.2** |
| library       | **50.0** | 27.5 | 12.5 | **0.0** | 80.0 | **100.0** | 61.5 | 43.1 |
| online_market | **42.5** | 20.0 | 15.0 | **2.5** | 73.9 | **88.9** | 54.0 | 32.7 |
| university    | 45.0 | **55.0** | 22.5 | **0.0** | 66.7 | **100.0** | 53.7 | **71.0** |
| **AGG**       | 45.7 | 45.0 | **13.9** vs | **1.4** | 76.6 | **96.9** | 57.3 | **61.5** |

(*healthcare F1 essentially tied: judge 72.0 vs CA 73.8.) **Bold = better.**

Aggregate recall is a statistical tie (judge 45.7 vs CA 45.0), but on the two
metrics that matter for an enforcement layer the deterministic checker dominates:

- **FPR: 13.9% (judge) vs 1.4% (ContrAgent)** — the judge over-blocks ~10× more.
- **Precision: 76.6% (judge) vs 96.9% (ContrAgent)** — ~1 in 4 of the judge's
  "VIOLATION" calls is a false alarm, vs ~1 in 32 for ContrAgent.
- **F1: 57.3 (judge) vs 61.5 (ContrAgent).**

## (b) Cost

| | LLM judge (gemini-2.5-flash) | ContrAgent |
|---|---|---|
| LLM calls | 560 (1 per instance) | **0** |
| Total tokens | **1,326,739** (≈531k in / 796k out+thinking) | 0 |
| Approx cost | **≈ $2.15** ($0.30/M in, $2.50/M out) | **$0.00** |
| Avg latency / case | **6.9 s** | **microseconds** (pure-Python LTL) |
| Wall clock (8-way concurrent) | ~12 min | < 1 s for all 560 |

Per-instance: judge ≈ **$0.0038 and ~7 s**; ContrAgent ≈ **$0 and ~0 s**.

## (c) Where the judge wins vs loses, and why

**Where the judge wins (recall):** `library`, `online_market`, `healthcare` —
domains where ContrAgent's hand-authored contracts are conservative (e.g. OR-
branch / guarded gates treated as inactive to protect FPR), so CA misses some
unsafe cases the judge's looser reasoning catches. The judge will flag a
plausible-looking precondition gap even when the deterministic rule was scoped
out.

**Where the judge loses:**
- **Over-flagging (FPR 13.9% vs 1.4%).** The judge invents or misapplies
  thresholds on *compliant* traces — it treats a borderline numeric value or a
  not-strictly-required gate as a violation. FP counts by domain:
  university 9, healthcare 8, online_market 6, bank/library 5. ContrAgent only
  fires when the exact grounded comparison fails, so it almost never over-blocks.
- **Truncation / non-determinism.** 33/560 calls hit `MAX_TOKENS`: the thinking
  model burned its entire output budget reasoning and never emitted a verdict.
  **24 of those were gold-unsafe**, so they became forced misses. A
  deterministic checker has no analogue of "ran out of thinking budget."
- **Precise numeric/temporal/counting constraints.** The judge's misses cluster
  exactly where the hypothesis predicted: multi-precondition goals (must compare
  several `state_*` vs `param_*` AND verify an ordering AND a login gate). It
  tends to check one or two preconditions and declare COMPLIANT, missing the one
  unmet numeric/ordering constraint — i.e. it does not reliably "count" all
  constraints. ContrAgent checks every conjunct mechanically.

## (d) Honest verdict

**Yes, with nuance.** On this balanced 560-instance pilot, ContrAgent matches the
LLM judge on recall (45.0 vs 45.7) while being **far more precise (96.9 vs 76.6),
far lower FPR (1.4% vs 13.9%), higher F1 (61.5 vs 57.3), and at literally zero
LLM cost and microsecond latency** versus ≈$2.15 and ~7 s/call. For an
*enforcement* layer (block before side effects), the judge's 13.9% FPR is
disqualifying — it would over-block ~1 in 7 legitimate actions. This supports the
claim **"ContrAgent ≥ LLM-judge on precise SOP constraints at ~0 cost"** on the
F1/precision/FPR/cost axes; recall is a tie, not a win.

Caveats kept honest: (1) the judge's recall is depressed ~4pp by 24 MAX_TOKENS
forced-misses — without truncation its recall would edge above CA's, but its FPR
and precision gap would remain. (2) ContrAgent's per-domain recall depends on
how aggressively its hand-authored contracts are scoped; the judge wins recall in
the 3 domains where CA is most conservative.

## (e) Sample size & next step

Pilot = **560 instances** (80/domain, balanced), single judge model
(gemini-2.5-flash), single run, temp 0. Because the head-to-head is favorable on
precision/FPR/F1/cost, the natural next step is a **larger sample and a stronger
judge (`gemini-2.5-pro`)** to test whether a more capable model closes the FPR
gap or improves recall enough to change the verdict — and to raise the judge's
output-token cap to eliminate the MAX_TOKENS truncations.

---

### Reproduce

```bash
set -a; . ../../../.env; set +a        # loads GEMINI_API_KEY (never echoed)
python3 llm_judge.py --per-domain 40   # -> judge_results.json, sample_manifest.json (cached)
python3 paired_contragent.py           # -> contragent_paired_results.json (same files)
```

Artifacts: `judge_results.json` (per-instance verdicts, metrics, token usage),
`judge_cache.json` (prompt-hash → response cache; re-runs do not re-bill),
`sample_manifest.json` (the exact paired file list),
`contragent_paired_results.json`.
