# Actual coverage results — active research snapshot

The strongest current development result is the ordinary realized-success head
on frozen C, not the proposed set mechanism. Formal five-seed results are pending;
do not read this snapshot as scientific acceptance or task completion.

## Frozen C, same1152TRAIN feedback,288reusedDEV

|System,seed0|U8|V8|U4|V4|
|---|---:|---:|---:|---:|
|Historical C adaptive|6.764|89.11%|3.743|94.27%|
|Success head,initial feedback2400|6.958|88.76%|3.774|94.36%|
|Scalar net head,initial feedback2400|6.750|87.80%|3.729|94.18%|
|Success head,shared two-context feedback2400|6.983|89.19%|3.774|94.36%|
|Dense peer-aware outcome net,shared feedback2400|6.885|89.37%|3.771|94.36%|

Main evaluation is exactly one eight-route trajectory decode, independently
checked, then the old full q returns four. Forward-call hooks verify the count.
Token-neighborhood work is extra inference cost, not free K=8computation.

## Recoverable omissions and net loss

FrozenC single-slot sampling: TRAIN1590omitted words can appear after replacement,
but only181have a positive net coverage change;180also preserve all quality counts.
DEV370can appear,77have positive net gain and no quality-count loss.
TRAIN5114andDEV1101replacements add some valid word without positive net coverage.
The sampled local oracle best improvement is0.181TRAIN/0.306DEV words per request;
these are diagnostics, not deployable model scores or global reachability bounds.
All seven companion(mode,variant)pairs are fixed, yet outputs can all change.

Original net predictor's supposedly safe gains are correct only16.0%onTRAIN and
9.86%onDEV. This motivated dense actual-word outcomes and the no-peer control;
the current dense result still does not beat the simple success method.

## Displacement-free continuation and refreshed feedback

All four geometry arms were trained to3600 with600/1800/3600curves, no displacement.
Adaptive outputs deteriorated in aggregate, including KL-only. At matched600steps,
matching-generator feedback and2400-step heads give:

|Geometry/head|U8|V8|U4|V4|
|---|---:|---:|---:|---:|
|Ordinary/success|6.837|87.46%|3.747|93.66%|
|Ordinary/dense net|6.813|87.59%|3.750|93.75%|
|Gap/success|6.965|88.76%|3.747|93.66%|
|Gap/dense net|6.983|88.76%|3.747|93.66%|
|Hard/success|6.823|87.20%|3.743|93.58%|

The gap/dense combination matches frozenC+success coverage but loses0.69percentage
point returned validity, beyond the registered0.5point margin. This geometry
screen matches data/budgets but branch-dependent RNG differs; it is exploratory,
not exact-stream causal replication. No weaker continuation replaces the strongest
simple baseline in the acceptance test. Full fixed-witness analysis is pending.

TEST_LOCKED remains untouched. All outcomes are end-effector route validity,
not whole-arm/closed-loop robot execution. See RESEARCH_STATE.md for active work.
