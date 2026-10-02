# Research log

## Initial audit — 2026-10-02

Read the attached brief and README, experiment, next_steps, demo candidates, model/trainer/geometry and manifests. No existing AGENTS.md found in project/parent or server project. Local Git was an unborn master branch with all source files untracked; preserved original source/reports in d0d97eb on codex/multiroute-v2. Server project exists and has historical checkpoints, logs and data; it has no Git repository. No remote exists locally.

Hypothesis: fixed three-mode supervision hides ambiguity under variable reference counts and candidate budgets. First isolate this with two-wall opening-sequence data before complicating architecture. Literature and genuine observation pipeline prepare concurrently. Initial resource audit found no compute jobs; preserved historical resource limits since no higher authorization exists.

Decision: historical TEST/OOD are historical/development evidence going forward. V2 selection uses DEV_MODEL; new locked sets must stay uninspected during iteration.

## Round 1 — variable reference sets (completed)

Hypothesis: randomly selecting K references when R>K encourages an average path between incompatible passages. Compared unchanged SetRegressor with subset matching versus full positive rectangular assignment, same seed0 initialization, same scene sampling, 3000 updates, batch64, K4, 768000 gradient target slots. Both have access to the same reference pool; positive permits assignment across all R. This is a loss-objective control, not identical per-step target selection information. Code 478c4bb, 23 tests passed.

DEV_MODEL results (128 independent parents): subset UniqueValid1.77344 / Valid0.52539 / AnyValid0.75781; positive3.28906 /0.89648 /1.0. Paired UniqueValid difference1.515625, parent-bootstrap95%[1.22656,1.80469]. Durations127.52s and134.85s; controlled batch1 head median1.40ms, no VLM included. Prediction/checkpoint hashes in reports/v2_round1/artifact_index.json. Historical regressor reproduced TEST3/OOD2.90625 seed0; historical15 tests pass.

Decision: strengthen the baseline. All 53 remaining invalid positive-assignment outputs arise with R<K. When R>=4 the current probe is saturated. Do not frame known multi-modal regression loss correction as core novelty. Next actual experiment: saturation-aware positive assignment.

## Round 2 — fewer types than candidate budget (completed)

Hypothesis: random duplicate target multiplicities make excess deterministic slots regress between valid modes. Modification: minimum-cost positive assignment covering each known type once; excess slots choose any nearest known positive. Matched all other training settings. Exact assignment tested against exhaustive multisets; 24 tests pass. Code1a1b6bb; physically separated development/scoring/locked arrays (same training/development examples).

DEV_MODEL best step1500 of3000: UniqueValid3.3984375, Valid1.0, AnyValid1.0, ReferenceCoverage0.75933; this reaches mean min(K,R) for this enumerated controlled probe. Duration120.56s. No test evaluation. The last step is slightly below best (99.80% valid), retained in history. Next actual experiment: equal-parameter attention versus max coverage completion, with duplicate and verified invalid drafts, saturation-aware loss shared by both.

Recovery audit: 16 uninterrupted CPU steps versus8+resume8 produces maximum parameter difference0.0. Checkpoint preserves optimizer, scheduler, global RNG, sampler state and step. Config write moved after resume compatibility validation following independent review.

## Round 3 — given-draft completion (seed0 complete, seeds1/2 replication)

Hypothesis: max aggregation retains coverage evidence from distinct paths better than normalized averaging. Equal403522 parameters, shared saturation loss, same initialization/scene/context RNG and400000 gradient targets. Attention and coverage each2500steps. Mean added valid types across five completion contexts:1.4546875 versus1.61875. In two_valid,1.0→1.1875; overlap with existing types67→42 while new-path collisions53→48. See COMPLETION_REVIEW.md for parent-paired differences and CPU interventions. Precise duplicate and invalid-gating invariance holds for both models, so it does not explain the unique contribution. Seeds1/2 use the unchanged recipe and are actually training.

## Round 4 — self-produced drafts reveal transfer failure

CPU evaluation of frozen seed0 checkpoints, strictK4: generate2 then condition on those2 and generate2, no discarded route. Attention joint4 Unique3.2031 versus rollout2.5078; coverage joint4 3.3203 versus rollout2.9531. All initial invalid routes remain inK. Two passes take more time and lower quality here; retain single-pass strong regression for empty-set generation. Hypothesis for next training: reference-only context creates a draft distribution shift. Implemented late training mix of actual model drafts (both methods), with explicit extra training forward counts. Not yet claimed effective.

## Observation milestone

Official Qwen3-VL-2B files verified by hashes; true RGB+instruction forward completed onGPU1 with about3.99GiB peak. Same image/three color instructions yield different finite2048-dimensional hidden summaries. No LoRA or route training is inferred from this check.

Derived reach dataset:32 independent parent scenes/96 instructions,24TRAIN+8DEV_MODEL,288 collection attempts,275 successful paths and13 planning failures. All288 restored states/RGB images match exactly. Four original RLBench task feasibility checks:12/12 original success predicates, strict restored states. These are collected demonstrations, not learned robot results. Continuous full-body collision and discrete route types remain unverified.

The derived32 collector completed but its shell wrapper failed after a running wrapper was edited, leaving stale running status. Preserve wrapper exit1 and label collection_complete_wrapper_failed; do not mark wrapper successful. Subsequent jobs use frozen launchers and source hashes.
