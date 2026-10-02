# TRAIN12 route-conditioning and observed-graph diagnostic

Status: implemented and preregistered; **no actual forward or graph-support run yet**. This is a bounded diagnostic, not a proposed paper contribution or a new task-quality score. Root freezes this source, runs real Torch tests, then separately authorizes the single audit. No DEV, locked payloads, new collection, training, or search.

## Evidence and question

At equal 12000 steps, the fully paired cosine control fits TRAIN with Tip717/756=94.84% and nearest-reference ADE.785cm, but its DEV Tip38/144=26.39%;67/144 DEV candidates have the correct endpoint yet collide. Classified-valid duplicates are zero there. This supports investigating conditional geometric generalization, not additional LR tuning or an assumed duplicate-collapse problem. Geometry jitter is narrow, but this does not prove appearance caused the failure; narrow passages can remain sensitive to small changes.

`routeset/observed_geometry.py` adds the direct Qwen feature branch to state and geometry at line192. The geometry branch itself is task-conditioned at lines124–139 and contains RGB features. Thus intervention on the direct branch cannot isolate all semantic influence, and a negative result cannot exonerate the indirect branch. Coordinate matching in `routeset/train_v2.py` supervises known positive trajectories separately; it does not literally average the reference set.

## Fixed scope and intervention

- Sole checkpoint: cosine **last12000**, SHA`cd1af0b572b80f9e7002ce866b7f3f062fed5afd6ab45ac16290db0c8b67c880`. No best, constant, or additional seed.
- Exact first four registered TRAIN parents283200–283203, all three target instructions:12 inputs. Missing inputs stop the audit; no replacement.
- Read original frozen Qwen caches and observed RGB-D/current/camera fields. The12 actual cache file hashes were mechanically read on the server before source freeze; no model or reference route was opened by that hash read. Their hashes, training config/summary, original saved TRAIN pool and previous planner fit are in the policy JSON.
- Run normal12, identity12, cyclic direct-feature swap12 in that order: at most36 complete K4 forwards/144 computed path states,0Qwen encodings,0optimizer updates. One requested swap per recipient, donor target`(t+1)%3` in the same image; no alternative donor selection.
- A scoped forward hook only replaces `head.feature_encoder(features)` output. The original recipient model inputs, geometry context/attention/anchor and current state stay unchanged and are checked exactly. Identity must reproduce every output bit. Hooks are removed even after exceptions; state_dict parameters and buffers must retain their byte hash.
- Hold each swapped candidate's final xyz to the corresponding normal endpoint; preserve the unclamped actual output separately. This is an explicit experimental intervention, **not a route repair or a normal quality claim**. Start, all22interior vertices and event output remain available for audit. Events are reported rather than silently corrected.
- Normal12 must reproduce the old189TRAIN saved-prediction pool at the same IDs, absolute tolerance1e-5, relative tolerance0, for XYZ and events. All per-input maxima are recorded. Reading this existing TRAIN prediction pool adds no forward.
- Report each candidate's full22-vertex displacement and all24vertex distances. Prefix uses the normal trajectory's cumulative arc length≤.5, excluding start/end; it is not the first12indices. Report every original-versus-swapped collision transition under the unchanged2cm tip-segment checker, including both damage and improvement. No candidate matching, rejection, or selection is performed.

All36pool files are hashed, all12observed graphs are built and sealed, then and only then the12selected label payloads are decoded. ID-first JSONL access never decodes other TRAIN/DEV label rows. Mechanical closure/hash metadata may be read across the registered export; no reserved raw content is decoded.

## Separate zero-search graph support audit

Load the **existing TRAIN32/31observed-parent** prototype/workspace fit from native100k TRAIN preflight, canonical identity`384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b`. This differs from the TRAIN64 neural fit; it is a separate representation diagnostic, not an information-matched neural/planner accuracy comparison. No refitting or reading the other31parents' routes.

On the same12inputs reuse original v2 observed grid, predicted target component,20mm clearance,2.5cm voxel, start/contact radii and exact original virtual-attachment checks. The three original planner source files retain their frozen accepted byte identities. A scoped guard prohibits calls to either A* function. True box/target coordinates, route types and routes do not enter graph construction.

After graph sealing, audit all corresponding known positive references unchanged, at both raw resolution and the original event-preserving H24 representation. Use original segment-supercover grid and finite-visible-point/fixed-ray checks; record attachment availability, unknown-ray counts, reference endpoint distance to the already-predicted endpoint, and raw/H24 recomputed passage types. No GT endpoint replaces a predicted endpoint. No path is trimmed, repaired, or filtered to improve its support count.

An admitted positive is a witness of support by these finite proxies. A rejected positive is **not proof that its type has no graph path**: another realization might fit, attachments/localization may be the limitation, and these proxies are conservative. Unknown types and absent references remain unknown; they are never negative supervision or a total solution count. Distinguish map rejection, unavailable attachments, localization failure and actual endpoint mismatch before interpreting any mode count.

## Falsifiable decisions; no retrospective effect cutoff

1. Identity/reproduction/source/state guards fail: stop as an engineering inconsistency; no research conclusion.
2. Direct swap yields exactly invariant route bodies: the direct bypass cannot explain the current saved model's route variation on these12inputs; do not remove it as the next claimed solution. This does not test indirect language-conditioned geometry.
3. Direct swap changes bodies, with or without collision changes: establishes only channel dependence under an off-manifold counterfactual. Report full distribution and signed changes, without a tuned “positive” threshold. Collision damage with fixed endpoints motivates a **conventional paired architecture control** sharing frozen grounding, observations, decoder budget and exposure; it does not itself prove DEV causality or novelty. Changes without damage weaken the claim that this sensitivity is the important failure mechanism.
4. Other known reference types have no admitted witnesses under current graph/proxy/attachments: do not launch a learned marginal field on the assumption it can recover them. First resolve whether endpoint attachment or conservative representation is excluding evidence. Rejection is not a mathematical impossibility result.
5. Several distinct known positive types have admitted witnesses yet fixed spatial planning misses them: this establishes an opportunity to consider learned **marginal** route costs. A learned static cost field plus the same Gaussian penalty would be the necessary strong control. Both must use the same grounding, graph, K4, native kernel/node/time budgets and all failures; ordinary field learning or sequential conditioning alone is not novelty. No such module is implemented by this protocol.

The first4parents are a low-cost engineering/feasibility sample, not sufficient to estimate a generalization effect or to claim that no method can work. There is no automatic larger audit, DEV test, retraining, or threshold sweep after this run. Root makes the next decision from all outcomes.

## Primary-work boundary

[Neural A* (ICML2021)](https://proceedings.mlr.press/v139/yonetani21a.html) already learns a guidance field with differentiable search; [official code](https://github.com/omron-sinicx/neural-astar) distinguishes current minimal code and the ICML2021 reproduction branch. [ModeSeq (CVPR2025)](https://arxiv.org/html/2411.11911v2) already generates modes conditioned on prior modes, without a dense hidden pool. [GoalFlow (CVPR2025)](https://arxiv.org/abs/2503.05689) and its [official code](https://github.com/YvanYin/GoalFlow) establish goal-conditioned trajectory generation as a standard nearby mechanism. [DSF/DPP](https://arxiv.org/abs/1907.04967) already targets quality/diversity of a learned candidate set. Neither grounding/route modularity, a learned field, sequential modes, nor DPP-style repulsion would alone constitute a new contribution. These works have been inspected, not reproduced here. Root maintains the broader related-work matrix.

## Execution, cost, and preservation

New files are isolated; no old model, planner, training source or existing result is changed. Launcher stages`tests`and`audit`are separate, fresh-only, record_job tracked, same immutable source. A second launch fails; failed/uncertain started calls and all unattempted slots remain in the ledger. CPU tests must actually include all three Torch cases with no skips before the audit.

Example after root supplies the frozen revision and CPU:

```bash
taskset -c "$CPU" /bin/bash "$SOURCE/scripts/launch_two_row_route_conditioning_v1.sh" "$REVISION" tests
taskset -c "$CPU" /bin/bash "$SOURCE/scripts/launch_two_row_route_conditioning_v1.sh" "$REVISION" audit
```

Audit uses one CPU, authorized GPU1/35% only for the small frozen head forwards, no new Qwen. Whole-process reserved GPU hours include startup and subsequent CPU geometry checking; this is disclosed rather than reported as pure forward time. Per-forward measured wall time is separate. The raw-reference finite-point checks may dominate the CPU time; runtime is not yet measured and there is no fixed-time advantage claim. All references are retained rather than truncated to meet a fabricated timing budget. `report.json`, status, forward intent/completion journal, source/input/pool/graph seals, original-pool reproduction receipt, and artifact index preserve the evidence. NPZ artifacts remain local/server ignored data.
