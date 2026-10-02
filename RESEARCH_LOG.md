# Research log

## 2026-10-02 14:17 UTC — completed SFT; autoregression launched; data scaling improves ordinary baseline

Hypothesis: real variable-K VLM serialization may preserve multiple route modes without a separate deterministic regression head. Completed ba984 real Qwen SFT1500 steps,3750 supervised route slots,1273081 answer tokens; DEV token NLL decreased .588300→.436727. Eight adapters updated,556.787s,4.834GB peak. This establishes actual training only. A failed pytest call in the Qwen runtime remains logged; the same fixed source passed15tests in the existing CPU runtime before training.

Actual next execution:8356b09 fixed-source7tests passed and same-best-checkpoint independent4/whole4 autoregression launched, all24DEV instructions, no retry/repair, raw text and all failed/extra slots charged. GPU1/35%,CPU1, source and launcher fixed. Quality will be independently checked only after the generation artifacts are complete.

In parallel, prospectively registered prefix60 contains48TRAIN+same12DEV parents,180/180 references. At identical1500x32 training exposure the ordinary free-endpoint head improved DEV macro ADE17.121→12.944cm and endpoint24.498→19.700cm; retrieval is15.393/25.895cm. Data enlargement helps but does not establish a mechanism. Complete source/init reconstruction and identical DEV audits before a paired causal description; task-validity and semantic correctness remain unmeasured.

Two-row v1 accepted18/27 but produced at most3 distinct lateral types per target. v2 explicit row-plane guides accepted4/27, with20 planning failures; strict within-run restoration passed, but cross-run arm joints differ by2.88969rad. The guide comparison is confounded. Actual next decision is a strict initial-state reconstruction/readback gate, not relaxed collision checks or another unpaired retry. Full negatives and cost retained.

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

## Round 3 replication — initial mechanism advantage is not stable

All three actual training seeds completed. Mean additional valid types across five completion contexts, coverage minus attention: +0.1640625, +0.015625, -0.046875. Mean difference0.044271, sample SD0.108347 across seeds. The fixed-three-model parent bootstrap is positive but does not establish training stability. In two_valid, seed2 has10 fewer overlaps but20 more collisions; R12 regresses in all seeds. Full failure lists and hashes retained in COMPLETION_THREE_SEED.md and its JSON.

Decision: do not promote max pooling to the core method. Before switching mechanism, run the already motivated self-draft distribution repair on both methods. Released3b9d3b7, 2500steps, equal scene exposure, 50% of eligible k2 batches afterstep1000 use model-produced2 drafts. Extra forward calls are counted. CPU uninterrupted8 vs4+resume4 parameters are exactly equal; all34 repository tests passed before launch.

## Observation route training — geometry localization is the bottleneck

True frozen Qwen cache plus ordinary set head trained1000steps,batch32,K4 on71 supervised instructions from24 parents, evaluated23 instructions from8 disjoint parents. DEV best ADE9.966cm, endpoint19.544cm, strict3cm semantic accuracy0; cached-head1.20ms is not end-to-end VLM latency. Training endpoint12.68cm atbest,10.75cm atlast: not just perfect-fit overfitting. DEV nearest-target identity81.5% is a diagnostic only; it does not replace strict goal accuracy. Same-image changed-language endpoint response28.5cm shows language signal.

Audit:96/96 labeled targets project into camera; deprojected observed sphere surfaces are~2.2cm from centers, consistent with sphere radius. No coordinate-system mismatch found. Next actual modification: learned RGB-D observed-point geometry and spatial anchor, shared by future baselines/mechanisms. This adds the originally intended observation input, not a new method claim.

## True online LoRA pilot — updates verified, no quality gain

Release6a4d719, frozen/LoRA each300steps,b1xacc4,1200 online RGB-language requests,4800 target slots, same head initialization/data order. Late2 q/v adapters:114688 parameters, all8 tensors have nonzero gradients and changed hashes. Frozen vsLoRA DEV ADE12.2591/12.2592cm, endpoint24.9827/24.9858cm, semantic0 forboth. Costs98.07/106.95s including setup, peak4.02GiB. Different measured request latency across sequential runs is not a claimed LoRA speedup.

Decision: short exposure substantially undertrains the new head compared with the128000-slot cached baseline. Prepare a matched warm-start frozen/LoRA comparison using the same already-trained head, not a backbone change alone. A subsequent genuine2-sample Qwen check finds online/cache features bit-identical (same84tokens,processor,pooling), ruling out cache mismatch. Preserve original failed pilot.

## Round 5 — model-draft repair completed; retain joint generation

Release3b9d3b7, equal400000 target slots and576 additional draft forward batches permethod,2500steps,seed0. Attention own2+2 UniqueValid2.50781→3.30469, Valid82.62%→97.07%; coverage2.95313→3.28125, Valid85.35%→96.29%. Parent-paired Unique improvement intervals[.64844,.94531] and[.21094,.44531], development descriptive only. This supports the draft-shift repair hypothesis but does not establish a novel mechanism.

Under the repaired recipe, joint4 reaches3.3125 attention/3.34375 coverage; two-pass2+2 reaches3.30469/3.28125 and costs about1.9x within each CPU1 evaluation. Thus retain joint4 as default and max pooling only as ablation. Original training usedCPU4, repairCPU2, so231/240s versus earlier205/193s cannot isolate computational overhead; record extra576 forward batches instead. Full before/after data and losses preserved in reports/v2_completion_selfdraft.

## Warm-start online control completed without gain

Both methods initialize the same ordinary frozen head beststep750 (96000 target slots atcheckpoint; full common pretraining run charged128000 slots). Each then500updates×4examples×K4=8000 additional slots. Both select commonstep0 as best on unchanged reference-ADE DEV selection. Last updated LoRA endpoint21.23cm/ADE10.69cm is worse than shared start19.54cm/9.97cm. All8 last adapter tensors actually changed; selectedbest adapters are zero-initialized. Do not describe step0 selection as successful LoRA quality improvement. Preserve best and last separately.

## Evaluation protocol v2 — retain tasks without successful demonstrations

Actual bug discovered: both cached/raw observation loaders omitted2 observations lacking successful references, including1 DEV instruction. Corrected loaders retain all96 observations; loss still samples the same71 TRAIN examples with positives. Semantic evaluation now includes all24 DEV instructions; reference ADE/events use23, and missing-reference rows containnull for those metrics. No changed3cm threshold, target identity rule or checkpoint selection. Original23-row outputs retained asv1. Frozen head re-evaluated onv2 retains the same reference metrics and strict semantic0; all online models are being re-evaluated with their true Qwen/adapter states before geometry training.

## Round 6 — diffuse spatial attention diagnosed and repaired

All four online v2 re-evaluations completed with actual Qwen/adapter states; original23 reference rows unchanged. RGB-D ordinary set head, seed0, initially improves ADE9.97→8.96cm but strict semantics remains0. Six predeclared DEV visualizations show learned point attention spread across targets/background. No camera sign/coordinate bug found.

Hypothesis: path regression alone supplies too weak a localization signal. Modification: training-only Gaussian-mixture attention supervision around each positive recorded endpoint, weight0.02/sigma2.5cm, otherwise identical architecture/data1000steps/batch32/K4/128000slots. All42 tests passed; real-data CPU4step vs2+resume2 bit-exact. Three seeds completed: mean strict semantics1.0417%→28.8194%, AnySemantic1.3889%→31.9444%, endpoint18.9179→14.5091cm, ADE9.5654→7.7252cm. Per-seed strict gains33.333/28.125/21.875 percentage points; no negative seed suppressed. This is a conventional strong-baseline repair on8 development parents, not core innovation. Full predicted paths/checkpoint hashes/parent rows retained.

Decision and actual continuation: larger independent natural-layout256 parents and randomized-obstacle128 parents are collecting onCPU1 each. Snapshot only completed new TRAIN parents for interim learning curves against explicitly reused8DEV; preserve formal splits. Full-path geometry/route types remain separate from endpoint semantics.

## Collection audit — stale rendered meshes, then verified restoration

Randomized obstacle pilot had identical native states but stale first-frame RGB-D. Diagnostics identified depth mismatch to actual box despite unchanged cameras/object poses. First attempted double restore without an explicit intermediate render did not fix all parents; keep failure. Fixed deterministic sequence restore/render-discard/restore/render at reference and every candidate. Four parents48 attempts then have exact state/inventory/RGB restoration;20 successful paths,17 classifiable routes,21 planning failures+7 actual collisions. Strengthened auditor also rejects stale smaller meshes lying inside true boxes: all4 actual parent point clouds lie within2.4mm of physical surfaces. Formal expansion uses88887f2 and fresh seeds272000+, no reuse of pilot as TRAIN.

Natural-layout fast collector pilot seed261900:9/9 paths, exact restore and official/direct state equivalence,27.99s. Formal seed262000–262255 job uses frozenbb5289;192TRAIN64DEV,2304 planned proposals. These are proposed denominators until observed completion, not fabricated successes.

## Round 7 — new training layouts preserve the localization repair direction

An immutable32-new-TRAIN/8-reused-DEV snapshot contains120 observations,96 supervised TRAIN instructions and24semantic/23reference DEV instructions. True Qwen BF16 encoding completed120 requests in9.46s. First command accidentally used the defaultCPU/FP32; verified own childPID was interrupted and original partial cache retained. Actual runs use a separateBF16 cache, never mix precision/features. Two same-content snapshot aliases do not count as extra data.

Six actual1000-step runs, releasef8ce88a,128000slots each: plain meansemantic0, aux28.8194%; mean endpoint18.5449→12.1939cm; ADE8.7100→6.3855cm. Allthree seed differences positive for semantics and negative for errors. New TRAIN pool supports the repair across training layouts; reused8DEV is still exploratory, not an independent confirmation setting. Reports retain each seed, per-scene failures, snapshot sources and artifact hashes.

Actual24 serial observation-to-route requests with liveQwen, RGB-D read/backprojection and learnedhead: median54.99ms, after first2 median54.96/p9556.38ms; model load7.81s, firstrequest760.76ms, peak4.005GiB. Predictions agree with cached training evaluation within3.58e-7m, all96 semantic decisions identical. Timing excludes collision checking, scoring and execution, explicitly recorded; no cached-throughput claim.

## Round 8 — physical-obstacle learning starts; geometry still fails

New16 TRAIN parents (90 accepted paths,1/48 no-reference instruction) plus4 reusedpilotDEV parents (20 references) form an immutable versioned snapshot. ActualQwen cache and plain/aux seed0 training1000steps completed underbb3077. Early DEV box-only TipValid2.083%→4.167%, AnyTipValid both8.333%; semantic2.083%→8.333%, TipClear62.5%→50%, endpointcenter15.10→16.37cm. Thus localization auxiliary is not a universal route-quality improvement. Per-candidate failure attribution and complete predeclared4parent visualization are in progress; do not claim full-arm/task execution success.

Parallel mechanism hypothesis: learned local edits may preserve useful portions after one channel closure. Three arms full_free/full_paired/local_paired share initialization/data/supervision and192000slots; free matching is the primary strong baseline.381TRAIN/57DEV parents, allvariants inheritparent split. All48 repository tests passed; teacher-copy TRAIN7092routes remain100%valid. The historical frozen head sees non-prefix gap masks it was not trained on; its poor zero-adaptation output is diagnostic only. Actual three-arm GPU job started frombb3077, no locked data used.
# 2026-10-02 — continued evidence-driven checks

- Hypothesis: the local copy gate may converge slower than full free completion. V1 at2000 has negative local-minus-free Unique b1/b2=-0.07588/-0.16197. No evidence supports fragmented edit boundaries as the failure cause. Implemented audited continuation into fresh outputs; 2→4 equals uninterrupted4 exactly and original artifacts remain unchanged. Actual three-arm4000 continuation launched from093a1b4 after source hash/config/resource checks; no extra gate module added.
- New64 natural TRAIN seed0 heads completed at equal128000 slots. Plain→aux reference endpoint15.70→10.50cm and ADE8.12→5.74cm, strict semantics0→19.79%. DEV is the reused8 parents, not a new test set; strict accuracy does not improve monotonically with pool size under fixed exposure. The selected auxiliary head is the common initialization for a forthcoming real online frozen/LoRA RGB-D pair.
- A TRAIN-only exact-instruction color prototype achieves strict endpoint19/24 and20/24. Background outliers remain, including multi-meter errors; full-path metrics remain null. This stronger task-specific endpoint control prevents overclaiming the neural grounding repair.
- Obstacle prediction analysis preserves all12 DEV tasks and original ADE selection. Fixed first6 eligible TRAIN gradient decomposition at best250/last1000 reconstructs loss within3e-9; interior versus endpoint+grounding cosine is positive. The proposed gradient-conflict explanation is unsupported on this batch; no detach fix implemented. More formal obstacle parents are being collected with all failures retained.
- 2026-10-02 further decision: matched constraint continuation completed; local-minus-full_free Unique=-0.06537/-0.11800. Discard local gate, retain full_free and all artifacts; no more training on that mechanism. Latest budget-neighbor check found ModeSeq already studies preceding-mode memory and dynamic candidate count; ordinary prefix/K conditioning is not a sufficient novelty claim.
- New real online RGB-D frozen/LoRA1000×4 pair finished at common step0 best; last quality degrades in both, despite real adapter updates. Keep the common pretrained head. A fixed-checkpoint all24DEV attention diagnosis shows peak strict87.5% versus soft anchor16.67%; implement only a same-parameter straight-through observed-point anchor control, with old soft behavior preserved. This is baseline repair, not the proposed paper core.
- Obstacle32 prototype endpoint succeeds12/12 on the reused4-parent DEV, with all original hyperparameters. Actual new32 Qwen cache→plain/aux paired training launched from9c19288 after online queue exit0. A traditional observation-only planning baseline is being prepared; no geometry or target labels may enter its planner.

## 2026-10-02 — paired anchor repair and stronger controls

- Hypothesis: multimodal soft coordinate averaging is a concrete localization bottleneck. Same-parameter hard observed-point forward with soft backward gradients, same data/initialization/128000 slots, completed three seeds in two settings from2dc026b. Natural64 original ADE-selected strict semantics17.36±13.19%→83.68±6.62%, referenceADE6.47→4.55cm, all seeds positive. Retain this conventional grounding baseline repair; do not claim route-set novelty.
- Obstacle32 original ADE-selected semantics19.44±4.81%→28.47±34.38%, referenceADE18.38→19.30cm. Seed0 box-tip valid4/48→8/48, but other seeds and path coverage need uniform checks. Decision: preserve original best and separately evaluate BOTH arms/ALL seeds at fixed1000 to diagnose selection sensitivity; no retroactive favorable replacement. This actual12-source CPU evaluation from94de7db finishedexit0; obstacle fixed-step semantics39.58→72.22% with ADE19.99→20.66cm. Full-path checks continue.
- Same-pretrained-head real online frozen/LoRA RGBD1000×4 pair completed: both beststep0, last degrades, eight LoRA tensors truly updated. Restore common pretraining head rather than spending more seeds on unsupported LoRA gains. Negative evidence, initial hash failure and all costs preserved.
- Traditional observation-only A* v1 emitted48 failed DEV slots; all count in budget. A first-TRAIN-only frozen diagnosis was prepared before repair, using no DEV targets or full geometry. This failure is not a credible strong planner comparison yet.
- Multigate diffusion controls passed5 targeted CPU tests including exact resume and actual paired stream hash. Initial GPU launcher failed before training due CRLF-vs-LF source hashes; actual deployed content was canonical-byte-equivalent after newline normalization. A second invocation passed source checks but failed its relative launcher self-hash after changing directory. Preserve both startup failures; invoke the unchanged version2 launcher by absolute path. Training only counts when real run records/logs confirm it.
- Natural collector is nearing256 requested parents. Prospective exact role reservation registered10:10:31UTC is authoritative; no locked-parent images/routes/model metrics have been inspected. A new role-specific export will apply this mapping while preserving original raw manifests and every failed parent.

## 2026-10-02 — new development layouts and numerical diffusion repair

- Natural256 collector finishedexit0:2304 attempts,2281 accepted references,23 failures,1 near-duplicate, all strict restore checks passed. New reservation-safe exporter passed Linux tests including cross-parent symlink rejection and forbidden payload nondecoding. Exact192TRAIN+16DEV export has624 input rows; fresh real frozen Qwen encoding and all624 cache/hash checks completedexit0. No reserved score/calibration/locked content was opened for method selection.
- Transferring the SIX existing64-parent ADE-selected heads unchanged to fresh16DEV gives strict semantics soft[31.25,29.17,25.00]% versus peak[72.40,75.00,68.75]%; all three seeds improve and ADE modestly decreases. This is new-layout DEV support for conventional grounding repair, not final TEST or core set-generation novelty. Next actual experiment: fresh paired new192 heads,3000×32×4 each;3× data/steps approximately maintains training exposure per input, so comparisons to old64 are not equal-budget claims.
- Both original3000-step epsilon diffusion arms completed and actual target/noise streams exactly matched. Fixed TRAIN32/DEV128 diagnostics show terminalalpha2.43e-7 amplifies epsilon errors, rawx0 RMSE159–198 with~99% clipping. Low-noise correct-condition error0.009–0.011m and shuffled-condition0.021–0.024m show learned conditioning. Decision: standard v prediction with direct stable reconstruction, same schedule/data/streams/K4/40steps and768000 slots, fresh two-arm training. Nine targeted tests passed including exact resume/default behavior and rejection of unsupported v+PG; actual GPU paired job launched from7f27f93. No new mechanism claim.
- Observation A* v2 keeps original20mm clearance, contact radii, target component and graph budgets. It adds exact virtual endpoint connections checked against the finite observed point cloud and2.5mm ray samples; scope remains an observation proxy. First TRAIN probe found4/4 routes but audit JSON failed on NumPy integer indices. Originalexit1/raw routes retained; only record serialization fixed in a32e91e, next fresh-output TRAIN probe precedes DEV. No threshold was relaxed using DEV.
- Four additional task feasibility trials launched from3b597f4 with strict RGB/state/language restore and discarded warm render. They remain DEV_COLLECTION, with true original task success but no invented route-type or learned-policy claim. Per-attempt paths/events/failures are saved; inspect actual status for completion.
# 2026-10-02: fresh observation development and functioning planner controls

- Hypothesis: peak anchors correct soft-coordinate averaging on unseen layouts. Actual unchanged old64 best transfer across three seeds: strict semantic28.47±3.18%→72.05±3.14%, ADE6.60→6.21cm. Fixed old64 TRAIN prototype reaches40/48 endpoints with severe outliers. No fresh-data fitting occurred for these transfer controls.
- Actual new192 paired training, same384000 candidate slots per arm: soft/peak original ADE-best semantic66.67%/95.83%, ADE2.872/2.649cm. Peak last3000 declines to87.5%; both retained. Larger pool and threefold exposure are disclosed. Conventional repair remains a baseline, not the core contribution.
- A* v2 changes only endpoint attachment after fixed TRAIN diagnosis; all12 DEV commands finished,44/48 TipValid,1.333 classified types,26.67% known-reference coverage,1.969s median. High validity but repeated route types is a real traditional-control result. Full arm/table/execution remain unverified.
- v diffusion pair uses exactly the original epsilon training stream and768000 slots: independent Unique.783854 and set.796875. Set improves over epsilon but roughly three of four candidates collide. Decision: no diversity guidance sweep; prepare one explicit-extra-exposure continuation, while investigating observed path training failures and collecting representative tasks.
# 2026-10-02 12:00 UTC — bounded diffusion decision and new observation experiment

Hypothesis: unstable epsilon reconstruction was the principal diffusion obstacle. Standard v repair and exact bounded continuation to12000 were executed, preserving3k originals and restoring full optimizer/RNG. Independent/set reach Unique1.21354/1.09375, Valid40.17%/33.72%, with4x regression exposure. Both improve but remain collision-limited; stop this branch without claiming convergence or hiding extra cost. MAIN_RESULTS now contains200 measured rows with cumulative and incremental costs separate.

Hypothesis: obstacle reference discretization caused early collisions. All181 TRAIN references pass both raw and actual H24 checks;154 semantically correct peak-last candidates collide,115 first in the first quarter. This rejects the label-discretization explanation. New96/8 ordinary soft/peak pair now runs at3000x32,K4 with prospectively fixed tip-coverage checkpoint selection;14 source-fixed tests pass. Same96 observed A*v2 all24 fresh DEV completed and will be analyzed jointly. No locked outcome was inspected.

Recovery milestone: real simulator subprocess exit3 followed by a new process reproduced complete initial observation/state/language exactly, preserved first-slot hash and closed3 successful attempts. Formal six-task144-parent collection is now running from6c44469,CPU1,with all failures/budget retained. Independently auditing demo rendering cost does not mutate this job.
# 2026-10-02 12:18 UTC — fresh observed obstacles and rejected branch representation

Actual new96 pair (same data/384000 slots): soft best3000 TipValid44.79%,Unique.75; peak best1250 52.08%,1.125. Peak last3000 semantic accuracy rises to80.21% while TipValid falls43.75%; keep both. Traditional same96 A*v2 remains stronger87.5%,1.25, with1867ms medianCPU request. Independent90535b3 analysis verifies all24 outputs,23 reference denominators, initialization reconstruction, actual sampler states and hashes.

Actual online Qwen24-request measurements: soft61.73ms/peak77.20ms median, checkpoint-matched output differences<=1.431e-6m. This includes current inputs/processor/Qwen/geometry/head, excludes checking/scoring/execution. Plot visually inspected; no equal-hardware/time claim.

Hypothesis: bounded observed boundary components represent distinct departure options. Fixed8TRAIN/24instructions/three radii yields72/72J1 with zero failures, no truncation, and no known-type separation. Reject this representation before building an allocator. Next actual action is TRAIN visible-support audit and one conventional local/global draft-refinement pair with identical added parameters and explicit8 path states. It is a baseline repair test, not a novelty claim.
# 2026-10-02 12:24 UTC — collision-location wording clarification

Earlier diagnostic prose called the start of the first intersecting segment a first-contact location. The recorded field is explicitly `normalized_arc_before_segment`, a lower bound on actual contact progress. Corrected the prose; raw JSON, validity, geometry thresholds, checkpoint selection and results are unchanged. New fixed8TRAIN exact slab-contact diagnostic finds17/33 before25% and30/33 before50%; this is a different subset from old32 totals. Prespecify the local/global experiment at Gaussian sigma.10m, prefix.50 arc, coordinatewise residual bound.10m. The bound remains a falsifiable implementation choice, not proven sufficient by point support.
# 2026-10-02 12:30 UTC — actual matched local-reading training launch

Fixed source5501c5f passed25 relevant tests. Fixed TRAIN8 support audit completed0.9503s: all33 contact locations have observed points within.10m;680 reference-prefix samples include32 occluded and5 outside-image queries, so missing points are not treated as free space. Prespecified Gaussian sigma.10m, prefix.50 normalized draft arc, coordinatewise bound.10m; sufficiency unproved.

Executed immutable sequential global/local launcher with actual preflight passed, outputs `runs/observed_refinement_reserved96_v1`. Both freshseed0,3000x32,1,236,960 trainable head parameters; K4 drafts +K4 updated paths and768000 complete training path states each. Original unrefined peak and observed A*4 are explicitly lower-state-count references. This tests ordinary local geometry reading, not a new set contribution. Next action: read logs, recompute draft/final validity and support, compare matched arms, then decide.
# 2026-10-02 12:48 UTC — matched refinement completed; isolate effects before another mechanism

Hypothesis: reading observed geometry near actual draft waypoints improves path quality versus an equal-parameter global update. Actual5501 pair both completed3000x32, seed0,8 complete path states/request. Global/local best TipValid.5000/.5104167,Unique1.083333/1.166667; last .520833/.4375 and1.0/.875. Local costs201.797s versus145.069s, so best-only coverage gain is insufficient and the final-step negative result is retained. Independent draft/final and visible-support analysis is prepared with artifact/shape guards; not yet executed. Next decision depends on whether the update itself helps and whether joint training degraded drafts.

New independent six-task seed282000 source84ee7a6 deployed, with five-task demo-render policy and mechanical layout leakage gate. Original6c formal collection continues. First-parent actual process-resume proof is next, not a claimed completed formal dataset. Local bundledPython lacked pytest; switched to existing Anaconda and17 collector/fingerprint/support tests passed0.66s. No environment modified. Future-only eval_every resume guard added after review; completed fresh pair unaffected.

## 2026-10-02 12:59 UTC — reject local update; actual new formal collection and multi-task interface

Fixed d305 analysis ran11 tests and3 independent jobs, all exit0. local best only repairs1 candidate; last repairs0/breaks3. No coordinate-bound saturation; most prefix queries have nearby observed points. local last draft46.88% Tip exceeds ordinary peak last43.75%, so generic draft deterioration does not explain this failure. Freeze-reference extra training is not justified; stop this conventional module, retain peak/A*. Saved96 draft/final slots per phase and quality/cost figure retained. MAIN_RESULTS209 rows explicitly count8 path states and avoid charging last-checkpoint rows for another training run.

New84ee7a6 collector actual process proof passed19 server tests and exact restoration over3successful attempts;46.397s. Formal run now started in observation_multitask_validated_five_formal_v1, PID316991, source84ee7a6, while old6c continues independently. First new snapshot fixed to each task's indices0/1 TRAIN and16/17 DEV, all24 parents must close and cross-role layout gate must pass; no substitution of successful parents. No snapshot/learning results yet. Generic multi-task head needs optional free endpoint because lift/lid endpoints may be in air, and preserves unmeasured semantic/validity asnull. Next research mechanism starts from real TRAIN predicted duplicates/known-reference coverage, not unsupported claims that across-scene mode frequency equals within-scene duplicate supervision.

JOBS mechanical snapshot now199 records; registry202 including retained unmatched history. Source d305 pushed and remote head verified. No after-session autonomous thought is implied; active scripts only collect registered attempts.
# 2026-10-02 13:31 UTC — multi-task data and direct-set SFT boundary

Hypothesis: a fixed surface endpoint cannot represent some lifted-object task references; full-sequence vocabulary logits also needlessly exceed the current memory cap during teacher forcing. Changes: opt-in free 3D endpoints, task/parent-balanced sampling and macro reference metrics with sealed snapshot/live mechanical gates; exact answer-only causal vocabulary loss in recomputed 64-token chunks.

Measured: prefix24 has24 closed parents/72 successful references/96 language inputs; all36 TRAIN references retain event order, cup6 and lid6 endpoints exceed the hard-anchor5cm-per-axis capacity. Server source6bc2b8 passes38 multi-task/history tests. Initial actual Qwen SFT source5fb74b failed OOM/exit1 and is retained. Source6bc2b8 v2 passes7 serialization/loss tests and4 real optimizer steps, all8 LoRA tensors updated, peak4.839GB,4.412s internally. This is boundary/memory evidence only, not trained generation quality.

Independent TRAIN96 audit found393 classified valid duplicate predictions but only28/288 instructions with both duplicates and known missing types. Mature type-coverage reweighting is not yet justified as a core mechanism; known per-scene reference duplication is zero. Actual next actions: frozen-Qwen six-task ordinary baseline, resumable direct VLM SFT and same-checkpoint independent/whole-set generation, plus physical multiopening pilot design. Locked roles remain unused for development.
# 2026-10-02 15:10 UTC — completed generation failure, larger ordinary baseline and bounded next tests

Measured follow-through:0eeeecb grammar TRAIN8 finished8/8 format but0/8 endpoints within3cm,57.372cm mean; stop DEV/K4 expansion. Each sampled instruction actually had3–8 training requests, so this does not establish adequate SFT fitting. Next is one fixed9-forward TRAIN coordinate/mask/shift/NLL audit, not another long rollout.0eeeecb endpoint IK24 completed48/48 exact restores, both orientations3/6 collision-aware and6/6 ignore; added configurations collide. No new routes/reference data. A corresponding static collision-body comparison of six frozen configurations is being implemented with no claim of original-instant reconstruction.

The aef3983 ordinary supported-event auxiliary passed37tests plus real initialization/48000 sampler replay, then completed1500x32 from reviewed launcher55c2c4a. Best=last1500 DEV macro ADE9.3822cm/endpoint15.9017cm versus unchanged ordinary best10.0004/17.4344 and last10.2442/18.7515. Head82.757s versus78.262s; same1,231,965 parameters. Improvements concentrate in slide/push/lid; lift best endpoint worsens7.570→10.533cm and cup slightly worsens. Retain both checkpoint views and all parents; prepare identical ordinary+aux seed1/2 replications, not seed shopping or a new-core claim. A saved-output first-close location diagnostic is separately being added because event-order accuracy does not measure event position. New formal84ee collector finished2026-10-02T15:10:13Z; only role closure/exit metadata will be inspected outside TRAIN/DEV.

Hypothesis: low teacher-forced loss might yield useful direct-VLM candidate sets. Actual8356b09 free generation refutes that at the current3750-route training exposure: strict-format21/96 independent versus5/96 whole; semantic/TipValid bothzero; every complete JSON endpoint also exceeds3cm. Preserve all120 raw requests and all24 instructions including the no-reference case. Training and generation cost are reported separately. Decision: repair syntax only as an eight-TRAIN capacity diagnosis, not immediately repeat expensive all-DEV inference.0eeeecb20 CPU tests and true tokenizer preflight passed; actual TRAIN8 generation is running, no further DEV launch.

Hypothesis: more independent TRAIN parents improve the same ordinary Qwen head. Actual6464 prefix108 (96 requested/95 positive TRAIN parents) best750 macro DEV ADE10.000cm/endpoint17.434cm, versus48 TRAIN parents12.944/19.700. Same1500x32 exposure, shared DEV bytes and reconstructed initialization/RNG verified; last1500 and worsened cup/lid cases retained. Data variation coverage changes too. Decision: one ordinary supported-event attention auxiliary, same architecture/exposure/weight.02/sigma.025, pending source-frozen tests and actual initialization/sampler proof; no new-core claim.

Data feasibility: strict two-row old-state reconstruction failed before proposals; record27 unattempted. Since v2 failures arise inside endpoint IK before OMPL, a bounded24-query diagnostic now checks six points×two actual orientations×collision filtering, same native snapshot and fixed search budgets. Failures and exact restore gates retained; no controller research expansion.

## 2026-10-02 16:02 UTC — three-seed evidence and next bounded controls

Completed the four additional source3f9cea5 runs; original best ADE improves in all seeds, fixed1500 only two, cup remains a cost. Preserve 84 indexed artifacts and actual index-chain pairing. Auxiliary stays an ordinary strong baseline, not a novelty claim. Source1915de7 nine-forward audit rejects unit/mask/causal leakage bugs; coordinate tokens dominate error. Conditional endpoint accuracy relies on previous true route points. Next actual decision is a single same-checkpoint greedy TRAIN8 control, with source/tests first; no temperature/beam sweep. Static collision diagnostic failed before all six saved configurations; legal root sentinel API repair is isolated and tested, to be retried only from a fresh immutable source/output. No thresholds, task geometry, saved joints or evaluation criteria change. Saved-prediction figures will include all12 DEV parents, first sorted language, shared paired axes.

## 2026-10-02 16:20 UTC — decoding control and physical bottleneck

Source2c47 greedy TRAIN8 completed eight calls/eight strict-format routes with no timeout or retry. Correct endpoints remain0/8; mean target error falls57.372→23.391cm, below neither the predeclared4/8-success nor15cm-mean gate. Keep partial decoding gain; no temperature/beam/seed sweep. Next baseline decision is bounded more-exposure SFT from the actual1500 state, under a new continuation protocol and fresh output, not a changed old resume config. No further training has launched yet.

Source2c47 static six-configuration analysis completed after preserving two API failures. It applied all6 configurations, strict12 pre/post restores, no IK/new paths/simulation starts. Full body-matrix analysis will decide one new collection geometry, not a controller innovation. Old formal collector remains running (112/144 closed markers at16:17UTC), while the newer144-parent collector is complete; locked contents never opened.

The next sole set-mechanism candidate is common failure under task/geometry changes. First perform a registered TRAIN-only opportunity check with all single closures and explicit no-solution cases; reference-pool optimum is a privileged diagnostic, not a K2 model. ExistingK4 prefix2 is zero-adaptation only. A genuine trainedK2 saturation baseline is mandatory before concluding a joint-risk objective can help. ModeSeq and LookOut already establish prior-mode memory and decision-relevant diversity; neither phrase is a novelty claim here.


## 2026-10-02 16:42 UTC — one physical feasibility gain; common-failure hypothesis rejected

Hypothesis: lowering the diagnosed gripper/post obstruction may permit more than four actual route relations. The preregistered v4 central-target nine-slot pilot from7b496e completed8 valid routes,5 distinct lateral sequences and3 valid unknowns, with9 exact restores. One raw/H24 type mismatch stays rejected. Total62.996s. This is a new single-parent feasibility result, not a paired causal height comparison or trained-method claim. Actual next implementation: four once-sampled physical layouts with three same-image target instructions; fixed height/checker and complete108-slot denominator.

Independent hypothesis: response-aware finite-budget selection may avoid common failures beyond geometry diversity. TRAIN768/3806 single closures falsify the registered opportunity: all720 evaluable parents tie between geometric farthest/DPP and fixed reference response oracle,100% solvable-closure AnyValid. Stop the proposed risk module; retain one genuine K2 ordinary budget control rather than treating a K4 prefix as strong.

Greedy SFT still fails TRAIN8 endpoint gate, so one bounded same-objective1500-to6000 continuation is prepared. Restore all states into a fresh tree, preserve original artifacts/best bytes, and account original/added cost separately. Local pure tests and peer review passed; actual server Torch equivalence remains required before GPU work. No new training outcome is inferred from source preparation.


## 2026-10-02 16:57 UTC — trained small-budget control closes the apparent gap

Actual5bb9087 K2 saturation3000x64 completed:static DEV best500 and last3000 bothValid100%/Unique1.9375;384000pathslots. TRAIN closure99.65278%AnyValid versus83.66353%unadaptedK4prefix. Every remaining16solvable failed closure across8parents involves an originally invalid candidate; zero failures from two originally valid distinct routes. Stop common-failure objective design in this setting.

Actual next execution: afterK2 releasedGPU, full-state Qwen6000 continuation launched16:52:48UTC fromsame5bb,child440254,13actualCPU tests passed. Original checkpoints unchanged. Independently, four-layout108proposal physical pilot launchedfrom9324efc after62tests; sameimage3target conditions and allfailures preserved. Oldformalcollector125/144closed mechanically at16:53; lockedcontents not opened.

## 2026-10-02 17:35 UTC — lower NLL fails free generation; physical diversity pilot closes

Hypothesis: bounded additional identical SFT exposure could fix the TRAIN free-rollout gap. Actual4500 added steps lowered best NLL .436727→.416633 but the unchanged eight greedy requests remain0/8 endpoints within3cm; mean error23.391→27.769cm,3better/5worse. All8formats pass, original9hashes unchanged, added1777.327s and11250route slots separately accounted. Stop this mechanism of repair, preserving full states and failures; no DEV/K4 expansion.

The fixed four-layout/three-target collection completed108 requested slots:81attempted/61accepted,49classified/12unknown,20failures/27unattempted setup slots. Six target conditions supply≥5known route types; all81route restores pass. This supports richer data feasibility only. Newly audited RLBench reset-internal planning was not individually counted by v4/v5; walltime includes it. Correct the scope of83/683explicit calls without rewriting original logs or inventing internal counts.

Actual next work: implement one new canonical static initial state using the already legal v4 joints, audit all4registered layouts without initialization planning, then only27slots for the formerly failed layout. Separately, eligibility of all12old DEV parents passed duplicate and indexed prior-use checks; prepare frozen transfer of all3seeds×ordinary/aux×best/last, with no retraining or reselection. The sole new set hypothesis remains conditional on a rich-data ordinary baseline; no core mechanism advantage is claimed.
