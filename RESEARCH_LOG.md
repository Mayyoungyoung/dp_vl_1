# Research log

## 最新实际快照 — 2026-10-02 22:41 UTC

本段覆盖下方较早的running/待运行记录。root实核MAIN267（原263所有字段严格不变）、JOBS484@22:40:58.411450UTC无新job运行、registry501；429部署已登记。cosine与native100k均已实际完成，不能按旧记录重复启动。

两臂共同100k/2秒/K4的native传统对照，source42904f0，DEV均Tip96/144=66.67%、Any24/36；edge/spatial已分类不同有效数24/36=.6667、62/36=1.7222，已知类型覆盖.0319444/.0685185。全部36请求首路径相同；两臂均44附着失败+4有限错误目标、0节点上限/0超时。完整请求中位.707706/1.220680秒，共同body57.463680秒；更多类型保留了有效槽率，但请求耗时更高。它是标准传统规划的质量—覆盖—成本取舍，不是新学习核心、不比原20k同节点预算，也不构成固定端到端时间优势。完整100k归档由observation负责；原192edge等价证据保留，新100k不替代它。

cosine同预算12000末步TRAIN Tip94.84%改善拟合，DEV26.39%低于constant43.06%；保留best/last全部结果，停止LR扩展。[完整结果](reports/observed_two_row_prefix76_cosine_v1/COSINE_RESULTS.md)。没有新核心方法胜利，也没有12000真实Qwen在线时延重测。formal116 reserved raw持续封存。

下一步仅只读评估独立256TRAIN+32新DEV父的数据扩展成本/去重/最小实现，保留旧12DEV的reused性质；尚未授权采集或修改旧collector。数据规模控制本身不能称方法贡献。

## 2026-10-02 22:21 UTC — cosine配对完成：训练拟合改善，DEV泛化不成立

假设：在同数据、初始权重、抽样链和12000步下，单周期cosine可能缓解恒定LR后期末端漂移。修改：独立scoped日程，仅LR变化；原constant代码/权重/数据/阈值不变；91实际服务器测试含真实Torch恢复/历史等价全部通过。随后root独立fresh运行到12000，未换seed或追加预算。

单次cosine普通控制已实际完成，source71e34850481b79bdbc828d1c1944f0de00329960，PID593217/593221于22:21:21.678075UTC退出0。91服务器测试均通过；初始化、完整384000抽样链及最终RNG/sampler严格同constant12000。两臂均1536000训练路径状态、48次DEV选择，只有LR日程变化。相同64请求TRAIN父中63可观测父/189输入，原缺失父不替换，DEV固定36输入。

固定last12000 TRAIN Tip575/756=76.06%→717/756=94.84%，语义81.75%→98.68%，候选→最近正参考ADE2.791→.785cm；DEV Tip62/144=43.06%→38/144=26.39%、Any28/36→22/36、knownUnique.6389→.4444。cosine原规则best4250 DEV Tip52/144=36.11%、Any27/36、knownUnique.8611；constant best5500为59/144=40.97%、30/36、.8333。保留cosine best多1个跨条件求和已分类有效类型的取舍，不写全面优胜。严格配对支持训练拟合改善，不能支持DEV泛化改善或新集合贡献。

判断：TRAIN末步Tip条件80改善/92相同/17下降，而DEV7改善/9相同/20下降；DEV失败同时有端点及碰撞。cosine TRAIN最后10个端点失败都在3.095–3.498cm；DEV39个端点失败包含16个>6cm，不能统一解释为身份错误。最后3000步DEV Tip仅26.39–31.25%，训练path loss仍降至.000021655。支持该优化调度改善本次训练拟合，否证“更好拟合足以改善此DEV”的预期；不从这一结果推导新的核心模块。

实际执行收尾：原始34文件、12000 LR、完整48次DEV历史、checkpoint索引及189/36四池归档；本地0forward重新聚合，12父36指令全部图逐图QA。base528.534s/.146815 GPUh，外层542.291s；固定lastTRAIN额外189请求756状态已纳入job成本，没有新Qwen。决定停止LR/步数/seed扩展、保留constant；下一独立传统100k共同节点预算控制已由root从42904f0实际启动TRAIN门禁SSH66842，尚无本记录中的结果。详见 reports/observed_two_row_prefix76_cosine_v1/COSINE_RESULTS.md。

## 2026-10-02 22:15 UTC — native conventional tradeoff verified; cosine engineering gate passed and fresh run launched

实际快照22:15:44.071911UTC：MAIN263（原261所有字段不变），JOBS479、registry495；cosine PID593217/child593221于22:12:19.387085启动，SSH99280仍在，约4500步。此为该时刻running记录，不是完成状态。100k控制仅此一个档位，两臂共同变更、K4和2秒上限不改，尚未冻结或运行。

Hypothesis: shared native implementation can remove the Python search-time bottleneck without changing either planner. Actual sourceb184060 passed74Linux tests,TRAIN andDEV stages each exit0. Full saved-pool comparison includes every192historical edge slots:raw/H24/events/deterministic counts/checker outcomes all exact. NativeDEV edge/spatial hasTip96/144→74/144,knownUnique24/36→52/36,Any24/36both,classifiedduplicates65→22;full-requestmedian.6264→.9263s.44goal-attachment failures remainboth,spatial adds24actual20000-node caps and0timeouts;4versus2finitewrong-goal routes retained. All12parent figures and288newDEVslots archived/QA. Judgment:more useful traditional coverage with a quality/time cost,not novel learned mechanism or fixed-time superiority. Preserve Python negative and20k source. Next one100k-node BOTH-arm control is independently in implementation;no result yet,clearly different node budget.

Optimization hypothesis remains unproven:one fresh12000 cosine may reduce constant-run late endpoint drift. Source71e348 passed91realserver tests in8.85s,no skip including3Torch cases:actualsharedloop continuous200 versus100+100 resume fullstate/history,andconstant-adapter equivalence. ActualrecordPID592391/59239322:10:44.793556→22:10:54.265947UTC,exit0. Archive18evidence items+335sourcehashes,wrapper byte equality andJUnitcase checks complete. Root then independently launched CPU0/GPU1/35% SSH99280,fresh sameinit/samplechain/189inputs/1536000states/48DEVchoices. Atthisrecord run isnot declaredcomplete;no dependentfinalproductsread,no researchoutcomeinvented. No warmup/restart/floor/LRsweep/newseed. Afterexit analyze allsavedbest/lastTRAIN189/DEV36 with0newforward. No failure-memorymicroaudit ornewsetmodule is scheduled.

## 2026-10-02 22:04 UTC：同预算cosine普通控制准备

独立五文件已实现、复审无阻断；本地13pass/3因无Torch而skip。下一步在冻结源码上通过实际Torch恢复等价/历史回归门禁，之后单独启动fresh12000。相同真实Qwen缓存、初始化、189 TRAIN输入完整抽样链、K4/H24、损失、1536000路径状态和48次DEV选择；仅替换为无warmup/restart/floor的单周期cosine。原constant12000记录保持不变。不因新DEV选择更多步/曲线/seed。

native b184060的Linux74测试、TRAIN24请求96槽和DEV72请求288槽均实际完成退出0。DEV edge Tip96/144、Unique24/36；spatial Tip74/144、Unique52/36；Any均24/36。spatial24节点上限失败全部20000节点、搜索中位18.434ms，无超时。历史Python/native edge全部192槽raw/H24/events/确定性搜索计数及checker决定一致。全归档/逐图QA正在收尾，MAIN尚未加入这两新行。此为标准传统对照的质量覆盖成本取舍，不是新集合贡献。

## 2026-10-02 21:54 UTC：下一配对控制源码冻结准备

证据e2f67751e05dec9986a15234b8492f773077d789已推送并核验。两臂共同native A*内核、三阶段runner/launcher及门禁测试已实现并独立复核；本地实际编译74 tests通过，Linux服务器编译/测试与真实请求尚未运行。生产沿用K4/20k节点/2秒/64节点计时，两臂共同使用同一库，原planner/空间场/评价器源码未改。首次tests stage必须有真实24项native差分及同编译器/flags/源码生产构建证明，root读后才能单独TRAIN12，再单独DEV36。

实现审查已在真实请求前修复两处测试入口问题：其它测试不能替代24项native差分；单元测试的模拟build receipt不能被当作真实编译。一次本地测试临时目录错误设在源码内被既有保护拒绝，保留失败日志，移至OS temp重测，未放宽保护。新的cosine12000普通基线仍在独立实现，未启动。

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

## 2026-10-02 18:10 UTC — canonical initial state works; independent-parent transfer is mixed

Hypothesis: define a legal common static q instead of relying on a preparation motion. Actual60a01ea passed4/4 initial/settling/restore/depth gates with zero initialization planning; only former failed geometry283102 received27slots, yielding23valid/14known/9unknown/4failures. Long routes remain (max3.773m), no threshold/success filtering. Decision: register a distinct narrow-ID116parent corpus, role-grouped, two CPU1 shards, first16TRAIN+all12DEV; implementation under way, no launch yet.

Independent test of prior conventional auxiliary: frozen12checkpoints on all12 previously unused old DEV parents, 48newQwen encodings. Actualfe564 best mean ADE improves4.74% butseed1worsens;fixedlast meanADE worsens1.49%,6/12parent means improve. Endpoint means improve both, event state slightly worsens; do not claim stable general advantage or new core. No further seed/tuning, retain stronger ordinary comparisons and all negative rows. MAIN adds12zero-training transfers with shared encoding cost deduplicated.

Next actual engineering: sealed closed-prefix TRAIN/DEV-only export, ordinary K4 positive saturation and TRAIN endpoint-support audit; pure two-row evaluator uses the exact collection passage classifier with unchanged semantic/start/event/tip validity. Near-neighbor audit finds cross-goal topology identity already explicit in T-MPC, so proposed cross-goal auxiliary remains conditional conventional hypothesis, not novelty.

## 2026-10-02 18:33 UTC — formal116 actually launched

Hypothesis: canonical static initialization enables a prospectively registered multi-goal route corpus beyond three fixed modes. Change: immutable2626487 freezes116once-sampled parents, five roles and3132slots, exact/1mm historical exclusions, two fixed CPU1 shards, mechanical partial recovery without replay, role-isolated raw outputs. Local111tests10.71s; server111tests7.88s, prepareexit0. Actual workers485599/485598 started18:30:34UTC, outer485548/SSH85859. ArchiveSHA767b3e0f5a78c1e78998dae8549100f329c31d2516d2e29b3381cfbb4ae90024 verified remotely; launcherSHA8e4a830ff0200d071dad2647f5a4366db1d0c1dfb7dcbf4e123a9fa0a7ea4d20. No collection/model success inferred.

Next actual work: all16TRAIN quality and full-slot visualization, sealed16TRAIN+12DEV export, TRAIN-only endpoint/H24 capacity audit then ordinary frozen-Qwen K4 training. No core module before ordinary prediction evidence. Current worker progress and failures must be read before any restart.

## 2026-10-02 19:12 UTC — rich reference pool passes TRAIN representation gate

Hypothesis: the registered two-row corpus provides more than K4 known route relations without relying on guide labels. Actual first16TRAIN analysis fromce548 completed432slots:285accepted,181known/104unknown,147failures;14/48conditions R>4, all432strict restores pass. All48full-slot XY/XZ plots and16original images retained and reviewed. Unknown positives and up to5.895m long arcs remain. Known mode counts are positive support, never total solutions. No learned advantage follows from these collection results.

The prospectively fixed16TRAIN+12DEV export completed19:10UTC with84actual inputs/490positive references. TRAIN-only observed-point endpoint capacity and modelH24 checks pass285/285; use the originally registered surface-peak representation. Sourcece548 actual83server tests include optimizer/scheduler/RNG/sampler resume. Source743 online timing14tests pass. Original failed zero-test invocation is preserved. Actual Qwen cache then ordinary1500x32/K4 training launched SSH7878 CPU0/GPU1/35%; no model result yet. Formal collection continues CPU2/3.

Decision/next execution: frozen same-data original A*v2 control (six local tests and independent no-leakage/source-budget review) runs on CPU1 after immutable server tests. Read ordinary full best/last candidate pools and all12parent plots, measure actual single-request Qwen latency, and identify one supported method intervention. No extra seeds/modules before an opportunity exists; paper-core remains unproven.

## 2026-10-02 19:30 UTC — ordinary fits TRAIN but fails to generalize; strong control preserved

Actualce548 ordinary1500x32 completed in64.700s internally,192000training path states; realQwen84cache9.554s internally is separate. Best500/last1500 DEV each33/144TipValid, knownUnique.5833/.3611, both zero classified-valid duplicate slots. Semantic47.22→64.58% butTipClear42.36→33.33%. All36requests/12parents/4candidates and failures retained;12plots independently checked. Source743 online36actualQwen requests match cached decisions with73.146msmedian/80.546p95/656.477first continuous latency including output sealing and labelcheck. No learned scoring or full-arm claim.

Do not infer failed TRAIN fitting from the early-selected best500. Actual743d9b2 new48TRAIN last1500 cached-head requests (192pathstates,0optimization) find Tip87.5%,Any100%,semantic96.35%,knownUnique1.4792,validunknown2.0208,dup0. Saturation exact matched loss.0050795→.00026637 and ADE11.437→2.728cm show real fit. Decision: first test fixed32 and64TRAIN expansions at unchanged exposure and same12DEV; do not add a duplicate-suppression module whose measured DEV opportunity is zero.

Traditional control initially failed5pass/1fail before generation because localLF SHA differed from archivedCRLF. Proven current server bytes equal historical successful A*v2 bytes; new guard accepts only the two verified exact forms plus canonicalcontent check, records actualSHA, shared planner/thresholds unchanged. Corrected743d9b2 tests10pass. Actual36conditions/144slots:76valid,68failed;8unsupported language conditions and9no-goal-attachment conditions. KnownUnique.5278 andmean classifiedduplicate1.3056, median779.898ms CPU. This control is higher quality but more repeated and slower; sameK is not samebackbone or fixedtime. Do not claim an overall new-method advantage.

MAIN now248rows; original training counted once, same-checkpoint last and online separated, actualQwen cache deduplicated. JOBS19:26:46 records372jobs; only two formal collection shards/parents active. Read fresh closure/PIDs before any restart. Next implementation is prefix44/76 exact-role export/capacity/ordinary equal-exposure pipeline, not running yet. Papercore A–H remains incomplete.


## 2026-10-02 20:05 UTC — fixed exposure expansion improves Any, not overall route quality

Hypothesis: additional independent TRAIN parents improve the ordinary frozen-Qwen baseline at identical exposure. Actual9e0094a runs32requested/31observedTRAIN parents after140tests,129realQwen encodings,556/556 TRAIN capacity/H24 positive checks, same36DEV bytes/features and initial weight digest. All1500x32/K4 exposure audits pass. Missing3input/27slots remain. Best1000 DEV Tip21.53%,Any66.67%,knownUnique.5 versus prior22.92%/44.44%/.5833; fixedlast21.53%/47.22%/.5. This is a mixed baseline result, not a method gain.

Further actual lastTRAIN93 evaluation:Tip36.02%,semantic81.72%,exact saturation loss.00298756, far less fit than16-parent last87.5%/.00026637. Decision: keep the fixed64/1500 scaling commitment, and independently implement bounded6000 ordinary convergence to avoid comparing a new mechanism against an undertrained baseline. Fresh6000 schedule first1500 must match original complete training state; only after that gate may it resume to6000. First16TRAIN-only cross-goal correspondence opportunity diagnostic proceeds in parallel; no new module before stable positive evidence. Actual prefix44 online and all12 saved-pool plots/lastTRAIN checks completed; original hashes and costs archived, MAIN251. Formal116 collection continues; no locked raw consumed.

## 2026-10-02 20:55 UTC — ordinary convergence established at32; a proposed association gate fails

Hypothesis: the poor32-parent ordinary result is partly insufficient optimization rather than an irreducible representation limit. Change: a separate preregistered constant-LR6000 control, same prefix44/93actual inputs/seed0/batch32/K4 and sourceac6882c. Actual73server tests passed withnoTorch skip. Fresh first1500 model/optimizer/scheduler/scaler/RNG/sampler/index-chain/best/history/identity exactly match the original reference, tolerance0; root then independently launched finish. Both stages exit0, outer cost69.831+212.837=282.668s,192000 input draws/768000 training path states. Original checkpoint preserved. No automatic extension beyond6000.

Measured: fixedlast TRAIN Tip36.02→97.04%,semantic81.72→98.39%,candidate-to-nearest-positive-reference ADE8.213→0.780cm. Newbest2500 DEV Tip31.25%/knownUnique.7778;last6000 Tip34.03%/knownUnique.4167. Best/last DEV still have68/52 semantically correct but colliding slots, zero classified-valid duplicates; unknown valid slots17/34 retained. Extra fixedlast TRAIN93 requests/372 states cost3.612s CPU, nonewQwen/DEV/optimization. Read-only independent recheck/all12plots cost20.997s and0forward; original47archived files unchanged, previousindex preserved. Judgment: ordinary capacity to fit31actual parents is established, while DEV geometry/classified coverage remains weak. This is fourfold training and24versus6selection opportunities, not same-budget mechanism improvement.6000 real-Qwen online latency has not been remeasured;1500 timing cannot be relabeled. Historical convergence table's shorthand matchedADE means candidate-to-nearest-positive-reference ADE, not saturation-assignment residual; clarification is in METHOD.md without modifying archived bytes.

The promised fixed64/1500 ordinary point also fully completed and archived:63actual TRAIN parents/189 inputs/1108positive references, absent283220 retained;best DEV Tip24.31% andlast18.06%,last TRAIN20.11% despite semantic92.99%. AllTRAIN capacity/H24 checks pass, but this run remains underfit.32-parent6000 success is not evidence that64parents have converged. Existing original1500 scaling, fixedlast andonline receipts remain intact; a possible separately bounded64convergence control is only being assessed, not launched or implemented.

Independent mechanism hypothesis: stable cross-goal positive correspondence identifies a useful correction target. The preregisteredfirst16TRAIN audit found stable reference crossings, but the decisive collision-minus-clear excess is.763345cm, below2cm. Decision: stop this specific auxiliary trial, no threshold relaxation/DEV opportunity search. Same-type closeness alone is partly induced by collection/type definition and is not novelty or causal gain.

Strong traditional controls now fit the same32/64TRAIN data: original observedA* gives96/144 and100/144TipValid, with4finitewrong-goal slots each and44/40ungenerated slots; classified duplicate means1.8056/1.9444. All144slots remain. Next actual implementation selected: standardGaussian spatial-path penalty, separate new module, scale2voxels=.05m/penalty4, sameK4/observationproxy/endpoints/20k-node/2s limits. First fixed4TRAINparents12inputs old/new feasibility before separately reviewed36DEV; noacceptance labels in generation, no novelty claim, no execution result yet.

At20:45UTC the mechanical snapshot had451jobs/463registry entries, formal116 parents98/99(CALIBRATION) onCPU2/3 andnoGPUjob. Only mechanical reserved-role information inspected; locked raw/metrics remain sealed. Root owns MAIN/JOBS/registry/Git updates; this documentation update does not launch work or claim the paper core is established.

Subsequent actual decision: root authorized implementing a distinct prefix76 fresh12000 convergence control so the visibly underfit64/1500 model is not used as a weak baseline. No launch yet. Keep prefix44/6000 source unchanged; exact first1500 original64 full-state proof precedes separately launchedfinish, with189input fixedlast TRAIN diagnosis.384000draws/1536000pathstates give2031.75draws/input versus32/6000's2064.52, approximately aligned per-input exposure but twice its total training and48versus24 DEV selection opportunities. Source/config/tests/protocol must freeze before execution.
# Implementation record — 2026-10-02 21:12 UTC

After actual32/6000 proved97.04% TRAIN TipValid while DEV remained34.03%, and64/1500 TRAIN remained20.11%, implement a separately bounded64/12000 ordinary control. Same architecture/known positives/Qwen cache/K4,1536000states, approximately matched per-input exposure but twice32/6000 total cost;48 versus24 selection opportunities and fixedlast are explicit. Before finish, a new first1500 must reproduce the original64 full state exactly. Copy-derived receipt constants were corrected before any experiment using budget identities, with regression tests; old6000/base source remains unchanged.

In parallel, original observation A*32/64 has higher quality but65/70 known duplicate slots. Implement one standard Gaussian spatial path-penalty baseline, fixed sigma.05m and unchanged4x cost multiplier, geometry/endpoint/20k/2s search limits. Four searches include every failure/duplicate; every prior returned raw path enters the penalty regardless of later checks. No learned mechanism or novelty claim. FixedTRAIN12 old/new mechanical preflight96slots, then separately authorized fixedDEV36/144spatial slots; preprocessing time is explicit, no equal-time claim. Local convergence61pass/2Torchskip and spatial23pass; real server tests and runs are next. Independent spatial review has no blocker. Current evidence push84cd563 verified. Formal116 mechanical112closure markers at21:06; reserved raw remains unopened.


## 2026-10-02 21:41 UTC — stronger64-parent ordinary control, unstable last localization, and a traditional diversity tradeoff

Hypothesis: original64/1500 was undertrained; approximately aligning per-input exposure to32/6000 may improve the ordinary baseline. Change: separate source1417cdc fixed12000, same63actualTRAIN parents/189inputs/1108positive refs/K4, after98realserver tests andexact original1500 full-state proof. Root independently launchedfinish; both stages exit0. Actual384000draws/1536000pathstates,48DEVselection opportunities;per-input ratio to32/6000=62/63,totalcost2x. Outer559.381s,base nested536.405s;extraCPU fixedlast189requests/756states6.527s already insidefinish. No newQwen encodings, no newonline latency measurement.

Measured:best5500 DEVTip59/14440.97%,Any30/36,knownUnique.8333;fixedlast12000Tip62/14443.06%,Any28/36,Unique.6389. BestDEV2classifiedduplicates,last0. TRAINbestTip85.45%/semantic97.09%,last76.06%/81.75%. Lastsemanticfail138allendpoint3.006–4.288cm,133within3–4cm;clear improves665→690whilecorrectendpoint734→618. Savedanchor shifts correlate with57degradedconditions,median3.54cm versus1.26cm in125unchanged;7improve. This is evidence ofendpoint stability/threshold failure,not proof ofwrongidentity orconstantLR causality. All48DEVhistory,120loggedrecent-losswindows,189TRAINconditioncomparisons andoriginalbest/lastpools preserved;no reselect/retrain/thresholdchange.12000stops atregisteredlimit.

Read-only root diagnosis21:29:20.698–21:29:42.312UTC exit0,newforward0,all12DEVparentplots actuallyviewed.42rawconvergence artifacts and4checkpointindexes verified;original51filearchive index preserved unchanged,expanded75fileindex includes12plots+fullQA. Reports/observed_two_row_prefix76_convergence_v1 contains reproducible saved-pool scripts/hash/costs;old1500/6000bytes untouched. Nextscientificcontrol must be prospectively isolated;not autoextendtraining.

Parallel standardspatial A*completed:fixedTRAIN12old/new firstpaths identical,thenfixedDEV36. DEVknownUnique24→29typesbutTip96→42/144;median.914→6.143s,56additional2s searchtimeouts,44originalattachmentfails,2finitewrongendpointslots retained. Fieldonly6.18%of extraDEVwall;newtypesdoexist,butquality/time worsen. Decision:preserve negative result;prepare SAMEalgorithm native implementation forBOTH arms andcheckcost/tie/neighbor/path equivalence before original20k/2s remeasurement. No native experiment or mechanism claim yet.

Formal116 source2626487session ended21:10:27UTC,twoshards exit0,116closure filenames mechanicallypresent. Only registeredroles/opaque hashes/statuses inspected forreservedroles;thisisnot116successful scenes or3132validroutes. No lockedraw/metricsread,no collector restart. Coremethod advantages remainunestablished;rootowns MAIN/JOBS/registry/Git.

Root follow-up at21:41UTC: MAIN261 rows retain all original258 fields; JOBS473 at21:40:55UTC has no live PID, registry489. The next neural control is now fixed: one fresh12000 cosine-LR ordinary pair, same initialization/sample chain/189 inputs/48 DEV selections, no warmup/restart/LR sweep/new seed. Constant-LR source and results remain untouched. Implementation only; no experiment launched and no causal or positive result claimed.

## 最新实施状态 — 2026-10-02 23:13 UTC

本段覆盖下方较早的待运行和资源状态。研究分支 `codex/multiroute-v2` 的905214e8050d594b157dc7cacf6b446e7c183c85已推送、核验并冻结到服务器。TRAIN条件诊断14项真实Torch测试全部通过（2.03秒、0跳过），随后独立audit于23:09:22–23:10:13UTC完成exit0，PID619059/619064。36次K4前向、144路径状态、0新Qwen编码、0搜索、0优化更新；不是MAIN质量实验。正常预测复现、identity、geometry和权重字节门禁通过。逐场景解释与归档正在处理，不重跑已完成诊断。

新extension288实现为独立注册和采集器：256TRAIN＋32后采集封存DEV，7776预登记提案，首个明确阶段只有train32；后续阶段按真实闭合状态单独推进。旧代码/116父数据/失败/角色保持。当前本地完整157测试及最终采集20测试通过，服务器tests/prepare/采集尚未运行，须按真实job receipt更新。8GiB新增内部预算、CPU2/3、GPU隐藏；原12DEV继续标记reused。用户已授权的研究闭环继续，新增数据不是新方法证据。

cosine和native100k结果已完整归档、实核并提交82c17d2。MAIN267保持；新诊断不写作泛化收益。cosine改善TRAIN拟合但未改善DEV，停止LR扩展。native100k的空间惩罚增加路线类型但请求成本更高，仅是传统对照。论文核心优势仍未成立。

## 当前执行与下一决策 — 2026-10-02 23:24 UTC

本段覆盖下方历史running记录。MAIN267不变；JOBS497@23:23:57.395618UTC、registry随后按同快照同步。source905214e的TRAIN12诊断已经完成并归档：36 K4前向/0搜索/0更新，直接分支swap使净空45→33/48（14坏2好）；仅通道敏感性，不是DEV因果或方法质量。69条正参考完整grid通过0受到终点voxel与其它保守阻塞共同影响，不能推导模式不存在。[完整结果](reports/OBSERVED_TWO_ROW_ROUTE_CONDITIONING_RESULTS.md)。暂缓learned-field，下一唯一神经控制为去direct分支的普通头：相同constant12000曝光/共有init/抽样，geometry等剩余模块继续训练，参数量减少明确披露；尚未server训练。

extension288 source5c8f8e4f5cd478c793a0e0d9640005deaf700973已实测158tests通过、独立prepare成功并封存来源。root单独启动train32 fresh，session `20261002T231813Z_train32_624110`，SSH89814，CPU2/3、GPU隐藏。两coordinator PID624140/624142，child624144/624145；23:23:57快照前四parent job exit0，父004/005运行中。这是机械进度，不是成功场景/有效路线数。此阶段只32父864请求槽，后续不会自动采集。新DEV须全部256TRAIN机械闭合后另启并继续封存。

恢复前先核实session/status、parent status与PID实际command；完成的stage和已发父绝不重采。确有中断且现有coordinator已退出时，同source恢复命令为 `bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/5c8f8e4f5cd478c793a0e0d9640005deaf700973/scripts/launch_two_row_extension288_v1.sh 5c8f8e4f5cd478c793a0e0d9640005deaf700973 train32 resume`。只有前32全部闭合并读完整质量后，才独立执行train64。GPU1当前无研究训练占用；不重复运行probe。没有后台自主决策服务，当前活跃会话继续实施与分析。

## 最新执行状态 — 2026-10-02 23:56 UTC

本段覆盖历史 running 记录。no-direct12000 已完成，source1a3eef1、PID628749/628753于23:38:29UTC exit0；118项真实服务器测试全部通过。相同初始化共有张量、384000抽样链、1536000路径状态和48次选择核验通过，但删除549120参数是容量混杂。固定last TRAIN有效率91.93%高于原76.06%，DEV39.58%低于原43.06%；best52/144低原59/144，已分类类型求和31比30多1但已知参考覆盖更低。停止该结构扩展；全部保存池/12父图/失败和权重hash已归档，0额外forward分析。见 reports/observed_two_row_prefix76_no_direct_v1/NO_DIRECT_RESULTS.md。核心方法优势仍未成立。

MAIN269，原267字段保留。JOBS快照为2026-10-02T23:52:09.965049+00:00，520条；registry更新实际状态。下一步仅固定两臂原best，各36次真实Qwen/K4，补齐12000模型在线成本；3文件已独立复核，本次冻结后单独tests/preflight/两臂，尚未在线运行，不搬用1500时延。

extension288 source5c8f8e4的train32 fresh仍活跃（session20261002T231813Z_train32_624110，child624144/624145，CPU2/3）。23:54机械检查24/32闭合，不等于24成功场景，raw尚未全32分析。首批全部闭合后，从1a3eef1单独运行TRAIN32分析，读质量后再独立train64 resume。新DEV保持封存。composite108固定旧64TRAIN+新32TRAIN+旧12DEV导出实现已61本地测试和独立复核通过，尚未实际导出；旧225行原字节保留，失败父不替换。后续质量/缓存/同总12000曝光普通训练仍待实际数据门禁，不自动启动。

恢复先检查上述PID的实际命令及status，禁止重复fresh或重采闭合父。所有服务器操作只使用不可变release。当前活跃会话继续执行；没有会话结束后自动分析/改代码服务。

## 当前执行 — 2026-10-03 00:13 UTC

本段覆盖下方旧状态。source1fccf48的88项服务器测试通过，两臂各36次真实Qwen/K4已完成exit0，72份特征字节及288候选检查决定均与原best一致。完整请求中位constant/no-direct为74.5681/74.5213ms，P95为86.1414/91.2371ms；未显示可靠加速。原质量59/144对52/144保留。MAIN271，原269字段未改，新两行只记在线成本；见 reports/observed_two_row_12000_online_v1/ONLINE_RESULTS.md。

extension288首批train32已于00:02:21UTC结束，两shard exit0，原SSH89814结束。source1a3的完整TRAIN32分析于00:07:08UTC exit0：864已发槽、555accepted、309failed、247有效unknown、0已知重复，864严格状态恢复全部通过；无缺初始观测/blocked/跨父重复。96条件中20个已知类型数超过4，96全图正在独立视觉核验和归档。没有将提案失败隐藏或将unknown视无效。

source1fcc的composite108 readiness/export已独立完成，export于00:08:26UTC exit0；321输入=285TRAIN+36原DEV，1868正参考=1663TRAIN+205DEV。原225行字节保留，旧3缺失输入不替换。新DEV封存。下一步是新冻结pipeline的真实Torch测试、完整TRAIN表示/端点容量检查、仅新96条Qwen编码，然后同总12000步普通模型训练；所有阶段分别启动，当前尚未运行后四阶段，不把export成功称训练完成。pipeline已独立复核，本地22pass，3项Torch待服务器验证。

JOBS快照 2026-10-03T00:11:37.242427+00:00 共534条；registry实际同步。当前没有GPU训练或采集作业运行。待读完32质量后才单独启动train64 resume，使用原5c8f8e4固定collector，不重采前32。核心新方法优势仍未成立，数据扩大是普通基线控制。恢复需先核实际PID/status/immutable source；不重启上述已完成作业。

## 当前实际进展 — 2026-10-03 00:26 UTC

本段覆盖下方旧状态。首32新增TRAIN完整质量报告及全部96图/32front已核验、归档：reports/observed_two_row_extension288_quality_v1/TRAIN32_RESULTS.md。555正例/309失败、247unknown原样保留，未发现系统采集错误；仅窄±5mm同分布，不称复杂场景或OOD。主报告包括每槽失败、19色分布、长弧和全部hash。

composite source71cf0c1实际70测试0skip通过，1663/1663TRAIN正参考通过H24/端点容量；真实新增96条Qwen编码完成，旧225份NPZ（含36DEV）容器字节全部保持，三阶段cost来源已归档。见 reports/observed_two_row_composite108_preparation_v1/QUALITY_CACHE_RESULTS.md。cache外层21.775545秒/.0060487625GPUh，编码body11.501351秒含加载，嵌套不相加。没有过时LoRA条件缓存或新DEV数据进入这轮冻结Qwen控制。

root独立启动同总12000步普通模型训练：source71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c，PID652623/child652624，00:17:54.174724UTC开始，SSH64194，CPU1/GPU1/35%；00:24进度7750/12000、31/48原DEV选择。当前仍运行，不用中间结果作新主结论。输出 /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1；完成后必须单独fixed-last-train诊断，再读原189TRAIN公共子集/新增96TRAIN/固定36DEV配对。初始化/总1536000路径状态同原constant，数据人口及每输入曝光不同，不能称方法贡献。

另按已读质量独立启动train64 resume，仍为原5c8f8e4 collector，session20261003T001613Z_train64_651720，SSH57754，CPU2/3，00:16:13UTC开始；仅追加注册indices32..63，不重采已闭合前32。该阶段全64闭合前不读取新增raw结果，不自动启动train128。新DEV继续封存。服务器实际job路径/PID/命令以JOBS与registry为准，禁止根据旧running段重复启动。

MAIN271保持。JOBS快照 2026-10-03T00:25:15.049030+00:00 共549条，registry同步最新转移。尚无核心方法优势；当前活跃会话继续实际训练后的分析和研究决定，没有会话结束后自动思考服务。

## 实际完成与接续 — 2026-10-03 00:50 UTC

本段覆盖下方旧running状态。composite108普通12000已于00:27:47UTC exit0，固定last285 TRAIN诊断于00:28:48UTC exit0。source71cf0c1，70服务器测试通过。全部46服务器原件、18离线分析产物、12父全图和权重hash已核验；见 reports/OBSERVED_TWO_ROW_COMPOSITE108_BASELINE_RESULTS.md 及 reports/observed_two_row_composite108_v1/INDEX.md。此轮不重启。

原36 DEV best Tip59→63/144、已分类类型总和30→34、Any30/36不变；last62→56/144、Any28→27/36。公共189 TRAIN last575→643/756（76.06→85.05%），新增96为346/384（90.10%）。扩大数据改善训练拟合，但DEV收益不稳定；数据人口/颜色频率/逐输入曝光变化不是方法贡献。保留48次历史、全部失败和unknown。新训练base580.198116s/.1611661434GPUh，完整job593.489648s；诊断CPU19.117344s，嵌套成本不相加，没有新模型真实在线时延。MAIN273，旧271字段保留。

下一实际研究决定为同一composite last头的冻结Qwen/末两层LoRA普通基线配对。先实现TRAIN6技术门禁：官方完整forward与layer26冻结前缀缓存后的原尾层回放，两逻辑微步、两独立optimizer共4次更新；上限10full+10replay+4head/16候选状态，无DEV。此时实现已完成、独立审查中，尚未运行真实probe，更未启动正式LoRA续训。不会使用更新后过时的最终4096维条件缓存；只核验serial原长度，未证明batched padding。技术通过后另行冻结续训协议，不自动启动。

extension train64仍由原5c8f8e4 collector/session20261003T001613Z_train64_651720在CPU2/3执行，SSH57754；只追加注册32..63，前32不重采。全64闭合前只读机械进度，不读新增raw；之后独立TRAIN64分析，不能自动启train128或解封新DEV。所有reserved原始数据保持封存。恢复先核PID/status实际命令，禁止重复启动已发父。

JOBS实际快照 2026-10-03T00:47:54.427626+00:00，566条；registry已同步。当前没有GPU训练任务。核心方法优势及论文A–H尚未成立，当前活跃会话继续实施，不存在会话结束后自动分析改码服务。
