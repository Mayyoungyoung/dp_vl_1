# Obstacle new32: matched-data heads and prototype localization

The 32-parent ordinary and positive-grounding geometry heads both completed their original 1,000-step training, seed 0, using genuine frozen-Qwen features and current RGB-D/camera/state. Each has 1,231,965 parameters and 128,000 trajectory exposures. Both selected step 500 under the unchanged DEV candidate-ADE rule. Evaluation includes every candidate for every one of the same four repeatedly reused DEV parents (12 instructions); these are development results, not final held-out confirmation.

The ordinary head took 53.23 training seconds and the auxiliary head 54.27 seconds. The exact source is `9c19288b0e2fb86b6bea42ac23758314134872c0`. The cache covers all 108 observations, including zero-reference TRAIN instructions. The 96 TRAIN instructions include 94 eligible positive-reference examples and two zero-reference examples that remain in evaluation and data accounting.

## Unchanged box-only evaluation

`TipValid` combines the original strict nearest-target identity and 3 cm endpoint criterion, measured-current start, reach event sequence, and complete line-segment checks against the recorded boxes with 2 cm clearance. It does not verify other environment bodies, complete robot geometry, IK or execution. Full-task and selected validity remain null.

| Fixed DEV metric | new16 plain | new16 auxiliary | new32 plain | new32 auxiliary |
|---|---:|---:|---:|---:|
| Strict semantic candidates | 1/48 | 4/48 | 4/48 | 8/48 |
| Any semantic-correct candidate | 1/12 | 1/12 | 1/12 | 2/12 |
| Mean target-center error (m) | 0.1510 | 0.1637 | 0.1660 | 0.1373 |
| TipClear candidates | 30/48 | 24/48 | 5/48 | 17/48 |
| TipValid candidates | 1/48 | 2/48 | 0/48 | 4/48 |
| AnyTipValid | 1/12 | 1/12 | 0/12 | 2/12 |
| Mean unique classified TipValid types | 0.0833 | 0.1667 | 0 | 0.1667 |
| Mean duplicate classified TipValid count | 0 | 0 | 0 | 0.1667 |
| Mean known reference type coverage | 0.0500 | 0.1000 | 0 | 0.1000 |
| Reach event sequence correct | 48/48 | 48/48 | 48/48 | 48/48 |

All new32 auxiliary semantic-correct candidates are concentrated in two instructions: `271100_target2` has one TipValid candidate; `271101_target2` has three TipValid candidates of the same classified route type (two duplicates). The other ten instructions have no semantic-correct candidate. The plain head's four semantic-correct candidates all belong to `271102_target0` and all fail box clearance. There are no valid unknown-type candidates in either head.

Thus adding TRAIN parents improves the auxiliary head's endpoint/AnyTipValid result, but does not improve the mean unique valid type count or known reference coverage over new16. The ordinary head loses substantial clearance performance. This is not evidence that dataset expansion alone established a useful route-set method. Both dataset sizes used the same total training exposure; the larger dataset therefore has fewer average revisits per parent.

On TRAIN, new32 plain/auxiliary TipValid is 3/384 versus 19/384, and AnyTipValid is 3/96 versus 11/96. Their strict semantic rates are 8/384 and 44/384. The fixed evaluator retains every failure; no geometry repair or candidate replacement was applied.

## Matched-data target-localization control

The unchanged RGB-D prototype is fitted on the same new32 TRAIN observations and positive endpoints and achieves 12/12 strict DEV endpoint correctness at K=1, with no abstentions. It uses an exact closed instruction and emits an observed surface point. It is a target-localization control, not a route generator, and cannot be assigned `TipValid`, unique-route or execution metrics. Its inference median is 12.86 ms including RGB-D loading and processing, with no Qwen pass.

The new32 auxiliary's learned surface anchor itself is correct on only 1/12 DEV instructions, with mean center error 0.1404 m. Its final endpoints are on average 0.0190 m from that anchor. The ordinary anchor has 0/12 correctness and 0.1929 m mean center error. These facts continue to locate a major bottleneck in learned grounding. The separate hard-anchor experiment and observation-only traditional four-route baseline are the next concrete controls; no new32 DEV threshold or selected checkpoint is changed here.

## Reproducible artifacts

- `obstacle_new32_evaluation/tip_evaluation_v1/{plain,auxiliary}/{dev_model,train}/`: unchanged evaluator metrics and all per-scene/per-candidate records.
- `obstacle_new32_evaluation/tip_evaluation_v1/all_dev_routes_{xy,xz}.png`: all 12 DEV examples and all candidates; both figures were visually checked. Projections are not collision evidence.
- `obstacle_new32_evaluation/model_runs/{plain_seed0,aux_seed0}/`: exact configuration, training history and summary.
- `obstacle_new32_evaluation/binary_artifact_index.json`: 12 actual copied checkpoint/prediction binaries totaling 59,992,512 bytes, server paths, ignored local paths and SHA256. Best-checkpoint and DEV-prediction hashes match the original training summaries. Binary data is not committed to ordinary Git.
- `obstacle_new32_evaluation/tip_evaluation_v1/execution.json`: CPU1 evaluation PID 221742 / child 221743, 7.31 seconds, exit 0, immutable source and exact command.

Best checkpoint hashes: plain `7bc1301712ebbcd0af7367f3e2b607926101877e3bb187f930d09a62691812dc`; auxiliary `ec64f31b04aecfb9f47d8b99827d8eac9f7e02849d194781be350cad12294e64`.

## Subsequent hard-anchor control: original selected checkpoint

The same-data straight-through peak-anchor control from source `2dc026b8a34fecbb3688d26ff544165835ebd938` also completed 1,000 steps. Its original ADE rule selected step 750. Evaluating those saved best-checkpoint outputs with the unchanged `9c19288` tip evaluator gives the following DEV result, without replacing them by a later checkpoint:

| DEV measure | Soft auxiliary best500 | Peak-anchor best750 |
|---|---:|---:|
| Strict semantic candidates | 8/48 | 9/48 |
| Any semantic-correct candidate | 2/12 | 3/12 |
| Mean target-center error (m) | 0.1373 | 0.1030 |
| TipClear candidates | 17/48 | 40/48 |
| TipValid candidates | 4/48 | 8/48 |
| AnyTipValid | 2/12 | 3/12 |
| Mean unique classified TipValid types | 0.1667 | 0.3333 |
| Mean duplicate classified TipValid count | 0.1667 | 0.3333 |
| Mean known reference coverage | 0.1000 | 0.1500 |

The hard-anchor TRAIN result is 84/384 TipValid candidates and 41/96 AnyTipValid instructions. This early single-seed comparison suggests the grounding choice also affects route clearance, while duplication remains. It is an architectural baseline control, not an established novel route-set mechanism. Later fixed-step, multi-seed results must be reported alongside the original selected-checkpoint results rather than silently changing the selection rule.

All original-best hard results are in `obstacle_new32_evaluation/hard_tip_evaluation_v1/{dev_model,train}/`, with per-scene/candidate data. The CPU evaluation exited 0 in 0.55 seconds (PID 232473); the execution record identifies both the evaluator source and model-training source. Full-robot validity and selected success remain unverified/null.
