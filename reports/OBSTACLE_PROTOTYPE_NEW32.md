# Obstacle new32: fixed prototype endpoint control

The existing observation-only color prototype baseline, unchanged from immutable release `9c19288b0e2fb86b6bea42ac23758314134872c0`, reaches **12/12 strict semantic target correctness** on the same four repeatedly used DEV parents. No hyperparameter or threshold was adjusted for this obstacle dataset. This is a closed-instruction, K=1 endpoint localization result; it generates no route and establishes no collision, full-robot, or execution validity.

Training uses only the first 32 requested TRAIN parents, their observed RGB-D/cameras and positive demonstration endpoints: 94 eligible instructions, with the two zero-reference instructions retained and explicitly skipped for supervised fitting. DEV target centers are first accessed after prediction, solely for the original nearest-identity plus 3 cm evaluation rule. No segmentation masks, obstacle truth, or DEV endpoint labels enter fitting or prediction.

| Fixed DEV measure | Result |
|---|---:|
| Semantic target correctness | 12/12 |
| Abstentions / unknown exact instructions | 0 / 0 |
| Mean goal-center endpoint error | 0.02551 m |
| Mean positive-reference endpoint error | 0.02581 m |
| CPU request median / p95 | 12.86 / 14.67 ms |
| Same-image changed-language correctness | 24/24 directed pairs |
| Stale-language correctness for changed goal | 0/24 directed pairs |

All individual goal-center errors lie between 0.02530 and 0.02570 m. The prediction is an observed object surface point; neither the 3 cm rule nor the prediction has been shifted to remove the remaining surface-to-center difference. Timing includes RGB/depth loading, backprojection, segmentation, component scoring and endpoint selection, and excludes Qwen and robot execution because this baseline uses neither.

The result shows that these visible goals can be localized from the declared observations within this narrow closed-instruction setting. It provides a strong baseline for the neural head's target-localization bottleneck. The earlier neural obstacle result used 16 TRAIN parents, so its numbers must not be presented as a matched-data comparison against this 32-parent prototype. The prepared new32 real-Qwen cache and ordinary/auxiliary geometry-head training will supply that matched-data comparison.

Artifacts are in `observation_prototype_obstacle_new32_v1/`: `report.json` includes all per-scene predictions, training inclusion/exclusion rows, components, source hashes, and unchanged configuration; `execution.json` records the exact command, PID 208312 / child 208313, CPU1 wall time 2.016 s, and exit 0. No GPU was used.

`scripts/launch_obstacle_new32_pair.sh` prepares the next sequence: genuine frozen-Qwen cache, then ordinary and positive-grounding auxiliary heads, each seed 0, K=4, H=24, 1,000 steps, identical exposure and original ADE checkpoint selection. It pins the same immutable source, uses GPU1 with the existing 35% memory limit and CPU1, checks every cached sample/hash, records per-stage recovery commands, and stops on failure. It has been prepared only; the research lead owns GPU scheduling.
