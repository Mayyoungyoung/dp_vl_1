# Obstacle new16: fixed evaluation and generation diagnosis

This is a single-seed development comparison, not an independent final result. The four DEV parents have already been used to diagnose collection and model behavior. Both models use the same real frozen-Qwen features, current state, RGB-D/camera observations, K=4, H=24, parameter count (1,231,965), and 1,000-step training exposure. The auxiliary model adds the already-declared positive-endpoint attention loss (weight 0.02, sigma 0.025 m); no target coordinate enters forward inference.

The original DEV candidate-ADE checkpoint rule is preserved: plain selected step 750, auxiliary selected step 250. No evaluation threshold, candidate budget, or checkpoint choice was changed after observing these results.

## Fixed results

All 12 DEV instructions and all four generated candidates per instruction are included. `TipValid` includes the original strict nearest-target identity and 3 cm endpoint check, fixed current-start and event checks, and complete tip-segment versus the recorded box AABB checks with the original 2 cm clearance. It does not check the complete robot, inverse kinematics, table/walls/other objects, or simulator execution. Full-task `Valid`, `AnyValid`, `UniqueValid`, and selected execution validity remain null.

| DEV measure | Plain | Positive grounding auxiliary |
|---|---:|---:|
| Strict semantic target correctness, per candidate | 1/48 | 4/48 |
| Mean target-center endpoint error (m) | 0.1510 | 0.1637 |
| TipClear, per candidate | 30/48 | 24/48 |
| Event sequence correct, per candidate | 48/48 | 48/48 |
| TipValid, per candidate | 1/48 | 2/48 |
| AnyTipValid, per instruction | 1/12 | 1/12 |
| Mean unique classified TipValid types | 0.0833 | 0.1667 |
| Mean known reference type coverage | 0.0500 | 0.1000 |

All four auxiliary semantic-correct candidates belong to `obstacle_reach_271100_target2`, of which two pass the box-only clearance test. The other 11 auxiliary instructions have no semantic-correct candidate. The plain model's sole semantic-correct candidate belongs to `obstacle_reach_271102_target0` and passes the box-only check. These results do not demonstrate a broad route-set advantage.

Independent gate failures are 47/48 versus 44/48 semantic failures, and 18/48 versus 24/48 box-clearance failures; failure categories can overlap. No candidate is repaired, removed, or replaced for evaluation.

## What the observed failures imply

Endpoint grounding remains the dominant obstacle. The auxiliary DEV learned surface anchor itself has 1/12 strict semantic correctness and 0.1641 m mean target error. The generated endpoint stays only 0.0061 m from that anchor on average (maximum 0.0088 m). The evidence does not support blaming a large residual for moving an otherwise correct anchor away from the goal.

The auxiliary model also shows a checkpoint-selection tradeoff: strict semantic candidate accuracy at steps 250/500/750/1000 was 0.0833/0/0.2708/0.2292, while candidate ADE selected step 250. This is recorded as a diagnostic observation; the reported checkpoint is not retrospectively changed.

The route head uses its learned target anchor to construct every point on an initial straight reference line. Thus the interior-path objective can send gradients through both that line and the shared geometric context. A minimal next hypothesis is to detach only the anchor used by the interior reference line, keeping endpoint and attention-grounding learning active. This is a hypothesis, not an implemented improvement or established contribution. A predeclared TRAIN-batch gradient decomposition at the unchanged step-250 and step-1000 checkpoints will test whether shared-geometry gradients actually oppose each other. Grounding gradients must be measured on the attention-producing parameters: zero derivative with respect to a downstream anchor tensor would not imply absent attention learning.

## Reference support and limits

The TRAIN snapshot includes the first 16 requested parent scenes: 48 instructions, 47 with at least one positive reference, and 90 reference trajectories. The zero-reference instruction remains in the dataset and evaluation denominator; it is excluded from positive-only route regression. Nineteen of the 48 instructions have at least two known passage types. Of the 90 successful references, 35 have unknown passage type and remain valid positive demonstrations; unknown is not interpreted as a new type.

The 55 classified TRAIN references contain 24 negative-x, 27 positive-y, and 4 positive-x passages, with no sampled negative-y type. The DEV set has 20 references, of which 17 are classified; six of its 12 instructions have at least two known types. These incomplete, imbalanced reference sets do not establish the number of all feasible routes.

## Artifacts and reproduction

- `obstacle_new16_evaluation/paired_analysis_v1/paired_comparison.json`: all fixed metrics, per-scene changes, independent failure gates, and input/configuration checks.
- `obstacle_new16_evaluation/paired_analysis_v1/reference_type_audit.json`: all reference counts and known/unknown type frequencies.
- `obstacle_new16_evaluation/paired_analysis_v1/all_dev_routes_xy.png` and `all_dev_routes_xz.png`: every DEV parent and target, every candidate, and reference paths. Projections are visual aids; the numeric checker uses complete 3D segments.
- `obstacle_new16_evaluation/paired_analysis_v1/execution.json`: immutable source release `7474782f903ed104fd590b6bc3181da1e3675175`, script hash, exact command, exit 0, and 7.44 CPU seconds.
- `scripts/compare_observed_obstacle_runs.py`: fixed evaluator and exhaustive paired plotting; `scripts/diagnose_observed_anchor_gradients.py`: pending actual gradient diagnostic at the time this note was created.

The comparison command, run from the recorded immutable release, is:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/wzy/dpvlm/route_set_v1/research_v2/releases/7474782f903ed104fd590b6bc3181da1e3675175 \
/home/wzy/dpvlm/route_set_v1/.venv/bin/python \
/home/wzy/dpvlm/route_set_v1/research_v2/releases/7474782f903ed104fd590b6bc3181da1e3675175/scripts/compare_observed_obstacle_runs.py \
--data /home/wzy/dpvlm/route_set_v1/data/obstacle_learning_curve_new16_v1 \
--plain /home/wzy/dpvlm/route_set_v1/runs/observed_obstacle_new16_v1/plain_seed0 \
--aux /home/wzy/dpvlm/route_set_v1/runs/observed_obstacle_new16_v1/aux_seed0 \
--output /home/wzy/dpvlm/route_set_v1/runs/observed_obstacle_new16_v1/paired_analysis_v1
```
