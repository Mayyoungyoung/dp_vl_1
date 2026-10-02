# Three-seed obstacle grounding: both checkpoint protocols retained

All results below use the same new32 TRAIN snapshot, the same four repeatedly used DEV parents (12 instructions), K=4, three real training seeds, and the unchanged box-only tip evaluator. Each arm has 1,231,965 parameters and 128,000 candidate-trajectory exposures over 1,000 training steps. The two arms differ in soft versus straight-through peak surface anchoring. This is an architectural control, not a claimed new route-set mechanism or independent final test.

The original candidate-ADE best checkpoint is retained for every run. A separate sensitivity analysis evaluates step 1,000 for **both arms and all three seeds**. Fixed-step results do not replace the original selection results. Neither evaluator thresholds nor DEV examples were changed. Reported standard deviations are sample SD over three training seeds, not confidence intervals over the four parents.

| Original ADE-best, mean ± SD | Soft anchor | Peak anchor |
|---|---:|---:|
| Strict semantic candidates | 19.44 ± 4.81% | 28.47 ± 34.38% |
| TipValid candidates | 9.03 ± 1.20% | 20.83 ± 23.20% |
| AnyTipValid | 19.44 ± 4.81% | 27.78 ± 29.27% |
| Unique classified TipValid types | 0.250 ± 0.083 | 0.417 ± 0.464 |
| Known reference type coverage | 0.117 ± 0.029 | 0.250 ± 0.312 |

Peak seed 1 has **zero** semantic correctness, TipValid and unique TipValid under its original ADE-selected checkpoint. Its better late training behavior does not erase that failure. The original checkpoint protocol therefore produces a highly unstable peak-anchor result and does not establish a robust advantage.

| Fixed step 1,000, mean ± SD | Soft anchor | Peak anchor |
|---|---:|---:|
| Strict semantic candidates | 39.58 ± 5.51% | 72.22 ± 4.81% |
| TipClear candidates | 69.44 ± 6.36% | 70.83 ± 10.42% |
| TipValid candidates | 29.86 ± 3.18% | 52.08 ± 6.25% |
| AnyTipValid | 38.89 ± 4.81% | 66.67 ± 8.33% |
| Unique classified TipValid types | 0.500 ± 0.083 | 0.972 ± 0.096 |
| Duplicate classified TipValid count | 0.639 ± 0.096 | 0.944 ± 0.192 |
| Known reference type coverage | 0.217 ± 0.076 | 0.494 ± 0.092 |

Fixed-step peak improves semantic correctness and box-only valid coverage in every seed, while TipClear changes little on average and duplicate valid routes also increase. This supports continuing the observed grounding baseline and studying how better grounding affects route generation. It also exposes a mismatch between the existing ADE selection rule and the task checks. The sensitivity analysis was motivated by development behavior and must remain labeled as such; an untouched final evaluation requires a subsequently frozen protocol.

`TipValid` includes target semantics, start and event checks, and complete tip-segment checks against the recorded physical boxes. Other objects, complete arm geometry, IK, robot execution and score-based selection remain unverified; full-task validity is null. Unknown valid passage types remain valid but do not contribute invented uniqueness.

Evidence: `obstacle_original_best_evaluation/three_seed_summary.json` and its six per-scene evaluation folders retain the original protocol; `obstacle_fixed_step1000_evaluation/three_seed_summary.json` and its six folders retain fixed-step results. The four additional original-best evaluations ran in 0.96 CPU seconds and all exited 0; seed 0 reuses already completed identical evaluations with explicit provenance. The six fixed-step evaluations ran in 1.39 CPU seconds and all exited 0. Every call uses evaluator source `9c19288b0e2fb86b6bea42ac23758314134872c0` and records predictions hashes, PID, commands, logs and exit status.
