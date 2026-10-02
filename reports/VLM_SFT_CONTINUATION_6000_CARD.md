# Bounded SFT exposure repair: original1500 → total6000

Status: actual bounded continuation RUNNING since 2026-10-02T16:52:48.306988UTC, record PID440253/child440254, SSH session95424, GPU1/35% and CPU1. Root approved after the genuineK2 GPU job exited. The total limit remains6000 and the original1500 run is unchanged. Source5bb9087c9ca511c2a68a4da08798e95e8c6d4cdc passed13 actual related SFT tests in2.93s, including the complete tiny continuation/resume equivalence below. This is stronger baseline training, separate from the paper's core mechanism. Greedy TRAIN8 reduced endpoint error but still scored0/8; the prior coordinate-token audit and low exposure motivate testing additional optimization before changing serialization. No endpoint-first sequence, beam width, temperature or random-seed sweep is included.

## Immutable source and exact training plan

Continue the original completed run `/home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_v1/seed0/last.pt`, SHA `d604b5a213bdf281e7976b460b7ceb2fc428488610b1b84670ddca04711cb00c`, from step1500 to total6000 in fresh output `runs/vlm_route_sft_continuation_v1/seed0`. Source configuration/checkpoints/journal remain unchanged. The new explicit `--continue-from` option is distinct from ordinary strict `--resume`; changing the original run's step limit through resume remains rejected.

Keep the same96 TRAIN parents/8 DEV parents, RGB/depth serialization, pinned Qwen revision/processor, last-two-layer q/v LoRA, AdamW learning rate1e-4/weight decay0, constant scheduler, batch1, gradient clipping1, and odd K1/even K4 sampling. Parent→language and positive-reference RNG streams continue without reseeding their state. DEV uses the same independently fixed seed200000 plan, weighted token-NLL selection every250 steps; no extra DEV evaluation is inserted at step1500 on restoration. Original teacher-forced best remains eligible. DEV token NLL remains a selection statistic, not generated route quality.

Only total planned steps and the registered continuation trainer/helper source receipts may change. Other original source helpers, data fingerprint, dependencies, optimizer settings, model, sampling, chunk size and selection cadence must match. Source metadata and every original small artifact are hashed. The source must be a completed declared run with a fully checkpointed journal; an interrupted original run must first use its original strict resume workflow.

## Recovery and accounting

Restore adapters, optimizer, scheduler, global RNG, target RNG, parent sampler, global step, history, selected best, gradient audit and complete request journal. The fresh directory copies only small adapter/checkpoint/journal/manifest artifacts, never base weights. An initial recovery checkpoint is written in the new tree before any added training step, without a model forward, DEV pass or RNG draw. Interrupted continuation then uses ordinary exact-config `--resume` in that new tree. Any successfully journaled but uncheckpointed replay remains charged by the existing ledger.

Keep `inherited_best.pt` byte-identical. Until a new NLL winner exists, `best.pt` also retains the old checkpoint bytes/config; it must not be relabeled as a newly trained checkpoint. The summary names `best_is_inherited` and `selected_checkpoint_origin`. If no new best appears, point diagnostics to the original run and reuse the already measured greedy result. For a newly trained best, a later separately reviewed inference entry must validate its continuation lineage and dynamic summary hash; the old fixed-SHA greedy guard is not weakened.

Track cumulative and added training requests/route slots/tokens/elapsed separately. Parent cost is the completed original summary's556.7868s, rather than the slightly earlier checkpoint-write timestamp. Added cost is cumulative minus that fixed inherited cost; do not add parent cost twice in research tables. Preserve the inherited original gradient audit, and independently audit nonzero gradients and actual adapter changes relative to the restored1500-step parameters for this added segment.

| Budget | Original | Added planned | Total planned |
|---|---:|---:|---:|
| Optimizer steps / TRAIN requests | 1500 | 4500 | 6000 |
| TRAIN route slots | 3750 | 11250 | 15000 |
| Fixed-plan DEV passes | 7 | 18 | 25 |
| DEV requests | 322 | 828 | 1150 |
| DEV route slots | 805 | 2070 | 2875 |

Actual token counts depend on sampled routes and must be measured. Replayed work can exceed planned counts and remains charged. Linear extrapolation of original556.7868s/1500 estimates roughly1670.36s additional (~27.84min,0.464 reserved GPU-hours), including the original run's mixture of training/DEV/startup overhead. This is an estimate, not a measured new budget; final added/cumulative elapsed and GPU-hour receipts are authoritative. GPU1/35%,CPU1; root owns queue order. No automatic extension beyond6000, and no competing GPU job is launched.

## Required validation and stopping decision

Pure tests reject altered data/LR/cadence/helpers, missing source receipts and erased inherited cost. The real shared tiny causal loop showed continuous6 steps equals completed2→fresh continuation4→strict resume6 exactly for adapters, optimizer, scheduler, all RNG/sampler states, history, best selection and exposure. It also verified original file hashes unchanged, copied inherited best unchanged, correct [1,4,1,4,1,4] exposure and real added-segment adapter updates. All13 related CPU tests passed from immutable5bb9087; three existing Torch CPU-autocast deprecation warnings are preserved in the log. This proves the tested recovery implementation, not real-Qwen training quality. CPU record PID438152/child438153 exited0 at2026-10-02T16:49:05.940819UTC; the worker was released. Full command/source/log/status receipts are under `vlm_route_sft_continuation_v1/cpu_validation/`.

The separate CPU launcher SHA is `dce92c192c0585cfb6b35fb833f16511f07328d38d74a2b8365f928746ff458f`. The reviewed, actually launched GPU launcher SHA is `c7655d2df3e6772f3c7dccbb4705813b48eaff04d3a73270f23bbcab1323be75`; it passed shell syntax checking and was launched by absolute path from the fixed5bb release. The immutable launch/source/initial running-status receipts are under `vlm_route_sft_continuation_v1/gpu_launch/`; this captured status is not a completion receipt. It requires the same immutable commit as the successful CPU record and rechecks all imported/test source hashes and the original1500 checkpoint. Its frozen recovery driver switches `--continue-from` to `--resume`, so the recorded resume command never incorrectly combines mutually exclusive options.

At6000 stop this run regardless of last-point trend. Inspect coordinate/token loss and the actual training curve, then run at most one fixed TRAIN8 greedy diagnostic for a genuinely new selected checkpoint under separately frozen lineage verification. The same predeclared gate (at least4/8 target endpoints within3cm and all-eight mean<=15cm) governs whether any larger DEV generation is worth proposing; it is not a changed benchmark threshold. If still poor, preserve the result and reconsider representation/data evidence once, not an indefinite exposure or decoding search. Any successful baseline repair must remain distinct from a new finite-budget route-set mechanism.
