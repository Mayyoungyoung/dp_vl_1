# State — 2026-10-02

- Branch: `codex/multiroute-v2`.
- Historical snapshot: `d0d97eb22ff1190f4f89d4d2fd7100db116c8a57`.
- Historical baseline: learned-query set regression TEST UniqueValid@3=3.000; historical OOD=2.878 mean across seeds 0/1/2. Controlled true geometry + frozen CLIP only.
- V2 controlled development seed0/K4: subset matching UniqueValid=1.7734, Valid=52.54%; all-positive matching=3.2891/89.65%; saturation-aware assignment=3.3984/100.00% (best step1500 of3000). These are stronger baselines, not a novelty or robotics claim.
- Completion seed0/1/2 finished: coverage-minus-attention mean additional valid types +0.16406 / +0.01563 / -0.04688. The initial advantage is not stable; retain max pooling as an ablation, not the paper contribution. Full evidence: reports/COMPLETION_THREE_SEED.md.
- Own-draft K4 rollout underperforms joint4. A paired late self-draft mixture is actually training in runs/v2_completion_selfdraft from release3b9d3b7 (2500 steps each, seed0, GPU1, 2 threads).
- Genuine frozen Qwen revision89644892: 96 RGB-language features; RLBench-derived reach32 has32 parents,275 accepted paths/288 attempts, strict restored state/RGB. Four original task collection checks12/12; this is collection success, not learned execution.
- Genuine Qwen route head1000 steps completed: DEV endpoint19.54cm, strict3cm semantic accuracy0. Camera/depth/label audit finds language target identity signal but poor metric localization. RGB-D geometry baseline is now being implemented.
- Online frozen/LoRA matched300x4 training finished. Eight LoRA tensors genuinely updated; both strict semantic0 and endpoint~24.98cm. This short pilot does not establish an adequately trained LoRA baseline. A matched warm-start comparison is being prepared.
- Obstacle RLBench-derived collector is undergoing strict1-parent validation; failed setup attempts are retained. No learned obstacle-route result yet.
- Resources: see RESOURCE_ENVELOPE.md. Keep sequential GPU1 queue.
- Git remote authorized by user: `git@github.com:Mayyoungyoung/dp_vl_1.git`; research branch pushed through `3b9d3b7`. Immutable release trees and every training source commit are recorded with jobs.
- Follow current logs in RESEARCH_LOG.md and job records in JOBS.json. Final-paper core evidence incomplete.
