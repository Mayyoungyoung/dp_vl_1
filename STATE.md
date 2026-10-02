# State — 2026-10-02

- Branch: `codex/multiroute-v2`.
- Historical snapshot: `d0d97eb22ff1190f4f89d4d2fd7100db116c8a57`.
- Historical baseline: learned-query set regression TEST UniqueValid@3=3.000; historical OOD=2.878 mean across seeds 0/1/2. Controlled true geometry + frozen CLIP only.
- V2 controlled development seed0/K4: subset matching UniqueValid=1.7734, Valid=52.54%; all-positive matching=3.2891/89.65%; saturation-aware assignment=3.3984/100.00% (best step1500 of3000). These are stronger baselines, not a novelty or robotics claim.
- Completion seed0/1/2 finished: coverage-minus-attention mean additional valid types +0.16406 / +0.01563 / -0.04688. The initial advantage is not stable; retain max pooling as an ablation, not the paper contribution. Full evidence: reports/COMPLETION_THREE_SEED.md.
- Self-draft mixture completed: attention own2+2 UniqueValid2.5078→3.3047, coverage2.9531→3.2813. Both remain below same-checkpoint joint4 (3.3125/3.3438) and use two forwards. Keep joint4; known training repair is not a novel contribution.
- Genuine frozen Qwen revision89644892: 96 RGB-language features; RLBench-derived reach32 has32 parents,275 accepted paths/288 attempts, strict restored state/RGB. Four original task collection checks12/12; this is collection success, not learned execution.
- Observation evaluation v2 fixes omitted no-reference instructions: all24 DEV instructions for semantics,23 for reference ADE. All five old cached/online checkpoints re-evaluated; thresholds unchanged and original reports retained.
- RGB-D ordinary baseline and endpoint-attention auxiliary each completed three real seeds at identical1000 steps/128000 slots. Strict3cm semantic accuracy mean1.04%→28.82%; AnySemantic1.39%→31.94%; endpoint18.92→14.51cm; all three seeds improve. Conventional grounding repair,8 DEV parents only; full-path validity and core mechanism still unproved. See OBSERVATION_GROUNDING_THREE_SEED.md.
- Online frozen/LoRA pilot and matched warm-start finished. Eight LoRA tensors genuinely updated; warm-start both select common step0, no DEV improvement. Do not equate adapter updates with effective fine tuning.
- Obstacle four-parent rendered-state pilot completed:48 proposals,20 successful paths,17 classified types,48/48 exact restore. All four depth/physical surface audits pass within2.4mm after explicit render warmup. Preserve old failures. Formal128 obstacle and256 natural-layout parents are being collected using immutable source; see JOBS and agent job records for current state, not this prose alone.
- Next mechanism probe: localized route updates after closing one opening, with paired full-path completion under cumulative4+b budgets. Implementation in progress, no result yet. Natural-layout learning-curve snapshot prepared while collection continues.
- Resources: see RESOURCE_ENVELOPE.md. Keep sequential GPU1 queue.
- Git remote authorized by user: `git@github.com:Mayyoungyoung/dp_vl_1.git`; research branch `codex/multiroute-v2`. Immutable release trees and every training source commit are recorded with jobs. Verify actual remote head rather than trusting a stale prose commit.
- Follow current logs in RESEARCH_LOG.md and job records in JOBS.json. Final-paper core evidence incomplete.
