# State — 2026-10-02

- Branch: `codex/multiroute-v2`.
- Historical snapshot: `d0d97eb22ff1190f4f89d4d2fd7100db116c8a57`.
- Historical baseline: learned-query set regression TEST UniqueValid@3=3.000; historical OOD=2.878 mean across seeds 0/1/2. Controlled true geometry + frozen CLIP only.
- V2 controlled development seed0/K4: subset matching UniqueValid=1.7734, Valid=52.54%; all-positive matching=3.2891/89.65%; saturation-aware assignment=3.3984/100.00% (best step1500 of3000). These are stronger baselines, not a novelty or robotics claim.
- Round1/2 complete on server, with predictions and resumable optimizer/RNG checkpoints. Completion attention versus coverage is next; Qwen and RLBench environments/data being prepared independently.
- Resources: see RESOURCE_ENVELOPE.md. Keep sequential GPU1 queue.
- Git remote authorized by user: `git@github.com:Mayyoungyoung/dp_vl_1.git`; branch pushed and verified through `1a1b6bb`.
- Follow current logs in RESEARCH_LOG.md and job records in JOBS.json. Final-paper core evidence incomplete.
