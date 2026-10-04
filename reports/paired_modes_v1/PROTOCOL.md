# Paired scene mode learning — 2026-10-04

User authorizes implementation, experiments and evidence-driven development toward a paper prototype. New family: `paired_modes_v1`. Local branch remains `codex/multiroute-v2`. Preserve all historical data and outputs.

Core comparison: R0 ordinary M8+clearance; R1 additionally uses within-scene relation-balanced supervision; R2 additionally uses partial correspondence between witnessed surviving modes of paired scenes. All arms share paired observations, initial weights, input exposure and M8/H24. No pi objective. Fixed-last models. Single-seed screen then replication of promising comparisons; at most two evidence-driven method revisions before reconsidering the mechanism.

Data: 128 TRAIN and32 DEV_MODEL independent families, each open/closed/shifted. All goals/colors/canonical state shared within a family. New RGB-D comes from the existing RLBench renderer. New reference paths are explicitly **geometric upper-level teacher polylines**, not robot rollouts; raw and H24 continuous segment clearance and goal/event contracts are checked. This avoids making low-level execution the data bottleneck and applies equally to all arms. Old actual robot-trace training data remain reusable common evidence. Missing collection/teacher paths never prove infeasibility. Closed low-gap claims require geometric certificates; over routes remain separate.

Primary: distinct valid modes at8 with candidate validity/target failures and finite-reference coverage. Pair responses: shared-mode retention, certified closed-mode generation, witnessed newly opened-mode coverage. All requested development families remain in denominators. No reserved TEST_LOCKED access. Score/calibration data stay separate from generator training.

Resource: only ssh wzy3090, GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,35% CUDA cap,4 CPU threads total; 2 single-thread collection workers, no concurrent GPU training during collection. New data soft budget8GiB. Serial compute accounting limit24h before reassessment; this is a self-imposed operational bound, not a claim of measured time. Record every actual command, failed job, imported source hashes and artifact hashes. Freeze immutable commit exports before launch.

Recovery: retain one rolling recovery checkpoint every1000 steps and final checkpoint, each with model, optimizer, scheduler, all RNG, sampler state/order digest, config/data/source identity. No deletion of original/final/key evidence. New failures get distinct outputs; never overwrite a live launcher.
