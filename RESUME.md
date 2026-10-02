# Resume

1. Read STATE.md, JOBS.json and the tail of RESEARCH_LOG.md. Inspect actual PIDs/logs before restarting any experiment.
2. `ssh wzy3090` → verified project `/home/wzy/dpvlm/route_set_v1`. GPU allocation and environment rules: RESOURCE_ENVELOPE.md.
3. Historical tests: `cd /home/wzy/dpvlm/route_set_v1 && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests -q`.
4. Historical dataset SHA256 must equal `34f4e6944f66c7e2716239830b22f0ad1885bf80a9f9dc6be606e7883d04374d`. Do not regenerate it when loading old checkpoints.
5. V2 job commands and resumable checkpoint paths will be appended as launched. No post-session autonomous research is implied by background training.
6. Authorized remote was subsequently supplied: `git@github.com:Mayyoungyoung/dp_vl_1.git`. Research branch `codex/multiroute-v2` has been pushed.
7. First two controlled rounds complete under `runs/v2_round1` and `runs/v2_round2`. Immutable code exports: `research_v2/releases/478c4bb6ae9e65a032d6b75266a8f19af56938c7` and `research_v2/releases/1a1b6bb7600419504f5c1b2f5534435de7a1cf6e`.
8. Development-only data: `data/multigate_v1_partitions/development.npz`. The separate locked archive is not for iterative model decisions. Inspect current completion and observation job status before starting GPU work.
9. All reference-context completion seeds finished under `runs/v2_completion/seed{0,1,2}` on server. Read reports/COMPLETION_THREE_SEED.md before making claims; seed2 reverses the initial positive result.
10. The draft-shift repair runs in `runs/v2_completion_selfdraft/seed0/{attention,coverage}`. Read selfdraft_seed0.status.json and launcher.log before any restart. Trainer --resume is supported; the wrapper's per-job command is the authoritative exact command.
11. Real observation source: `data/observation_derived_reach32_20261002/{observations.jsonl,supervision.jsonl,qwen_cache}`. Separate `.venv-qwen` is required for live Qwen; cached head uses existing `.venv`. Model `data/qwen3-vl-2b-instruct-89644892` and official provenance must be preserved.
12. Frozen head checkpoint: `runs/observed_frozen_v1/seed0/best.pt`; actual online paired runs: `runs/observed_online_v1/{frozen_seed0,lora_seed0}`. LoRA checkpoints contain adapters/head only. No cached hidden features may be used after changing LoRA.
13. Read reports/OBSERVATION_READINESS.md and observation_head_diagnostic before expanding collection/training. The32-parent collection completed but its old wrapper exited1; do not erase this failure or restart it blindly. New collectors use immutable source+launchers.
14. `scripts/snapshot_jobs.py --root /home/wzy/dpvlm/route_set_v1 --output <snapshot.json>` inventories recorded jobs read-only. A stale `running` field is not evidence that a PID is alive. Do not stop or restart unrelated processes.
