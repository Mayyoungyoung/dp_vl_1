# Resume

1. Read STATE.md, JOBS.json and the tail of RESEARCH_LOG.md. Inspect actual PIDs/logs before restarting any experiment.
2. `ssh wzy3090` → verified project `/home/wzy/dpvlm/route_set_v1`. GPU allocation and environment rules: RESOURCE_ENVELOPE.md.
3. Historical tests: `cd /home/wzy/dpvlm/route_set_v1 && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests -q`.
4. Historical dataset SHA256 must equal `34f4e6944f66c7e2716239830b22f0ad1885bf80a9f9dc6be606e7883d04374d`. Do not regenerate it when loading old checkpoints.
5. V2 job commands and resumable checkpoint paths will be appended as launched. No post-session autonomous research is implied by background training.
6. Authorized remote was subsequently supplied: `git@github.com:Mayyoungyoung/dp_vl_1.git`. Research branch `codex/multiroute-v2` has been pushed.
7. First two controlled rounds complete under `runs/v2_round1` and `runs/v2_round2`. Immutable code exports: `research_v2/releases/478c4bb6ae9e65a032d6b75266a8f19af56938c7` and `research_v2/releases/1a1b6bb7600419504f5c1b2f5534435de7a1cf6e`.
8. Development-only data: `data/multigate_v1_partitions/development.npz`. The separate locked archive is not for iterative model decisions. Inspect current completion and observation job status before starting GPU work.
