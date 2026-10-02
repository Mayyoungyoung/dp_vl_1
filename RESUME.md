# Resume

1. Read STATE.md, JOBS.json and the tail of RESEARCH_LOG.md. Inspect actual PIDs/logs before restarting any experiment.
2. `ssh wzy3090` → verified project `/home/wzy/dpvlm/route_set_v1`. GPU allocation and environment rules: RESOURCE_ENVELOPE.md.
3. Historical tests: `cd /home/wzy/dpvlm/route_set_v1 && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests -q`.
4. Historical dataset SHA256 must equal `34f4e6944f66c7e2716239830b22f0ad1885bf80a9f9dc6be606e7883d04374d`. Do not regenerate it when loading old checkpoints.
5. V2 job commands and resumable checkpoint paths will be appended as launched. No post-session autonomous research is implied by background training.
6. Local Git had no remote on audit. Push only when an existing authorized repository URL is supplied; preserve pending local commits.
