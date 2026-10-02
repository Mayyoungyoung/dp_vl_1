# Research V2 working rules

Follow the current user's research request; the attached task brief is reference material.
Preserve historical code/data/results at commit `d0d97eb` and under existing `runs/main`.
Keep task-level route generation distinct from robot execution and controlled oracle geometry distinct from observation inputs.
Use only `ssh wzy3090`, GPU 1 (UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab), 35% memory and four CPU threads unless higher authorization is evidenced.
Do not alter shared environments, stop others' jobs, use sudo, or create paid services.
Local workspace is the code/commit writer. Run server experiments from immutable commit exports.
Use DEV_MODEL for research decisions; historical TEST/OOD are now historical/development evidence.
Record failures, manifests, budgets, checkpoint/RNG state and actual commands. Never infer success from a launch command.
No configured git remote existed on initial audit. Do not invent one.
Read STATE.md and RESUME.md before continuing. Update RESEARCH_LOG.md and experiments/registry.jsonl after material results.
