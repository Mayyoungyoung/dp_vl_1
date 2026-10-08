# Canonical control timeout and exact-state continuation

The original immutable queue14da6aa ran8 passing tests, then canonical_train
hit its registered180s timeout: exit124, wrapper180.323454s, no evaluation
launched. This is an execution-budget estimate failure, retained in the ledger.
The reduced reference tensor count does not remove the per-request Hungarian
group-matching overhead; the74s sampled arm used a different matching loop.
Do not hide the failed command or claim a5x wall-time speedup.

Preserve the existing recovery.pt in a new immutable result directory before
resuming. Record its actual step and SHA, then use the existing strict resume
path, which asserts identical settings and restores model/optimizer/scheduler,
Python/NumPy/Torch/CUDA RNG, input sampler, assignment RNG, exposure and history.
Use the same exact scripts/routeset/configs file bytes as14da6aa. The recovery
launcher is stored under research_v3 so it does not alter source_identity().
No hyperparameter, loss, planned1200 steps or selection gate changes. Only the
execution cap is amended to240s for the remaining steps, inside the remaining
1537.052819 command seconds. Wall-time reporting includes failed and resumed
commands; successful training summary alone excludes the interrupted prefix.

Frozen launcher research_v3/run_canonical_resume.sh runs resume, then the
previously registered DEV evaluation and analysis. Unique job IDs and coordinator
directory preserve the failed attempt. All optimizer steps reported relative
to planned1200; no partial result or launch is a successful training claim.
