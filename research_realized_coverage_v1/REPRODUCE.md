# Reproduction (active research; final artifact audit pending)

Local code writer F:/dpvlm, branch codex/realized-coverage-v1. Remote
git@github.com:Mayyoungyoung/dp_vl_1.git; never force push. Server ssh wzy3090,
root /home/wzy/dpvlm/route_set_v1, existing .venv/bin/python. Only GPU1 UUID
GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,0.35memory,four threads/CPU0–3.
No environments changed. New user authorization removes old total-time caps;
new costs live in runs/realized_coverage_v1/jobs, old ledgers remain untouched.

## Inputs

Fixed C0: runs/mode_geometry_v1/canonical_C_seed0/last.pt,
SHA2566090df425d0ac237f275620f5de5e9453ce8e5c9638fbda1afed2e39c4ea6718.
Complete frozen q: original scorer bundle SHA256
2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e.
All1152TRAIN and288DEV reuse the paired export. All56920TRAIN positives reuse
verified_edit_all_modes_support_v1/support.npz. See previous study REPRODUCE.md
for exact unchanged dataset/cache/support hashes. Never use raw collector directory
or TEST_LOCKED. The128TRAIN/32DEV layout families retain their historical roles.

## Immutable launch

Commit locally; export `git archive` containing routeset,scripts,configs,tests,
research_realized_coverage_v1. Copy to server research_v2/incoming/source_SHA.tar,
extract into a NEW research_v2/releases/FULL_SHA directory. Launch its frozen
scripts/launch_realized_coverage_v1.sh with a fresh --id and actual command.
Receipts bind all imported research code, policy, commands, source commit, PID,
timings and exit status. Do not edit a running export or rerun into old outputs.

```bash
bash scripts/launch_realized_coverage_v1.sh --id NEW_ID -- \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  -m research_realized_coverage_v1.feedback --name NEW_FEEDBACK --split TRAIN
```

Collector --checkpoint binds another generator; --proposal binds the query policy.
Per-request NPZ contains queries/variants, all paths/events, q, checker flags,
official valid words, auxiliary raw signatures, selected indices, utility changes,
added/lost words and observed context. Manifest/summary bind every file to a
snapshot and role. --resume reuses completed files, checks input snapshot/design,
and records resumed source; it never treats an unfinished summary as success.

## Training and evaluation

`train_allocation --kind success|net|dense|no_peer|added_only --name NEW_NAME
--feedback POOL_A,POOL_B --seed S --steps 2400` trains only TRAIN outcomes.
All paired head runs use the exact same feedback minibatch stream. Checkpoints
at400/1200/2400 preserve curves; formal paired comparison locks2400final.
Ordinary success is averaged realization probability, not physical feasibility.

`train_geometry --arm ordinary|kl_only|gap|hard --name NEW_NAME --steps 3600
--seed S` continues C with no displacement. Gap/hard require --head and --feedback
bound to the initial generator. Curves600/1800/3600 remain saved. First screen
shares budgets, not exact request RNG across branch-specific samplers; report this
limitation. New decoder versions require new feedback and bound head training.

`evaluate --name NEW_EVAL --checkpoint GENERATOR --head HEAD_RELATIVE_TO_RUN
--start-head OPTIONAL_SUCCESS_HEAD` generates exactly eight routes once, then
uses complete frozen q to return four. A hook asserts one trajectory-output call
per request. No verifier/reference enters proposal or decoding. Fixed288DEV only.
Saved pool and row metrics include actual words, requested words and returned slots.

`analyze` reuses original2169/270/255witness definitions. `statistics` reports
exactly five paired seeds and family/crossed seed-family intervals. `summarize`
lists every finished evaluation and real job cost. `audit` binds all files,
feedback/checkpoint versions and immutable source exports.

## Recovery and cost

Both trainers save model, optimizer, RNG, step, actual sample-stream digest,
settings/history and source binding. Use same planned total --steps and --resume
after --stop-after N; completed last.pt outputs are protected. Actual100vs50+50
checks are provided by scripts/run_realized_checks_v1.sh.

Reported runtime is cached-feature research runtime, not full Qwen inference.
Feedback includes tens of thousands of eight-route counterfactual sets; they are
training/diagnostic cost, never hidden within K=8deployment metrics. Initial
collector3bc/723performed one extra discarded8-route decode per request; include
those1440sets as cost. Later deployment/collector obtains base queries from logits.
No final-test or real-robot success claims are made.
