# Reproduction: completed experiments and preserved artifacts

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

## Actual completed evidence

Frozen source f2f3e2414f6e9aa229b281effa215e479368dd1b completed the final 18-test
suite, corrected-scope FIVE_SEED_FINAL.json and combined RESULTS.json/tables.
An initial archive verifier mistakenly indexed its transient active.lock; local
verification caught that after the completed job removed its lock. Source
4b3e128 excludes lock files and writes a NEW ARTIFACT_AUDIT_FINAL.json, retaining
the original audit. This is an artifact-index fix, not an experimental rerun.
Exact full source IDs and tar hashes are in the final audit and job receipts.

The completed five seeds are allocation-head training seeds on one fixed C0.
Within each seed success/dense training sample-stream hashes match exactly.
Actual100-step and50+50-step runs compare model, optimizer, RNG, stream, history,
settings and step with exact equality (RESUME_HEAD.json, RESUME_GEOMETRY.json).
Serialized file hashes differ normally; tensor/state equality is the criterion.

Public deployment replay for simple/dense has zero path/event/q error, exact
returned slots and one decoder call. Files *_DEPLOYMENT.json contain measured
full-API cached-feature latency and input keys. To load the ordinary option:

```python
from research_realized_coverage_v1.deployment import load_planner
model = load_planner(
    'runs/mode_geometry_v1/canonical_C_seed0/last.pt',
    'runs/mode_geometry_v1/fixed_assets/scorer_bundle.pt',
    'runs/realized_coverage_v1/success_context_fit0/step2400.pt',
    device='cuda')
# Only current observed input tensors accepted; wrap calls in inference_mode().
out = model(**observed_inputs)
assert out['paths'].shape[1:] == (8, 24, 3)
assert out['selected_paths'].shape[1:] == (4, 24, 3)
```

Seed0 is the predesignated example; seed1 is not substituted because its U8 is
slightly larger. Dense alternative uses dense_context_fit0/step2400.pt and
starter_checkpoint=success_context_fit0/step2400.pt. Checkpoint hashes prevent
combining these heads with geometry continuations. The API is an opt-in research
option; no original loader/default was redirected.

For exact historical experiment commands consult copied jobs/*/receipt.json:
initial feedback (including failure/resume), initial allocators, four geometry
curves, shared-context allocators, joint snapshot refresh, five-seed followup,
analysis, tests and final artifact audit. Fresh orchestration scripts are
run_realized_*_v1.sh; they intentionally fail on existing output names. Do not
blindly rerun them in the populated results directory. Rename outputs/IDs for a
fresh reproduction or replay immutable exports in a separate project-root copy.

Large feedback NPZ, all checkpoints (including optimizer/RNG), prediction pools,
stdout/stderr and receipts are copied to local runs/realized_coverage_v1 and remain
on wzy3090 at the corresponding project root. They are ignored by Git. Compact
results and real figures are committed under this research directory. Source
tar exports are locally saved alongside the closure archive. After extraction,
`python scripts/close_realized_coverage_local.py` checks every indexed file,
source export, fixed C/q asset and receipt, then writes results/CLOSURE.json and
LOCAL_HASH_VERIFICATION.json. Costs have no historical remaining-time cap.
