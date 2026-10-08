# Reproduction and artifact entry points

Run from F:/dpvlm locally. Binary archives are intentionally Git-ignored; a Git
checkout alone is not the dataset/checkpoint delivery. Closure manifests name
the local roots and every verified artifact SHA. Keep those roots or restore
their archived contents before these commands. Use fresh output names; never
overwrite an old experiment or a running source export.

## Local, read-only table/model verification

This command was actually run with output research_v3/delivery_verification_v1
and passed16rows. For another invocation choose a NEW output directory:

```powershell
F:\ProgramData\anaconda3\python.exe -m scripts.research_v3_verify_delivery --workspace F:/dpvlm --output runs/research_v3_local/verification_NEW_ID
```

It hashes all16listed checkpoints, checks recorded input/target exposure and
every numeric table metric against a matching sealed evaluation, verifies pool
hashes and recomputes validity/Brier/Top1 from saved candidates. It writes a
rebuilt CSV plus input/script/command hashes. It does not rerun geometry checks,
fit a model or prove method novelty. It keeps original-q and paired-domain-q
evaluations separate. Canonical training_seconds is the resumed loop only, as
stated in its row; command receipts contain its retained timeout and full cost.

## Audit report and 3D failure-figure regeneration

These are deterministic postprocessing of previously sealed permitted records,
not new model experiments or reserved TEST access. Existing checked products
are research_v3/audit and research_v3/figures_failures_v1.

```powershell
F:\ProgramData\anaconda3\python.exe -m scripts.research_v3_write_audit --source research_v3/audit --output runs/research_v3_local/audit_reports_NEW_ID
F:\ProgramData\anaconda3\python.exe -m scripts.research_v3_plot_failures --snapshot runs/research_v3_checkpoint_20261009/runs/research_v3_v1 --closure runs/research_v3_scoring_closure_20261009/runs/research_v3_v1 --inputs runs/research_v3_qualitative_inputs_v1 --output runs/research_v3_local/failure_figures_NEW_ID
```

Failure figures preserve all8routes/q values for mechanically selected DEV
examples. Actual rendered RGB and evaluation-only goal/obstacle overlays are
identified in their captions. These examples are not prevalence estimates.

## Server data checking, training and evaluation

Only ssh wzy3090, GPU1 with UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,
35%memory and4CPUthreads. Use existing .venv, no installation or shared env edit.
Every model/data-check command goes through scripts/launch_research_v3.sh.
That wrapper checks the physical GPU identity, enforces the cumulative7200s
policy and saves source hashes/actual commands/exit status, including failures.

**Current remaining20.568820s cannot support new training or full audit replay.**
The following are explicit reproduction templates for a newly authorized budget,
not instructions to launch them now. Policy changes must be made locally,
committed and exported into a fresh immutable release; never edit old releases.
Reusing sealed input/parent checkpoints permits a fresh run name without
recollecting data or touching existing runs. Full from-scratch collection has
separate source-indexed receipts and is not claimed as a tested one-command rebuild.

Recorded all-mode control source:
`/home/wzy/dpvlm/route_set_v1/research_v2/releases/60e562f483fcda7455329b105b06b495c6e23885`.
Actual coordinator: `scripts/run_research_v3_all_mode_completion.sh`.
It is a closed-run record with fixed output names; do not rerun it over outputs.
From a suitably authorized new immutable release with the same imported code,
the individual interfaces are:

```bash
bash scripts/launch_research_v3.sh --id replay_data_NEW_ID -- \
 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.research_v3_audit audit --output replay_data_NEW_ID
bash scripts/launch_research_v3.sh --id replay_all_modes_train_NEW_ID -- \
 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.research_v3_frequency train \
 --arm set_matching --name replay_all_modes_NEW_ID --linear-control mean --all-mode-edit-support
bash scripts/launch_research_v3.sh --id replay_all_modes_evaluate_NEW_ID -- \
 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.research_v3_frequency evaluate_fixed_q \
 --name replay_all_modes_NEW_ID --paired-score
```

Training above reuses the exact safety_mean parent and sealed56920-reference
pool, original grounding labels and seed0. Initial/input stream and matching
policy can be checked against the archived config, summary and last/recovery
optimizer/RNG/sampler state. It reproduces the rejected control, not the
retained reference. To reproduce the ordinary average-penalty reference from
its sealed full-set parent, use train --arm set_matching --name NEW_NAME
--safety-control mean, preserving its original1200steps/config and parent hash.
Original and new q versions must not be interchanged when comparing scores.

The uncompleted vocabulary-statistics phase of the final TRAIN audit is
explicitly failed (exit124); it is not in the successful-reproduction list.
Do not restart its full forward when saved predictions already exist, and do
not relocate it outside the wrapper to evade its cap. A future authorized
recovery must register reuse of those exact saved predictions and preserve the
failed receipt.

## Actual model inference

Use the fully specified observation-only CLI in [DEPLOYMENT.md](DEPLOYMENT.md).
The retained reference generator SHA is
ff2dfbc5463a38e9acb2af740e9b605cbfda5d1317cc146f5f2c4e65c2a7a60c;
the preselected paired-domain complete-q seed0 bundle SHA is
2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e.
The actual independent CLI replay preserved paths/events/selection exactly,
qmax difference1.1921e-7; its2.947s timing excludes online Qwen. It does not
include oracle geometry as model input or execute a robot.

Each closure manifest points to full receipts rather than substituting launch
commands for success evidence. Local verifications/plotting/packaging are
bookkeeping; server experimental command seconds remain separately reported.
