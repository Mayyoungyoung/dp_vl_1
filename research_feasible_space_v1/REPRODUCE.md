# Reproduction and recovery

Local code writer F:/dpvlm, branch codex/multiroute-v2; remote
git@github.com:Mayyoungyoung/dp_vl_1.git. No force push. Historical files/defaults
unchanged. Server wzy3090 root/home/wzy/dpvlm/route_set_v1, existing.venv/bin/python.
Only GPU1 UUID7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,.35memory,fourthreads/CPU0–3.
No shared environment changes. New runs/feasible_space_v1/jobs ledger has no old
total-time cap; actual commands/sources/timings/exits are retained.

Fixed C0 runs/mode_geometry_v1/canonical_C_seed0/last.pt,
SHA6090df425d0ac237f275620f5de5e9453ce8e5c9638fbda1afed2e39c4ea6718.
Complete q SHA2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e.
TRAIN support SHA101c2a4acff9ff08a34ac6b5c0b3138f04ffd5deaea04c2ac8daf0bf7512ca33;
prepared observed context SHAef2e3a75bc1c56aea8254bd9c6563f750e097807a64e7948b4f4447cd4d5cccd.
Corridor labels SHA03f3761710480c2bf27d6119702041036d2d509f9f89098c9126f43aeab3bdeb.

Immutable git archive exports contain routeset/scripts/configs/tests and both
needed research packages. Extract into NEW research_v2/releases/FULL_COMMIT;
never modify a running export or launcher. Absolute launcher:

```bash
bash RELEASE/scripts/launch_feasible_space_v1.sh --id NEW_JOB -- \
 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_feasible_space_v1.train \
 --name NEW_OUTPUT --arm bounded --steps 2400 --seed 0 --cell-loss --tapered-cells
```

All trainers protect completed outputs. Checkpoints preserve model, AdamW,
Torch/CUDA/Python/NumPy RNG, actual sample-stream digest, history, settings,
initial tensors and frozen C hash. For interruption use SAME planned total steps,
same immutable source/settings and `--resume`, retaining recovery.pt. Changing
`--stop-after` affects interruption only. Source/data/policy hashes are in receipts.

Stage A:`prepare`, then `train --oracle --arm xyz|bounded --steps600` and
`oracle_diagnostic --checkpoints...`. TRAIN reference cells are explicit oracle
diagnostics. Observed screen uses train/evaluate; no oracle corridor arguments.
For each final snapshot,`feedback --checkpoint PATH` collects TRAIN mode outcomes
with fixed16word/two-slot companions;`feedback --pool POOL_NAME` fits ordinary
success2400. `evaluate --head MATCHING_HEAD` rejects stale generator hashes.
Center and projection inference are `evaluate --kind center|projection`, with
exactly the same predicted cells and fixed q. Main outputs8x24XYZ+events, return4.

The opt-in API is `research_feasible_space_v1.deployment.load_planner`; only
observed tensors are accepted. Predicted cell containment is never a true safety
certificate. Internal widths/connectors are serialized in every evaluation pool.
`check_deployment` replays actual API paths/events/q and checks one decode/request.
`analyze` reuses immutable2169/270parent opportunities. `summarize` includes all
finished rows, costs, failed jobs and paired-family intervals. `figures` uses real
RGB and saved predicted cells, explicitly marks gray oracle obstacle overlays.

Job scripts are fresh-output orchestration examples; they intentionally fail on
existing names. Rename all outputs/jobIDs for reproduction; do not blindly rerun
scripts in the populated root. All DEV results are repeatedly used development
evidence. New scene/general robot claims require separate frozen protocols.

## Frozen evaluation-only extension

Immutable source `326c57f271c8b64b147380429226b85e3f04388e` runs
`scripts/run_feasible_generalization_v1.sh`. Its prepare_v2 job follows a preserved
failed prepare_v1 from36741984feeb30ca5902c2a1bd9d31e66023ceba; no data existed at
that failure. The config fixes16fresh families /336requests before collection.
Outputs live only in data/feasible_space_generalization_v1 and the new RUN ledger.
One sequential worker uses the existing .venv-sim, softwareGL and its own Xvfb.
The launcher stops only that helper on exit. No simulator package is installed or
modified. Observation features use the existing pinned .venv-qwen/model snapshot,
GPU1/.35memory/four threads. Cached input contract/revision/dtype/max_pixels match
the inherited paired cache. No raw collector directory or TEST_LOCKED is evaluated.

Fresh inference is `evaluate --data-root NEW_DATA --expected-requests336` using
the same frozen generator/head hashes. A guard accepts only the registered new
data root and the expected evaluation-only protocol. Known-mode recall is based
on incomplete newly verified geometric teachers, not old all-mode support.
`generalization_stats` reports all3seeds and all7variants, without model selection.

After all jobs finish, run summary_v3 from an immutable export, then `audit`
outside the job wrapper (the audit rejects an active lock). ARTIFACT_INDEX.json
hashes every dedicated RUN/new-data file and checks actual receipt source hashes
against immutable exports. The archive contains both dedicated roots. Verify its
SHA and every indexed file after copying locally; preserve failed receipts and
partial artifacts. Source tar archives remain in ignored source_exports.

Compact JSON and figures are delivered under research_feasible_space_v1/results;
weights, full prediction/outcome pools, receipts and new rendered observations
remain in ignored runs/feasible_space_v1 and data/feasible_space_generalization_v1.
Local scientific plots and Markdown tables read those actual JSONs only. Scripts
plot_results.py,plot_generalization.py andwrite_reports.py record their inputs;
the closure manifest records local rendering/archive verification separately.


## Delivered closure

All 151 ledger jobs are terminal: 148 completed and three failed, with every
failure retained. Exact aggregate command time is 5,988.6731 seconds, including
CPU scene rendering; it is not active GPU time. The corrected reference recall
uses `correct_fresh_recall` followed by `generalization_stats` v2, without a new
model decode. `witness_statistics`, `parameter_transfer`, `benchmark_online`,
`dependency_snapshot` and final `summarize`/`audit` receipts are preserved.
Final summary/audit source: `1ac5977fdf60871d37bbdefecac6ca602c15434d`.
Real paired tapered figures v3 were rendered outside the RUN from immutable
`2c42e5749c9c4a7f0cc691a8bd378580459208ef`; their own manifest records inputs,
command and source hash. plots_v2 and fresh_plot_v1 are the final aggregate plots.
Older compact results/figures remain explicit superseded evidence.

The archive is `runs/feasible_space_v1/artifacts/feasible_space_v1_closure_1ac5977.tar.gz`:
2,502,586,877 bytes, SHA256
`3fed768a0c1b5191b84e31a8392936ce3da422a20820564a75d5001ed6cc2a8e`.
After path guards and extraction, 24,806 indexed files totaling 2,975,246,783 bytes
and immutable source exports passed hash verification. ARCHIVE_COPY.json records
actual commands; LOCAL_HASH_VERIFICATION.json records the local verification.
Archive creation wall time was not separately measured; no estimate is substituted.
`dependencies_v1` includes verified C0, complete frozen q and TRAIN support/context
copies. Generic frozen Qwen and simulator assets are existing pinned server
resources, not installed or altered by this study.

To verify the delivered local mirror without rerunning experiments:

```powershell
python research_feasible_space_v1/verify_artifacts.py --root F:/dpvlm --index F:/dpvlm/runs/feasible_space_v1/ARTIFACT_INDEX.json --output F:/dpvlm/research_feasible_space_v1/results/LOCAL_HASH_VERIFICATION_NEW.json
```

Use a fresh output name for any future reproduction. The publication-potential
goal is unmet; completed queues must not be relaunched as if pending.
