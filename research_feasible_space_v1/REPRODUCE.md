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
 --name NEW_OUTPUT --arm bounded --steps 2400 --seed 0 --cell-loss
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
