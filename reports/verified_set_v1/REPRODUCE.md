# Reproduction and evidence map

This round uses the user's additional7200 experiment-command seconds, separately
from the exhausted historical research_v3 ledger. No TEST_LOCKED payload was used.
The only server is `ssh wzy3090`, root `/home/wzy/dpvlm/route_set_v1`.
GPU1 UUID `GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab`, memory fraction.35,
four CPU threads and affinity0-3 remain in force.

## Immutable sources

| Source commit | Purpose | Export SHA256 |
|---|---|---|
|a6096e44122c14e56ab2613b14f6c429a539efe3|Initial tests, restricted population and diagnosis|0bba312813ca6326d9b74d694a36b9995469c6ae1efb4611bf59d89aa65bb349|
|8ac68f92e9485de360545df06ba11f8baa8f45e4|Full-population preparation and initial three arms|bfdb3085698fa8eb856e69d61ee9945be463bf970871194740f3272bbdaa53b8|
|2761bc1b60567b77e0583f665d9a43e8f6129b46|Joint controls, replication and acquired-positive replay|9c3296c045bb686a7a596c08757e0143d4f39675185d0e3f7ff62a6885877d2b|
|5b0e0f6d21ab711f0f66874f15e4a6e2fa3b8c2c|Fixed-witness secondary diagnostic|57dde87fc0dc1627518eb43948fa9a0b2105c281b5f5fca600cc3bd81edb2650|
|a2cab9c920878159f31ce59e97cce74af44561c0|Paired replication aggregation|bbba75c23ea290dc2d6c82ff048fd8be1432532a0e7cf242c734342f5a6d866f|
|e4682476cbc0b8ffba2f9d93a7cd4dae771f156c|Final analysis, tests, checkpoint/RNG/replay audit|d30831e2ea12b09372be94a4efb31c98f5a9fcabd48e73be2983fb3605809b68|

Exports contain tracked routeset/scripts/configs/tests. Releases live at
`research_v2/releases/<full-commit>`. Actual imported source hashes and exact
commands are recorded per job, rather than relying only on a branch name.
Do not overwrite a release, launcher, completed job id or output directory.

## Artifacts

- Budget receipts and logs: `runs/verified_set_v1/jobs/<job>/`.
- Restricted diagnostic artifacts: `runs/verified_set_v1/prepared`, diagnostic,
  and ordinary/gate/project_seed0. These are confounded closed-only experiments;
  do not present them as the final fair control.
- Full population: `runs/verified_set_v1/all_population_v2/<arm>_seed<seed>/`.
  Each successful train stores `last.pt`, `recovery.pt`, `summary.json`, settings;
  checkpoint includes model, optimizer, scheduler, RNGs and actual sampler audit.
- `evaluation_matched_q_v1` contains metrics.json, rows.json and sealed pool.npz.
  Evaluation generates paths from observations before oracle label checking.
- set_point/set_project/project save the actual per-draw target arrays under
  replay_*.npy. Replay checks exact sampled IDs and hashes the acquired positives.
  It uses those positives with ordinary group matching; it does not copy online
  target assignment. Group RNG and processed-reference counts can consequently
  differ; input draw sequence is identical.
- Parent remains `runs/research_v3_v1/safety_mean/last.pt`, SHA256
  ff2dfbc5463a38e9acb2af740e9b605cbfda5d1317cc146f5f2c4e65c2a7a60c.
- Complete fixed scorer is `runs/research_v3_v1/matched_q_paired_v1/reliability/mean_seed0/calibration_seed0/scorer_bundle.pt`,
  SHA2562d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e.
  Do not combine these results with a changed scorer observation encoder.

## Entrypoints

From an immutable server release, commands run through the frozen budget wrapper:

```bash
bash scripts/launch_verified_set_v1.sh --id NEW_UNIQUE_JOB_ID -- \
  timeout 1200 /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  -m scripts.train_verified_set train --population all --arm gate --seed 0
```

This is a template, not permission to overwrite or rerun completed outputs.
`evaluate` is a separate120s-capped job. Analysis entrypoints are
scripts.analyze_verified_set, scripts.summarize_verified_replicates and
scripts.diagnose_verified_set_results. Their exact completed commands should be
read from the final copied receipts, including every compared arm.

No VLM weights were updated; the experiments reuse frozen cached VLM features,
and train the existing observation-conditioned geometry/route head. Frozen VLM
does not mean all observation encoders or the route generator are frozen.
This round tests controlled task-level reach routes, not executable robot motion.
