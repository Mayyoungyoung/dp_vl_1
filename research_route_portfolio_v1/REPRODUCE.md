# Reproducing the route portfolio prototype

Local writer F:/dpvlm, branch codex/multiroute-v2. Server ssh wzy3090, root /home/wzy/dpvlm/route_set_v1. GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab, .35 memory fraction, CPU0-3/four threads. No environment changes. Historical d0d97eb and runs/main stay preserved.

## Immutable sources and data

Prediction sources: a299ca5868d03393748d42897cfff73e32612c62 for the initial336B0 predictions, baded90a4883233850014acd1b3f2ad4c5ae41d9 for its check-only recovery and22remaining frozen arms, 4e0f6739ad0a702cb8e935c5ee6ff47198bb207d for preference tests/experiments, and2e0cc00365ee232cc6f6f00dcff1cc020c2ceb72 for exact replay/RNG capture. Each release is an immutable local Git archive of routeset, scripts, configs, tests and imported research directories. Server source hashes and local runtime-archive hashes are in results/ARTIFACT_AUDIT.json. Windows Git archive converts Python line endings to CRLF; actual export hashes are recorded and verified, rather than assumed equal to canonical Git blob hashes.

configs/route_portfolio_v1.json allowlists one DEV_MODEL export by observation/supervision SHA. It contains336requests, all retained. Never point this evaluator at a raw collector or any reserved TEST_LOCKED payload. Earlier TRAIN fitting and RNG/optimizer checkpoints stay at runs/mode_geometry_v1/canonical_{B0,Bset,C}_seed{0,1,2}. Optional heads are runs/realized_coverage_v1/success_context_fit{0,...,4}/last.pt and are bound to the C0 generator hash. No new training occurred in this round.

## Actual commands and recovery

Actual job commands, statuses and command seconds are indexed in results/ARTIFACT_AUDIT.json and full receipts under runs/selective_repair_v1/jobs/portfolio*. Source launchers enforce resource identity, immutable cwd and an exclusive lock. Do not rerun completed launchers into these output names.

The first checker failed after storing the entire prediction pool and PREDICTION_SEAL.json. Its successor used evaluate --name shift_B0_seed0 --check-only, verified the existing SHA and completed checking. Every other job has a distinct output name. The interrupted oversized source transfer failed extraction before any research workload; its partial export is preserved with a failed_full_export suffix. The successful scoped runtime export used a fresh directory.

For a genuinely new run, use a new job/output name and an immutable export:

```bash
bash scripts/launch_selective_repair_v1.sh --id NEW_UNIQUE_JOB -- \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  -m research_route_portfolio_v1.evaluate --name NEW_UNIQUE_OUTPUT \
  --checkpoint /home/wzy/dpvlm/route_set_v1/runs/mode_geometry_v1/canonical_C_seed0/last.pt \
  --sampling adaptive
```

The main stage writes and hashes every observation-only prediction before references/truth are checked. Each categorical trial sets its registered inference seed; the exact replay stores initial/final Torch CPU/CUDA RNG in runs/route_portfolio_v1/REPLAY_V1/sampling_rng.pt. Prefixed checkpoint/RNG states from historical training remain unmodified.

## Local analysis and artifacts

```powershell
python -m research_route_portfolio_v1.statistics --root runs/route_portfolio_v1 --output NEW_MAIN_SUMMARY
python -m research_route_portfolio_v1.preference_statistics --root runs/route_portfolio_v1 --output NEW_PREFERENCE_SUMMARY
python -m research_route_portfolio_v1.audit_and_figures
```

The numerical archive runs/route_portfolio_material_20261010_v1.tar.gz has SHA f1821b0918b3a1972815970195c6bc9b9b3d1b296af8d29bb96b0bb97a03be79. The replay/RNG archive runs/route_portfolio_replay_20261010_v1.tar.gz has SHA72ae6703f5f24c248187f846b678b44ea05135316247f6b68e61c740173f75c1. Both server/local archives match. Their unpacked pools and receipts are under dedicated runs directories. Figures use those actual pools and the earlier local development-data mirror; they do not regenerate predictions or filter successful cases.

All30jobs are terminal. The current prototype's claims are supported development effects, not independent final-test or robot-execution results. No new model became a deployment default.
