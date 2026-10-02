# Bounded diffusion convergence check

This is a standard baseline convergence check on controlled oracle geometry, one real training seed and DEV_MODEL only. The epsilon and v 3000-step experiments remain unchanged. The 3000-step v results show roughly 75% collision rate, so particle repulsion is not the next intervention: removing all valid duplicates would still leave fewer than one valid route per scene on average.

## Fixed decision and budget

Continue both independent and set-attention v denoisers from their respective **last3000** checkpoints to **12000 total steps**, once. Use the same data, positive-reference selection, optimizer, constant learning rate, batch64, K4, diffusion100, DDIM40 and DEV selection rule. No new loss, extra candidates, repair, PG strength sweep or model-size change. The selected best from the original run remains eligible in the new output.

Each arm has 768000 prior target slots, 2304000 additional slots and 3072000 cumulative slots. This is **four times the original regression/diffusion exposure**, not an equal-total-budget comparison to the 3000-step regressor. Intermediate evaluation remains every500 steps and the final K1/2/4/8 evaluations contain three separately evaluated sampling repeats. K1/2/8 remain unseen-K inference transfers, not budget-conditioned training. Three sampling repeats are not three training seeds.

The cap is12000. Read the final learning trend and failure decomposition, then retain the stronger baseline or conclude this implementation is still inadequate. A best checkpoint near the cap alone will not justify an automatic extension. Saturated controlled data will not become a venue for indefinitely adding mechanisms.

## Exact state and audit

`scripts/train_multigate_diffusion.py --continue-from <completed source/last.pt> --steps 12000 --output <fresh output>` loads model, AdamW moments, scheduler, global RNGs, independent parent/reference/noise sampler RNGs, cumulative sampled-stream digest, step, DEV selection value, history, loss tail and exposure counters. It copies the original best checkpoint into the new tree before training. All method and data-hash fields must match; only increased total steps and explicitly recorded output/runtime/provenance fields may change. The source must be completed, unlocked and consistent with its summary hashes. The original five artifacts are preserved.

Both cumulative `training_stream_sha256` and new-segment `incremental_stream_sha256` are recorded. The latter starts with a zero hash chain at step3001 and hashes the actual parent/positive-reference indices, timesteps and Gaussian noise with shape/dtype metadata; interruption/resume preserves both chains. The two arms must match both hashes, cumulative and incremental target slots, and reference-pool access. No assertion of matching streams is made from a seed alone.

`elapsed_s`, `incremental_elapsed_s` and `gpu_hours_reserved` belong to the new output, including its evaluations. `cost_origin` records the prior run once; `cumulative_elapsed_s` and `cumulative_gpu_hours_reserved` add that prior cost once. Exposure fields without the `incremental_` prefix remain cumulative. Summing a prior cost and the new cumulative cost would double count and is forbidden.

The targeted continuation test compares continuous4 against completed2 → new-output3 → resumed4. It requires bitwise-identical model, optimizer, scheduler, torch/noise RNG states, losses and cumulative sampler digest; independently replays the two-step segment digest; rejects changed data, learning rate and parameterization; and verifies original last/best/config/summary/history hashes are unchanged. Test success and GPU completion must be read from recorded immutable-release jobs, not inferred from prepared commands.

## Planned output and recovery

- Source: `runs/multigate_diffusion_v2/seed0/{independent,set_diffusion}/last.pt`.
- New output: `runs/multigate_diffusion_v3/seed0/{independent,set_diffusion}`.
- Launcher: a newly frozen file with actual deployed source hashes, normalized Git blob hashes, source checkpoint/metadata hashes and successful targeted-test guard. Root owns GPU startup.
- An interruption resumes only its new output using the recorded command plus `--resume`; do not relaunch the fresh two-arm launcher or change the completed source output.

Status at preparation: implementation ready for frozen-source targeted tests; no continuation training started by this worker. Actual job records and subsequent result report supersede this preparation status.
