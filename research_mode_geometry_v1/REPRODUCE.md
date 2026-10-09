# Reproduction

Use branch `codex/multiroute-v2` in `F:/dpvlm`; no historical source/result was
overwritten. The authorized remote is `git@github.com:Mayyoungyoung/dp_vl_1.git`.
Only server `ssh wzy3090`, repository root `/home/wzy/dpvlm/route_set_v1` is used.
GPU1 UUID `GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab`, .35 memory fraction,
CPU affinity0-3/four threads. Existing `.venv/bin/python`; no environment changes.
Final exact environment/version/source/file indices are in ARTIFACT_AUDIT.json.

## Immutable experiment inputs

- Parent: `runs/research_v3_v1/safety_mean/last.pt`, SHA256
  `ff2dfbc5463a38e9acb2af740e9b605cbfda5d1317cc146f5f2c4e65c2a7a60c`.
- Complete fixed q: `runs/research_v3_v1/matched_q_paired_v1/reliability/mean_seed0/calibration_seed0/scorer_bundle.pt`,
  SHA256 `2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e`.
- Existing TRAIN pool: `runs/research_v3_v1/verified_edit_all_modes_support_v1/support.npz`,
  SHA256 `101c2a4acff9ff08a34ac6b5c0b3138f04ffd5deaea04c2ac8daf0bf7512ca33`.
- Prepared observation contexts and pairs: `runs/mode_geometry_v1/prepared_v2/train.npz`,
  SHA256 `ef2e3a75bc1c56aea8254bd9c6563f750e097807a64e7948b4f4447cd4d5cccd`.
- Fixed2169 parent witnesses: `runs/research_v3_v1/strong_counterfactual_v1/rows.json`,
  SHA256 `515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084`.

The original paired export has1152TRAIN/288DEV_MODEL requests from128/32 layout
families. No DEV positives feed updates. The shared dataset loader materializes
permitted DEV caches while preparation encodes/fits explicit TRAIN indices only.
Do not replace these inputs with a full raw collector directory. Do not open
reserved TEST_LOCKED or score/calibration-role payloads for fitting this model.

## Code and commands

Every server job runs from a committed local `git archive` exported to
`research_v2/releases/<full-commit>`. Both launcher and imported code are frozen.
Actual command/source hashes live in `runs/verified_set_v1/jobs/mg_*/receipt.json`;
failed commands are included. Final source exports and per-file hashes are indexed
by the artifact audit. The additional7200-second ledger is reused, never reset.
Inspect current ledger/lock before any new run, use fresh output/job names, and
never relaunch a completed queue into its old directories.

Main verified training source: `902f545eb2f51d8f2bf341275e0de99c40e32254` for
canonical seed0, `d642c3d90f23794d7948a12eae17ecb6dbba186d` for seeds1,2;
B_set seed0 uses `fae5e532782a6a0fa1d87741ca446a66879b7890`. Shared active training
behavior is unchanged except the explicitly selected full-positive branch;
actual initial tensor and sampled target-stream hashes must match within seeds.

From a fresh immutable export, the command template is:

```bash
bash scripts/launch_verified_set_v1.sh --id NEW_UNIQUE_ID -- \
  timeout 180 /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  -m scripts.mode_geometry_experiment train \
  --arm D --seed 0 --steps 1200 --name NEW_OUTPUT
```

Arms and options:

- B0: `--arm B --pair-weight 0`.
- B_set: `--arm B --pair-weight 0 --full-positive-set`.
- C: `--arm C`.
- D: `--arm D` (nonzero path-displacement supervision).
- Communication ablation: add `--independent-decoder` to C or D.
- D2 connection repair: add `--pair-allocation` to D; ordinary masked symmetric
  KL is added inside L_pair, while coordinate displacement remains active.

All main arms use1200final updates, batch32 (16 scene pairs), AdamW lr.0003,
weight decay.0001, lambda_mode.001, lambda_pair1, segment/floor penalty160;
no scheduler/DEV checkpoint choice. Seeds0,1,2 are continuation randomness from
the same historical parent, not independent VLM pretraining seeds.

Evaluation template (same wrapper, new job id):

```bash
python -m scripts.mode_geometry_experiment evaluate \
  --name NEW_OUTPUT --sampling adaptive
```

`adaptive` is the main model API setting; `balanced` is top-eight distinct words;
`ordinary` samples with replacement using fixed inference seed71239. Evaluation
first generates every prediction from observations, then runs the original
independent checker. All288 requests remain. q and its complete observation
encoder/normalization/temperature are unchanged; no DEV recalibration.

Analysis entrypoints: `scripts.analyze_mode_geometry` (fixed retention and repair,
optional true-output3D figures), `scripts.summarize_mode_geometry` (all-seed table,
32-family paired bootstrap), `scripts.diagnose_mode_geometry` (TRAIN interventions
or actual process-resume equality), and `scripts.audit_mode_geometry`.

## Recovery and inference

`last.pt` and `recovery.pt` contain model, optimizer, step, all RNG states, target
stream digest, settings and actual source hashes. Interrupt using `--stop-after N`
only in a new experiment; resume the same planned total with `--resume`. A finished
last checkpoint cannot be resumed/overwritten. The actual100-step versus50+50
process comparison is exact for model, optimizer, RNG, stream, history and settings;
the initial failed version and its unordered-key cause remain recorded.

The new inference wrapper preserves the complete fixed scorer:

```python
import torch
from routeset.mode_geometry import load_fixed_scored_mode_planner
planner = load_fixed_scored_mode_planner(generator_checkpoint, scorer_bundle, 'cuda')
with torch.inference_mode():
    output = planner(**observation_inputs, return_k=4)
# output['paths']: B,8,24,3; output['selected_paths']: B,4,24,3
# observation_inputs: features,current,world_xyz,rgb,uv,depth,valid_mask
```

The language is already represented in pinned Qwen features. This loader takes
neither old-scene routes nor truth geometry/correspondence labels. Run it under
the same device/memory limits; a CPU load is available for inspection. The
deployment replay check compares actual saved paths/events/q and selected indices.

## Artifacts and limitations

Weights, prepared contexts, all prediction pools, optimizer/RNG, figures and logs
are retained under `runs/mode_geometry_v1` on server and copied locally. Large
artifacts are intentionally ignored by Git; pushing code does not upload weights.
The six requested Markdown files and compact JSON/figure evidence are committed
under `research_mode_geometry_v1`. See final artifact hashes for exact binding.

Historical pilot/full outputs before canonical sampling are exploratory only.
They cannot substitute for the verified same-stream comparisons. Every loss-scale,
sampling-policy and architecture distinction is recorded in PROTOCOL.md. Runtime
reports exclude online Qwen extraction because this study reuses frozen features;
do not call cached head throughput full VLM inference speed or robot success.
