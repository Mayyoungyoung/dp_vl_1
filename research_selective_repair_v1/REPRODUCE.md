# Current reproduction and continuation

Local code writer:F:/dpvlm,branch codex/selective-route-repair-v1. Historical
defaults/data/runs remain unchanged. Read STATE.md,RESUME.md,PROTOCOL.md and
RESEARCH_STATE.md before launching anything. Every job has a new name/output.
Never duplicate an active queue or overwrite a running script/source export.

Server root:/home/wzy/dpvlm/route_set_v1. Only ssh wzy3090,physicalGPU1 UUID
GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,.35GPU memory,CPU0–3. Existing .venv
is the Torch runtime,.venv-sim is the PyRep runtime,.venv-qwen is the VLM runtime.
The simulator needs no Torch; child locale and owned Xvfb are launcher-managed.

Freeze a local Git commit with git archive containing routeset,scripts,configs,
tests,research_realized_coverage_v1,research_feasible_space_v1 and this directory.
Copy to research_v2/incoming/<fullSHA>.tar and extract into a new
research_v2/releases/<fullSHA>. Actual wrapper cwd/source hashes are recorded.
Launch only the exported scripts/launch_selective_repair_v1.sh,which enforces
hardware identity,thread limits,exclusive study lock and terminal exit receipts.

Current renderer/target coordinator is frozen bc22d916027d5e9309e60a7517e855298f7f5a65,
scripts/run_selective_recovery_v3.sh. Initial executor/model screens are already
terminal; do not rerun that script into existing output names. Rendering continues
from registered plans; collector skips only completed receipts. A failed/running
receipt requires explicit diagnosis and new recorded recovery,not silent replay.

Current successor14334bb664fe04900b6c5e27fd8ce2eceb3e06f7 runs
scripts/run_selective_calibrated_v1.sh after exact predecessorPID622165 exits
and interventions_goal_v1/MANIFEST.json exists. It executes structural tests,
shared TRAIN calibration,current evidence caches,matched labels,rule/geometry
controls,and3paired2400step seeds. It is already waiting; do not launch another.
Next action is inspect terminal source/receipts/results,then decide the measured
next mechanism step or formal matching-head/frozen-confirmation stage.

Train CLI supports --stop-after with recovery.pt containing optimizer and all
RNG states. Resume with the exact immutable source/config/data/seed using --resume
and the same output; last.pt prevents accidental completed-training replay.
views.py freezes exact prototype/threshold/scale into a decoder SHA before
matching TRAIN success-head fitting. feedback.py verifies that frozen view.
All matching heads must precede any formal accepted result or final confirmation.

Figures can be rebuilt locally without Torch:

```powershell
python -m research_selective_repair_v1.figures --receipts runs/selective_repair_v1/initial_receipts --output research_selective_repair_v1/figures/initial
```

This initial figure input contains actual sealed grids and fixed executor data.
Weights,prediction pools and full receipts stay in dedicated runs/selective_repair_v1.
No whole raw collector directory or reserved TEST_LOCKED payload is evaluated.
