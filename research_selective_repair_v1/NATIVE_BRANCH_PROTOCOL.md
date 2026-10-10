# Native branch pilot, frozen before fits

Motivation: TRAIN held prefix-loop probe returns to <1mm of start while joint
state changes .24..6.14rad. This supports a branch-state learning diagnostic,
not a loop primitive novelty claim or reliable execution gain.

Dataset: original2304 TRAIN event rows plus192 null/loop attempts, target0,
fit families0..3 and held14..15. All original16 families remain in training
except last2. Native completed endpoint joint labels are explicitly enabled
in this NEW contract. Original event-only fits do not use these labels.
Joint labels outside native public intervals are masked and counted; no
approximate FK/IK is invoked. Previous5mm FK gate is unchanged.

Model: four free-running bounded joint hypotheses; causal current requested
prefix features and current observation. Public registered initial joints
only. Measured future joints never enter forward. Native joint truncated
Gaussian auxiliary density averages seven coordinates and observed endpoints,
fixed weight1. Same censored hazard/crossing/complete-clear objective as prior
crossing model. No teacher forcing, threshold/loss/amplitude search.

Primary control: identical initialization, parameters, stream and labels,
but recurrent input masks predicted joint feedback to public initial joints.
Additional same2496row route-GRU and binary-success controls; empirical actual
word table conditioned on current inferred nominal mode and INTERNAL proposal
recipe, fixed .5 smoothing, excludes held TRAIN families.

Seed0 pilot,2400 updates,batch32,AdamW lr.0003, memory.35 GPU1/fourthreads.
Exact120 vs60+60 full optimizer/checkpoint/RNG equality required first.
Prediction bounds, unknown suffix masking and prefix causal invariance tested.

Inference: same identity/null/loop geometry-eligible three options per slot,
24 internal curves/6561 assignments, final8 and original full q returned4.
No extra controller/native/FK/IK query or actual state input. Every arm receives
same inferred boxes and prefix recipes. Only geometry screens on36 TRAIN-held
requests first; no DEV/fresh result or matched SuccessHead claim. Inspect state
quality and same-feedback control before any native returned4 expansion.

No declared acceptance change: eventual E4+.30, successful-clear+.075 and
family CI, original geometry/damage gates, three paired seeds and new frozen
confirmation remain required. This pilot cannot by itself meet these.
