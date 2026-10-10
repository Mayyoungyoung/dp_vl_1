# Prospective execution-prefix study,2026-10-10

The17-class actual-word forecaster is rejected as the current mechanism. Its
word-safe correction restores planned geometry but yields actual returned4
E4=1.75 and8/16 successful-clear versus planned-word control2.75 and12/16.
Ordinary coordinate allocation gets the same1.75. OriginalU8+.15 gate remains
failed; the prospective body gate in BODY_PROTOCOL.md remains unchanged.

One new learning object: finite fixed-controller execution prefixes. Predict
the evolving public seven-joint state and within-segment actual tip path, plus
conditional first budget noncompletion. Compose operational words from the
predicted tip path with current-observation inferred boxes. This investigates
whether dense physical supervision and state propagation generalize where an
opaque word-class label failed. It is not a claim that recurrence,kinematics,
mixture prediction or coverage optimization alone is novel.

Same current24-node draft,NN completed posts,cached current observation and
public initial joints as inputs. The fixed public robot frame chain is recorded
read-only at a restored TRAIN pose. No scene geometry/IK/controller observation
enters forward. Actual q/tip traces are TRAIN-only supervision. Successful
segments provide four actual normalized-time states. Exact first-row-crossing
joint/tip events additionally come from the full trace,not uniform compression.
A prefit TRAIN check85successful traces finds4/8/16samples per segment retain
78/82/84words; even16samples alias a real crossing. This representation defect
motivates exact event supervision before any learner fit,not a resolution/loss
sweep. Predict crossing occurrence/public-FK state from each propagated segment
and compose row categories with current inferred boxes. Residual row dependence
conditional on joint-state hypotheses remains an approximation. A failed segment supplies
its observed budget noncompletion; its unexecuted suffix is unknown,not a
negative or fabricated state. Public FK must match every TRAIN sample within
5mm before fit; the first266-state check gives max1.182mm,mean.549mm.

Minimal state model:64-unit causal GRU with four trajectory hypotheses,public
FK and per-segment hazard. Train2400AdamW updates,batch32,lr.0003,three shared
seeds; reserve final two registered TRAIN families for diagnostics. Reduce
teacher forcing linearly to zero by update1200; report free rollout diagnostic
joint/tip errors and noncompletion calibration separately. Successful-prefix
joint and tip likelihoods have fixed scales. Actual finite failures remain
budget-dependent outcomes,not proofs of kinematic infeasibility. Native lower
planner randomness is not controlled by Python seed and requires repeat trials.

First test dense state model versus nonrecurrent state model and separately
trained binary risk classifier with the same actual allgoal TRAIN feedback.
Keep17-class model as historical rejection,not a tuning sweep. Strong geometry,
ordinary planned-word and coordinate optimization all share alternatives/legal
features. Preserve8finals/24internal alternatives/current-NN word-safe mask and
frozen fullq/return4. No new scoring search. Prior categorical models did not
use matched new success heads,so no formal acceptance was possible.

Before formal method claim:all3goals,matched B1/B3 TRAIN feedback,matched
TRAIN allocation heads,three paired actual runs,all8 common rawC0/base/M
damage cohort,native-repeat evidence,family CI,fresh frozen independent family
confirmation,full online latency and clear attribution. State/hazard forecasts
and preservation constraints are predictions,never safety certificates. If
held-out state transport is poor,do not hide it behind final outcome accuracy.

Relevant prior work includes learned feasibility and differentiable kinematic
proxies:[DiffCo](https://arxiv.org/abs/2102.07413),
[Forward Kinematics Kernel](https://arxiv.org/abs/1910.06451),
[3D action feasibility](https://ieeexplore.ieee.org/abstract/document/10161114).
Any eventual novelty must be demonstrated by selective counterfactual
execution preservation under observation uncertainty and finite output budget,
not by using these established ingredients.
