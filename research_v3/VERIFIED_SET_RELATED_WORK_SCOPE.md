# Scope and novelty audit (2026-10-09)

Primary-source checks during replication:

- [CLIC project](https://clic-webpage.github.io/) and
  [paper](https://arxiv.org/abs/2502.07645): corrective feedback is formulated as
  desired action sets. Thus replacing point labels by admissible sets is not our
  standalone novelty. Feedback type, policy family and supervision source differ
  from our geometric task-route oracle.
- [Set-Supervised Diffusion Policy](https://arxiv.org/abs/2606.01865) and
  [official implementation](https://github.com/ZhaotingLi/Set_Supervised_DP):
  action-chunk sets from corrections already exist for diffusion policies.
  Our current head experiments do not establish a superior diffusion objective
  and are not a reproduction of that method.
- [Diverse Probabilistic Trajectory Forecasting, DIVA](https://cedric.cnam.fr/~thomen/papers/ICPR22_Calem.pdf):
  trajectory diversity and admissibility are jointly considered in forecasting.
  We cannot claim that combining these two requirements is new. Its forecasting
  setting and drivable-area objective differ from instruction-conditioned
  task-level reach routes with finite output budget.

Do not cite an unrelated DIVA robot/environment-generation or driving-video
dataset as the trajectory-diversity paper. No state-of-the-art claim follows
from this short scope check; broader current literature and suitable implemented
baselines would be required before a submission.

The present claim boundary is especially narrow: frozen cached VLM features plus
an observation-conditioned route head, controlled TRAIN-only geometry checks,
fixed task-route evaluator and old DEV families. No LoRA/RFT, independent VLM
backbones, new locked test, robot control success, full-arm clearance, or unknown
real-scene verifier accuracy was tested in this round.

The target-set construction retains current verified modes by assignment, but
that is not a property of the optimized network. A self-target has zero own
regression gradient at construction; other slots, other requests, endpoint
grounding and geometric losses still update shared parameters. Gradient clipping,
AdamW and its weight decay also affect the update. The observed loss of fixed
edit-mode retention is compatible with this gap, but does not isolate any one
of those causes. Do not turn this explanation into a causal result.

For a CVPR main claim, the relevant question remains whether a fixed number of
observation-conditioned routes preserves useful feasible alternatives under
scene/instruction changes. The current positive raw-diversity result is a useful
control, not proof of this broader property. The failed region mechanism must
not become the title contribution merely because it is more complex.
