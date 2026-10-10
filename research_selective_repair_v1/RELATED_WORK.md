# Direct precedents and falsification controls

Checked primary sources on 2026-10-09. The initially requested arXiv abstract
fetch had a network failure; HTML and publisher search then succeeded.

[Cascaded Diffusion Models for Neural Motion Planning](https://arxiv.org/html/2505.15157v1),
section IV-B, finds connected violating subtrajectories and applies its local
diffusion model once between nearby valid states. This directly precedes local
patching. Our proposed experiment concerns locating modifications from current
RGB-D while controlling damage to eight strong baseline routes; its distinction
requires measured same-information controls. We do not call a geometry-triggered
MLP a faithful implementation of that paper's diffusion architecture.

[DiffusionSeeder](https://proceedings.mlr.press/v270/huang25f.html) learns trajectory
initialization for motion optimization. Learned seeds or correction coordinates
must demonstrate gains over ordinary seeds with the same solver; changing the
sampler name does not establish a contribution.

[Learning Large Neighborhood Search Policy for Integer Programming](https://proceedings.neurips.cc/paper_files/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html)
learns which variables to release before solver reoptimization. Learned edit
support is therefore an established design principle, not by itself novelty.

[Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)
keeps baseline behavior where offline evidence is insufficient under its own MDP
assumptions. Our finite RGB-D route repair does not inherit its safety theorem.

The decisive contrasts are same-data center updates, whole residuals, global
scaling, observed geometric triggers and finite local optimization, followed by
frozen new-family and full-arm evidence. If ordinary methods account for gains,
the claimed mechanism must change. No prior-art novelty is assumed in advance.
## Closest body-feedback precedents checked2026-10-10

[DiffCo](https://arxiv.org/abs/2102.07413) learns differentiable collision
proxies and uses their gradients in trajectory optimization. Thus a learned
body collision field plus gradient repair is already a direct precedent;
neural feasibility + optimization is not itself an innovation claim.

[Active Learning of Abstract Plan Feasibility](https://arxiv.org/abs/2107.00683)
learns plan feasibility from robot success/failure interactions and uses an
infeasible-subsequence property for efficient acquisition, including Franka
Panda experiments. Predicting sequential execution likelihood or using prefix
failures is therefore also not independently new. Our measured current gap
is metric Cartesian route prefixes under one unchanged controller and current
RGB-D; any narrower differentiable repair contribution must be established
against same-feedback/same-constraint controls, not inferred from task names.
No body method or novelty is accepted at this stage. This is a bounded primary
source check, not a new survey project. Static IK unknowns and root-planning
successes in BODY_DIAGNOSTICS.json cannot be substituted for full execution.

[Optimizing Sequences of Probabilistic Manipulation Skills Learned from Demonstration](https://proceedings.mlr.press/v100/schwenkel20a.html) learns probabilistic skills/confidence and optimizes continuous parameters in 7DoF manipulation. Probability forecasting plus continuous parameter repair is established prior art; any proposed contribution must come from measured cross-scene mode preservation and actual execution distribution, with same-feedback controls.
