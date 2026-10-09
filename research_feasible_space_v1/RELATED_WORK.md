# Bounded primary-source check,2026-10-09

[3D HAMSTER](https://arxiv.org/html/2606.31329v1) predicts metric3D trajectories
from RGB-D/language and integrates them with control. RGB-D3D generation is an
existing capability; no contribution is claimed for freezing its feature analogue.

[CorDriver / Drive in Corridors](https://arxiv.org/html/2504.07507v2) learns
rectangular corridors and uses differentiable trajectory optimization. Learning a
corridor then planning inside it is direct prior art. Our projection/centerline
controls must challenge whether learned internal generation adds useful behavior.

[Graphs of Convex Sets](https://manipulation.mit.edu/trajectories.html) addresses
motion planning using convex sets; convex/Bézier constraints are conventional.
[Neural GCS](https://arxiv.org/html/2608.15440v1) learns surrogates for relaxation,
flow and rounding in mixed discrete-continuous planning. Fast amortized planning
in convex sets is therefore also prior art.

[C-IRIS](https://arxiv.org/abs/2302.12219) certifies convex regions in a rational
configuration-space parameterization. The present swept boxes are Cartesian
end-effector regions, not certified robot configuration-space cells. Predicting a
region from incomplete observation does not inherit the oracle certificate.

Candidate distinction to test: observation-derived, mode/variant-specific boundaries
remain invariant to companion replacement while peer-aware internal parameters
remain contained; assess practical modes, adaptation and set quality beyond same
geometry ordinary XYZ/projection. This is a hypothesis, not established novelty.
