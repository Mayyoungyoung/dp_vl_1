# Multiple paths with task and conditional-feasibility scores

This is an implemented research prototype, not a claim that the mechanism has already outperformed its controls. Final conclusions belong in RESULTS_REPORT.md after the complete crossed-seed comparison.

## Interface and scope

The existing frozen R1 observation-conditioned generator emits eight H24 XYZ paths and reach events. Qwen3-VL-2B remains frozen, RGB-D is backprojected using supplied camera calibration, and no target coordinates, physical obstacle boxes or reference paths enter inference. A path-conditioned scorer emits task matching a, conditional geometric feasibility b, and q=a*b. The highest-q route can be passed to an existing executor; all eight candidates and scores remain available. No controller is trained here, and q is not whole-robot execution probability.

## Event factorization

For observation/instruction/actual-path input x, define T as the existing checker event of correct semantic goal and finite/correct reach-event sequence. Define F as finite XYZ, start alignment, continuous tip clearance and the registered workspace lower bound. The historical joint event is checked exactly to equal Y=T AND F for every reused candidate.

The conditional model targets a=P(T=1|x) and b=P(F=1|T=1,x). The chain rule gives P(Y=1|x)=a*b without assuming conditional independence of T and F. The feasibility output is explicitly conditional: it is not an independently validated marginal geometry probability on wrong-task routes. Empirical estimation and distribution shift can still make the product miscalibrated.

Training minimizes unweighted task BCE on every actual candidate plus conditional feasibility BCE averaged only over task-positive candidates. An empty task-positive minibatch contributes zero conditional loss. There is no invented negative label for a route absent from demonstrations, no candidate softmax and no fitted demonstration-frequency objective. Multiple reasonable paths may all receive high scores.

## Architecture and controlled alternatives

Each head uses the pre-existing route-observation representation: 24 vertices plus23 segment midpoints, each with80 features from observed nearest surfaces and route geometry; mean/max aggregation of a64-wide node MLP is combined with131-dimensional scene/endpoint context. This feature extractor already existed; adding along-route geometry is not a claimed contribution of this experiment.

The marginal control uses the same heads but learns P(F|x) over all candidates, then multiplies two marginal predictions. The matched-capacity joint control has exactly the same two heads and initialization; its combined logit is (z0+z1)/sqrt(2), trained on Y alone. The smaller original single-head architecture is also retrained. These separate parameter count, component supervision and conditional masking.

Generator and scorer seeds are crossed, not merely paired:3 frozen generators x3 scorer seeds x4 arms. Every arm sees the same sampled requests within a cell, with no change in generator candidates or reference coverage. A score improvement must not be described as improved raw multi-route generation.

## Calibration and evaluation

SCORE_TRAIN fits parameters and normalization; DEV_SCORE selects the minimum joint-NLL checkpoint; CALIBRATION fits the final two scalar calibration parameters. Factor heads get separate positive temperatures, conditional feasibility fitted only on T=1. The joint controls get positive Platt slope and intercept. This matches scalar count but not functional family or fitting objective; report raw predictions as well.

Primary evaluation uses unchanged natural generator pools on reused paired DEV plus both historical development splits. Report joint Brier/NLL, per-factor task and conditional-feasibility quality, highest-score validity, selected-route calibration, fixed-K4 all-valid and distinct-valid-mode counts. Classifier labels do not certify occluded geometry or arm execution. Family-bootstrap uncertainty conditions on the trained models; crossed seeds are not new independent scene families.

## Research position

[SayCan](https://arxiv.org/abs/2204.01691) combines language-model skill relevance with learned skill affordances. This implementation transfers that functional separation to actual continuous waypoint candidates. Unlike SayCan's RL value learning, this round uses full upper-level success/failure labels and supervised critics. The product and probability chain rule are established ideas, not new theory. Any contribution must come from demonstrated reliability/diversity improvements, robust transfer and a precise empirical explanation over strong controls; these are hypotheses until measured.

The concise intended research question is: does explicitly separating task errors from conditional geometric failures improve the reliability of selecting among multiple observation-conditioned paths? If matched joint or marginal controls perform equally well, that result limits the conditional-factorization claim and must be retained.
