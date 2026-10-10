# Route portfolios and related trajectory models

Primary sources checked on 10 October 2026. Similar research problems motivate this work; methodological and empirical differences establish its position.

## Three dimensional guidance and spatial traces

[3D HAMSTER](https://arxiv.org/html/2606.31329v1) grounds hierarchical trajectory guidance in depth and couples a 3D planner with a point-cloud control policy. [RoboTracer](https://arxiv.org/html/2512.13660v4) emphasizes metric spatial referring, measuring and multi-step tracing, with scale supervision and process rewards. These are direct neighbors for observation-conditioned metric paths. Our focused question is how several alternative passages should share a small route budget for the same observation and task, including a preference supplied after observation. It is not a claim that these systems cannot sample diverse paths. Their pretrained systems would provide useful external comparisons, but their different data and backbones prevent treating an uncontrolled reproduction as a same-data mechanism ablation.

## Explicit modes and probabilistic trajectories

[MultiPath](https://arxiv.org/abs/1910.05449) predicts distributions over trajectory anchors and continuous offsets. [Motion Transformer](https://arxiv.org/abs/2209.13508) uses motion queries to separate intention localization from movement refinement. Explicit mode tokens and shared coordinate decoders are therefore established ideas. Here passage words refer to alternatives for one commanded robot task, rather than forecasts of another agent's behavior. The learned coordinates depend on current RGB-D; truth geometry only supplies TRAIN labels and independent evaluation. Masking a mode vocabulary is ordinary conditional generation, not a new theoretical contribution.

[Diffusion Policy](https://arxiv.org/abs/2303.04137) learns multimodal visuomotor actions through diffusion. A same-observation, same-demonstration trajectory diffusion model is an important baseline for route coverage per decoded candidate and per unit compute. Existing patch-diffusion repairs from the earlier branch do not constitute this whole-route baseline, and action diffusion must not be confused with high-level geometric route generation.

## Concrete differentiation

The research contribution being developed is an observation-conditioned route portfolio with operational passage identities, partial support evidence, and allocation before geometry generation. Its narrow empirical claims concern actual valid alternative coverage at eight candidates, same-mode adaptation, and actual preference compliance after the fixed four-route selection. The existing method already provides a concrete implementation and matched continuation comparisons; the preference interface adds a useful test of mode controllability.

The work does not depend on a completely unoccupied topic. It also does not claim that adding a mode embedding, success MLP, mask, or a new name is sufficient novelty. Its submission case should rest on the complete design, transparent comparison to ordinary assignment and sampling, demonstrated advantages in its chosen use case, and a broader benchmark. Strong simpler centerline results and failure of displacement/joint-feedback extensions remain part of the evidence.
