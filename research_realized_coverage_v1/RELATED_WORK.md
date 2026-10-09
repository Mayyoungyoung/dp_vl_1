# Targeted primary-source check, 2026-10-09

[Diverse Trajectory Forecasting with Determinantal Point Processes, ICLR2020](https://arxiv.org/abs/1907.04967)
learns context-to-latent-set sampling for a pretrained decoder, optimizing diversity
and likelihood jointly. Learning a proposal set before one trajectory decode is
therefore established, not a new contribution of this project.

[Diverse Probabilistic Trajectory Forecasting with Admissibility Constraints](https://arxiv.org/abs/2302.03462)
is directly relevant to diversity under feasibility/admissibility constraints.
Different robotics inputs alone cannot establish novelty over that research line.

[SetPO,2026preprint](https://arxiv.org/abs/2602.01062)
uses a set objective and leave-one-out marginal contributions for LLM diversity.
Marginal contribution itself is not new. Here replacing a query changes ALL routes
through decoder attention, unlike removing an already-generated fixed item;
actual generated net changes therefore need measurement. This is a potential
mechanism distinction, not proof of novelty, submodularity or a greedy guarantee.
No external benchmark superiority is claimed; strong simple controls must first
establish that learning these interactions has independent value.
