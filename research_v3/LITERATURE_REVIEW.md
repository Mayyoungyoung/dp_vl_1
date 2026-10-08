# Primary-source novelty audit — accessed 2026-10-09

This is an initial scope review, not a novelty claim or external benchmark result.

- [3D HAMSTER paper](https://arxiv.org/abs/2606.31329) and
  [official implementation](https://github.com/DAVIAN-Robotics/3D_HAMSTER):
  metric depth-conditioned upper-level trajectories with a separate controller.
  The released repository supplies planner inference, not low-level policy code.
  Its 9B checkpoint and backbone differ from our frozen 2B frontend. Official
  inference cannot be relabeled as a capacity-matched learned M8 baseline.
- [RoboTracer](https://arxiv.org/html/2512.13660v1): spatial referring, measuring
  and multi-step tracing, with scale-aware supervision and process rewards.
  It establishes prior art for 3D grounded tracing. Neither a 3D waypoint head
  nor adding process constraints alone establishes novelty here.
- [rMCL](https://papers.nips.cc/paper_files/paper/2023/hash/12d7ba753894ed348904df1bf0ce02ec-Abstract-Conference.html):
  multiple hypotheses, WTA regression and learned scoring already exist.
  Our checker-conditioned q must stay distinct from hypothesis occupancy mass.
- [ModeSeq](https://arxiv.org/html/2411.11911v2): sequential prediction across
  modes, iterative refinement and early-match assignment seek diverse sparse
  forecasts. Sequential modes, query interactions and assignment are established
  alternatives, not automatic innovation.
- [CoverNet](https://arxiv.org/abs/1911.10298): classification over trajectory
  sets constructed for coverage and physical feasibility. Explicit route sets
  and geometry-aware candidate organization have substantial prior art.
- [Safe planning with conformal prediction](https://arxiv.org/abs/2210.10254):
  calibrates prediction uncertainty for planning in dynamic environments.
  Calibration and risk selection need their own assumptions; they do not prove
  occluded geometry safe or justify multiplying marginal route probabilities.

Current comparison standard: a candidate mechanism must exceed mode balancing,
ordinary assignment and identical edit augmentation before expanding the paper
claim. Historical R2/R3 and dual-factor negative results remain constraints.

Additional primary-source checks in this continuation:

- [MPD](https://arxiv.org/abs/2412.19948) learns trajectory priors and combines
  them with planning costs during diffusion sampling; B-spline representation
  and constraint guidance are established alternatives. A cost-guided sampler
  with full geometry is not an observation-only, same-compute head baseline.
- [Trajectory-Level Mode Guidance](https://arxiv.org/abs/2609.36530) uses partial
  trajectory priors and timestep-dependent guidance. Merely conditioning a
  generator on a desired route mode cannot establish novelty.
- [Topological Trajectory Prediction](https://arxiv.org/abs/2301.09821) separates
  high-level class prediction from geometric trajectory prediction. Our passage
  vectors have no demonstrated strict homotopy interpretation.
- [Counterfactual trajectory analysis](https://arxiv.org/abs/2107.14202) addresses
  environment bias through counterfactual interventions. Paired scene edits are
  not by themselves a new contribution; the actual supervision and mechanism
  must differ and be ablated against identical edit data.
- [ReconVLA](https://arxiv.org/abs/2604.16677) and
  [flow-policy uncertainty](https://arxiv.org/abs/2606.18043) are relevant to
  uncertainty and failure prediction. Expert-action discrepancy or ensemble
  disagreement differs from checker-defined probability of this concrete path
  satisfying the task. None licenses treating candidate probabilities as
  independent or using demonstration frequency as validity.

These checks constrain mechanism selection; they are not an exhaustive novelty
search or reproduction of the external methods. RoboTracer has a newer v4
(2026-07-03); pin the cited version when writing the eventual related work.
