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
