# Route / mode mass / validity interface

The implementation keeps the existing frozen Qwen3-VL-2B semantic encoder and observation geometry pipeline. The packaged planner consumes their existing tensors. It does not include Qwen model weights or a robot executor.

```python
import torch
from routeset.observed_probability import load_scored_planner

planner = load_scored_planner("planner.pt", device="cuda")
with torch.no_grad():
    result = planner(
        features=features,       # B x 4096, cached or live frozen Qwen features
        current=current,         # B x 8, existing pose/open-state convention
        world_xyz=world_xyz,     # observed depth projected into world frame
        rgb=rgb, uv=uv, depth=depth, valid_mask=valid_mask,
        return_k=4,              # 1, 2, or 4; always generate all internal M
    )
paths = result["selected_paths"]   # B x K x 24 x 3, world metres
events = result["selected_events"] # B x K x 24
q = result["selected_q"]           # independent sigmoid validity scores
pi = result["selected_pi"]         # original full-pool mode mass, not renormalized
```

Inputs must use the same frozen Qwen revision, prompt/feature extraction, RGB normalization, depth projection, camera frame, and state conventions as the existing observation pipeline. `scripts.run_observed_probability.package` is the concrete reference for constructing an actual request. Invalid depth points are masked before nearest-surface queries. The checkpoint loader expects a locally generated, trusted checkpoint.

`paths`, `events`, `q`, and `pi` retain all M hypotheses; `selected_indices` identifies the returned subset. Selection starts with the highest q, prefers geometrically distinct paths among q >= 0.5, and fills remaining slots by q. The fixed threshold is a selection rule, not a guarantee. No checker label or reference path is available to this selector. Changing K does not regenerate the candidate pool. All M candidates must be charged when comparing candidate budgets.

The proposed M8 bundle has trained pi. The ordinary M8 comparison has a uniform, untrained pi head; the original M4 bundle returns `pi=None`. Never present those control outputs as learned mode probabilities. pi approximates equal mass over geometric reference clusters, not natural route frequency. Multiple hypotheses can represent the same mode, and absent reference modes cannot be inferred from this objective alone.

q predicts the registered upper-level tip-path checker: start/target tolerance, event sequence and continuous segment clearance. It is not whole-arm feasibility or execution success. Temperature calibration uses a separate reserved parent set, but empirical calibration results must still be examined. q and pi remain separate outputs; the implementation does not multiply them into an undocumented confidence.

Packaging verifies one actual cached-Qwen/RGB-D request against the saved generator/scorer outputs, reloads the bundle, and verifies exact path/q replay. This is an integration check, not an end-to-end VLM latency benchmark. The manifest records source model hashes, calibration hash, input identity, candidate counts and observed head latency.
