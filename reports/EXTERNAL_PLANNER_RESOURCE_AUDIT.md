# External planner resource screening — 2026-10-03

The official public checkpoint metadata was queried at fixed revisions. No weights were downloaded, no model was loaded and no external result was reproduced. The reproducible metadata script and full configuration/hash index are [audit_external_planner_resources.py](../scripts/audit_external_planner_resources.py) and [metadata](external_planner_resource_audit_v1.json).

| Official checkpoint | Pinned revision | Serialized weights |
|---|---|---:|
| [HAMSTER](https://huggingface.co/yili18/Hamster_dev) | `794f1f925c87e861d2f562943e978cc11f8c344d` | 25.134 GiB |
| [3D HAMSTER](https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER) | `ddc5987a56cdcb14e5e2297817612532e46e912b` | 17.024 GiB |

Both exceed the retained 35% allocation cap on the 24-GiB GPU for ordinary full-weight GPU loading, before activations. Serialized size is a screening calculation, not a measured inference peak. CPU offload or quantization could change feasibility and cost; neither has been tested here. Do not silently weaken a system baseline, exceed the GPU envelope, or label source inspection as reproduction.

HAMSTER's [official beta README](https://github.com/liyi14/HAMSTER_beta) also gives inconsistent VILA revisions in its dependency header and setup instructions; a reproduction must resolve and pin an actual working dependency. Its public checkpoint is nested under a training-run directory, rather than a root config. 3D HAMSTER packages its geometry encoder with the planner and provides metric depth output, but its pretraining and model size differ from this project's 2B model. Any eventual comparison must be labeled as a pretrained system comparison with its full encoding/projection/offload cost, separate from the same-data mechanism comparison.
