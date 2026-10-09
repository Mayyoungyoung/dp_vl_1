# Ablations and repairs

Every exploratory screen uses seed0 and is labeled as development evidence. Final acceptance uses the locked three-seed paired representation and fresh-head strong controls.

|Exploratory arm|V8|U8|V4|U4|Region geometry feasible|Membership|
|---|---:|---:|---:|---:|---:|---:|
|screen_xyz_seed0|86.76%|6.5729|93.40%|3.7153|7.55%|94.49%|
|screen_relative_seed0|86.98%|6.6111|93.40%|3.7222|7.64%|98.61%|
|screen_bounded_seed0|87.11%|6.6285|93.49%|3.7257|7.60%|100.00%|
|screen_peer_seed0|85.72%|6.5382|92.88%|3.6944|8.20%|100.00%|
|envelope_xyz_seed0|85.98%|6.5278|92.19%|3.6701|71.18%|34.90%|
|envelope_bounded_seed0|86.68%|6.5868|92.27%|3.6667|71.14%|100.00%|
|refreshed_envelope_xyz_seed0|85.72%|6.7188|92.27%|3.6910|70.05%|33.33%|
|refreshed_envelope_bounded_seed0|86.46%|6.7708|92.36%|3.6944|69.92%|100.00%|
|tapered_xyz_seed0|87.41%|6.6493|93.58%|3.7188|73.57%|48.61%|
|tapered_bounded_seed0|87.98%|6.6806|93.66%|3.7257|73.05%|100.00%|
|refreshed_tapered_xyz_seed0|86.89%|6.7986|93.06%|3.7188|72.35%|46.66%|
|refreshed_tapered_bounded_seed0|87.67%|6.8542|93.40%|3.7361|72.09%|100.00%|

Initial XYZ/bounded use identical features/modules/init/draws, differing in the mapping. Relative removes the saturating bound while retaining center/width scaling. Peer boundaries enable cross-query attention in the corridor branch. None receives extra oracle geometry at inference.

Uniform envelope supervision addresses optimistic reference widths around shifted predicted centers. Tapered envelope supervision addresses unreachable cubes around fixed endpoints. Both repairs apply identically to XYZ and bounded arms, retain the160clearance coefficient and 2,400-update schedule, and were registered before outcomes. Region definitions differ: uniform swept boxes versus the tapered reachable set; their feasibility rates are not an invariant common-region metric.

## Shared-head geometric views versus standalone controls

|View, three-seed mean|V8|U8|V4|U4|Known recall|
|---|---:|---:|---:|---:|---:|
|Shared-head projection|86.43%|6.7581|93.06%|3.7211|79.20%|
|Projection, fresh head|86.37%|6.7523|93.08%|3.7222|79.08%|
|Shared-head XYZ center|90.67%|7.0833|93.66%|3.7465|82.83%|
|XYZ center, fresh head|90.93%|7.1100|93.69%|3.7477|82.64%|
|Shared-head bounded center|90.91%|7.1088|93.72%|3.7488|82.86%|
|Bounded center, fresh head|90.99%|7.1100|93.75%|3.7500|82.48%|

Shared views isolate the path construction for identical queries. Fresh-head views additionally account for changed realization labels and query allocation. Main conclusions pass neither center control; inherited mismatched heads cannot explain away the negative result.

## Companion query perturbation

|Arm|Mean companion path motion mm|Mean maximum boundary change mm|Valid companions lost/request|Raw words changed/request|
|---|---:|---:|---:|---:|
|screen_xyz_seed0|2.766|0.000|0.0347|0.0694|
|screen_relative_seed0|2.851|0.000|0.0243|0.0590|
|screen_bounded_seed0|2.838|0.000|0.0347|0.0521|
|screen_peer_seed0|3.952|10.473|0.0729|0.1285|
|refreshed_tapered_xyz_seed0|3.465|0.000|0.0382|0.0486|
|refreshed_tapered_bounded_seed0|2.407|0.000|0.0208|0.0486|
|refreshed_tapered_bounded_seed1|2.423|0.000|0.0174|0.0312|
|refreshed_tapered_bounded_seed2|2.734|0.000|0.0243|0.0382|

One slot is changed while the other seven query/variant identities stay fixed. Net U8 includes the deliberately changed slot and must not be interpreted as companion-only harm. Independent boundaries remain invariant; peer-aware interior parameters can still change geometry and realized words. Returning independent centerlines eliminates these interior changes by construction.

## Width, shape and frozen scorer

|Arm|Width p10/median/p90 mm|Cells below2mm|Mean valid length m|Mean turn rad|Degenerate segments|Missed available valid slots|
|---|---|---:|---:|---:|---:|---:|
|refreshed_tapered_xyz_seed0|17.90 / 37.14 / 56.37|0.00%|0.8288|0.2663|0.0000%|24|
|refreshed_tapered_bounded_seed0|17.98 / 37.11 / 56.35|0.00%|0.8285|0.2743|0.0000%|19|
|refitted_control_center_seed0|17.90 / 37.12 / 56.41|0.00%|0.8558|0.3045|0.0000%|16|
|refitted_control_boundcenter_seed0|17.97 / 37.09 / 56.36|0.00%|0.8556|0.3056|0.0000%|15|

Turning angle describes the delivered polyline, not smooth robot motion. Missed available valid slots uses the checker only after selection; it is an analysis upper bound, never an oracle return rule.
