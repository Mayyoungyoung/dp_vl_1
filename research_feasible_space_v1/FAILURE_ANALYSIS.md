# Failure analysis

The representation is expressive under reference geometry but does not establish useful learned interior generation. The following mechanisms are supported by actual outputs; associations are separated from causal controls.

## What the controlled comparisons establish

1. Reference centerlines already solve the oracle diagnostic. Allthree approaches are100%valid; the bounded decoder does not add oracle capability and occasionally changes the requested word.
2. Merely giving XYZ the same predicted geometry produces most of the task quality. Bounded mapping adds only0.08565U8 over XYZ, below the registered0.15gate.
3. Removing interior generation is a stronger control. Fresh-head centers gain0.30671U8 over bounded mapping; the crossed interval excludes zero. Geometric containment alone cannot establish neural necessity.
4. Boundary independence is architectural. Center-only generation shares it, so invariant boundaries are not an independent benefit of the relative neural decoder.
5. Repair gains trade against retention. Boundedseed0 repairs224/270 but retains1727/2169, below historical1872/2169. Center-only repairs243/270 and retains1760/2169. Neither supports a claim that the proposed mechanism preserves prior feasible options.

## Where route validity fails

|Saved seed0 output|Certified + inside|Geometry violations of implication|Endpoint failures within certified cell|Wrong center raw word|Output word changed from center|
|---|---:|---:|---:|---:|---:|
|refreshed_tapered_xyz_seed0|783|0|49|108|64|
|refreshed_tapered_bounded_seed0|1661|0|78|103|59|
|refreshed_tapered_projection_seed0|1667|0|81|108|63|
|refreshed_tapered_boundcenter_seed0|1661|0|78|103|0|

Zero certified-and-inside geometry failures independently supports the conditional geometric implication on saved outputs. It says nothing about uncertified predictions, goal correctness or passage identity. Boundedseed0 has643/2304uncertified regions and78goal failures inside certified regions. There are103wrong center raw words and59word changes induced by the output mapping. These categories overlap and are not additive failure partitions.

## Measured-cause repairs

The initial XYZ weighted clearance gradient norm0.22190 is about11.1times its route gradient0.01997. Final-path clearance does not directly supervise the whole cell. Reference-centered widths are optimistic after center prediction errors; whole-envelope supervision raises cell feasibility from7.55%to71.18%. This is a geometric supervision improvement, not proof of task improvement.

Uniform endpoint cubes create pressure on regions the fixed-endpoint decoder cannot visit. XYZ goal failures rise137→172 while geometry failures fall185→167. Tapered endpoint cells remove this unreachable volume and restore some route quality, but the final three-seed comparison remains below center-only and historical references.

## Observation and deployment limitations

Single-view depth leaves unknown space. About9.23%of bounded node-corner probes are unknown, with nearly identical fractions for valid and invalid routes. Sparse probe visibility does not explain or certify all region errors. The implementation does not learn calibrated cell confidence, abstain on missing evidence or explicitly model hidden occupancy; the public API labels actual feasibility unknown.

Adjacent regions share a node cube and are structurally connected, including zero-width endpoint nodes. This avoids a separate empty-intersection construction failure but cannot make a colliding or wrong-mode corridor physically valid. The23degree-one Bézier pieces are precisely the24-node public polyline; they provide positional continuity only. No smoothness/dynamics/full-arm guarantee is inferred.

The frozen scorer may miss available valid routes; the ablation table records this separately. It cannot repair a weak candidate pool and was not retrained using evaluation truth. The API timing uses cached frozen VLM features, not full online RGB-D-to-result latency.

## Failures retained

Failed server test on Sequential.weight access (test bug); failed oracle diagnostic JSON serialization (saved paths retained); failed fresh-family prepare before data creation (inherited uniqueness check before new geometry edits). Each correction uses new source/output/job IDs. Pre-launch release path typos and local test-discovery/rg glob errors are command mistakes, not model outcomes. No failed experiment is relabeled successful.

Older figure v1 renders an oversized uniform envelope and picks a different comparison seed. The delivered actual_reachable_figures_v2 uses true tapered reachable cells and actual paired seed0 XYZ/bounded outcomes, retaining both improvement and loss cases. Visual examples never replace aggregate gates.

## Research decision

Do not extend interior-decoder loss searches or claim a new corridor-planning principle from these results. A further mechanism would need to solve observed region/goal/mode prediction errors and justify why simple centers cannot deliver the same ability. Fresh-family results are frozen diagnosis; they are not permission to optimize against a new development set. TEST_LOCKED and whole-arm validation remain unopened. The publication-potential goal is unmet.
