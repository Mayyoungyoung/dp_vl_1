# Versioned recovery of the two hash-blocked TRAIN parents

Only the original TRAIN parents **layout_variation_401005 and layout_variation_401007** may acquire new attempts. Their configurations, seeds, colors, guide plans, canonical robot initialization, obstacle values, target values, type definition, state restoration, full-body checks, 2cm tip checks, 3cm target checks, and route resampling remain unchanged. The original ten successful parents are not replayed. Original failures and all 54 unattempted slots remain in v1.

This is a numeric identity repair, not a method contribution, better geometry, extra training, or a new success criterion. The existing source `eba09945ecade4bdb9a5b4ab123a92da898283ed` and its dataset are immutable.

## Change and empirical prerequisite

The old 1mm hash quantizes floating meters directly. At nominal ±.0275m, actual readback ±.027499999850988388m falls on the other side of a half boundary despite a 1.49e-10m physical difference. The new helper first rounds to integer micrometers, then applies round-to-even integer millimeters. It preserves sorted arrays, legacy four-box padding, variable-N schema, integer byte order, and SHA256. Raw exact hashes remain unaltered and separately recorded.

The physical readback gate stays **atol1e-6m, rtol0**, before hashing. Actual physical arrays are never rewritten. A scoped project-local validator is installed only while delegating to the original worker and is restored in `finally`; no installed library or original source file is patched. Route acceptance and type logic remain the original functions. This convention is not claimed to eliminate every possible quantizer boundary for arbitrary future layouts.

The local proposal's 19 tests verified all twelve saved readbacks. The production implementation's new27 + original21 tests actually passed locally (48 total, zero skips, 3.47s). The fixture contains only already-authorized TRAIN geometry and source hashes. The original ten actual hashes and all twelve registered hashes remain unchanged; only two failed readbacks gain the intended identity. Server tests must pass separately before preparation/collection.

## New lineage and inherited budget

New output roots:

- `/home/wzy/dpvlm/route_set_v1/data/observed_layout_variation_hash_recovery_v2`
- `/home/wzy/dpvlm/route_set_v1/runs/observed_layout_variation_hash_recovery_v2`

Each of 54 new possible attempts has a distinct v2 attempt ID mapped to its original unattempted v1 slot. Original parent IDs and TRAIN role remain identical, preventing cross-split reuse. Registered old plans are copied without changing their config fields. The combined record has 324 original proposal slots plus54 explicit recovery slots; it does not erase original unattempted slots or claim378 independent scenes/routes.

The original twelve closures consumed **1882.7102640580852s of coordinator-measured worker time**. Their exact original closure hashes and this sum are frozen in the new config. The same cumulative45min **parent-boundary soft cap** leaves **817.2897359419148s** before any recovery worker. Each completed new worker subtracts its actual elapsed cost. A worker already started can exceed the soft cap; no further parent starts once cumulative cost reaches2700s. This is not a fresh45min allocation or a hard per-parent timeout. Wrapper/display startup overhead is recorded separately and must be included in overall wall cost without adding nested worker scopes.

The new coordinator records each child through record_job. Completed parents are skipped on explicit resume. Existing interrupted data/status are closed without repeating their slots; unknown interrupted cost blocks another parent until manual review. Workers independently check inherited cumulative cost and reject an already closed or unselected parent. A known runtime failure closes that parent and stops automatic continuation. No retry, extra-layout stage, or certify command exists.

## Explicit execution recipe, after a new immutable release is frozen

`scripts/launch_observed_layout_hash_recovery_v2.sh <40-character-commit> <stage>` accepts separate `tests`, `prepare`, and `recover` stages. The root operator must read each completed stage before invoking the next; the launcher never chains them. It validates the same-source actual48-test JUnit receipt (27 new cases, zero skipped/failed), source/config/fixture hashes, and completed preparation before collection.

The launcher pins **CPU3**, hides CUDA, sets all native threads to1, uses the existing `.venv` coordinator and `.venv-sim` worker, and starts/cleans up only its own software-rendered Xvfb process. Parent-level existing disk-budget guards remain. Immutable source path and actual wrapper hash are recorded. An explicit administrative continuation, when permitted, uses:

```text
bash <release>/scripts/launch_observed_layout_hash_recovery_v2.sh <commit> resume recover_resume_<new-id>
```

This does not replay an interrupted parent. Old corpus registration/source and all twelve closure hashes, together with the selected failures' summary/attempt evidence, are checked before every stage and worker. New data, closures, lineage, identity audit, process state, logs, exit, and costs live only in the v2 roots.

Successful initialization would only open the existing route-attempt budget. It does not establish an open/closed-route certificate; an actual collected positive witness and original relation checks are still required in a separately authorized analysis. No automatic certification, model training, next dataset stage, or simulator execution has occurred during implementation.
