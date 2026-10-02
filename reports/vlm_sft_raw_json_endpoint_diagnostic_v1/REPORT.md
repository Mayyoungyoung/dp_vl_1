# Complete-JSON endpoint diagnostic

Both original autoregressive arms finished before this analysis. Their original raw journals were verified against the completed generation summary. Evaluation-only supervision metadata has original SHA `37da38855af0357732ac48f125f140e8ed5d1a60008f1424fd5080c7a9421237`. No raw reference path, obstacle verification file or model was opened.

The analysis extracts the final point from each complete JSON route without changing any text, waypoint or candidate count. Wrong-horizon/missing-candidate outputs remain failures under the original main protocol. These additional endpoint descriptions do not replace any strict H24 metric.

| Complete JSON routes | Independent4 | Whole4 |
|---|---:|---:|
| Requests with valid complete JSON | 90/96 | 15/24 |
| Routes with parseable integer points | 90 | 30 |
| Scenes represented | 24 | 15 |
| Minimum requested-target distance | 9.697cm | 12.988cm |
| Median distance | 43.139cm | 63.212cm |
| Mean over candidates | 59.455cm | 71.584cm |
| Maximum distance | 207.934cm | 157.329cm |
| Within the original3cm target tolerance | 0 | 0 |

All90 independent valid JSON requests contain one route. Among15 whole valid JSON requests, ten contain only one route and five contain four; thus the latter totals30 routes. Malformed JSON is not salvaged. Among the wrong-horizon routes, candidate-mean target error is51.593cm/70.885cm. Among exact-H24 routes it is85.287cm/75.083cm. The original main report averages per-scene evaluable errors, so its92.8cm/105.9cm values are a different aggregation and remain unchanged.

The format defect is substantial, but accounting for otherwise parseable wrong-length outputs does not reveal correct endpoint behavior. A standard grammar control may improve shape compliance; it cannot be presumed to solve this semantic/spatial error. The next authorized model check is a bounded eight-TRAIN K1 constrained probe, not another full DEV comparison.

Reproduce with `scripts/diagnose_vlm_raw_json_endpoints.py`, the complete copied generation directory, and the unchanged supervision metadata. `summary.json` records all source hashes, per-route raw counts/endpoints, aggregate definitions and exclusions. This used local CPU only, with no additional model requests or changed main scores.
