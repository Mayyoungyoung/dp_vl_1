# Independent4 format diagnostic — partial comparison snapshot

Only the completed independent4 arm is analyzed here: 96 original K1 requests for all 24 DEV observations, four calls each, one sampling repeat. The same-checkpoint whole4 arm was still running when this snapshot was taken. This is a format-only diagnostic, not a substitute for the complete comparison. No observation, geometry, reference path, semantic label or model was opened or evaluated.

Original generation source: `8356b09b8436d00a4f98a76ffdd8c7d84033e6ca`. Server source journal:

`/home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_autoregressive_v1/best_seed0/repeat0/independent4/requests.jsonl`

SHA-256 before copying, after copying on the server, and for the local unchanged copy all equal `66fe9c78d4326708134e58ad1ccaa24a7e2cdd29634fb59a7febe61dbf000b43`. The copied completed-method summary agrees with the journal. All raw output texts remain unchanged in `requests.jsonl`.

## Actual mechanical findings

21/96 requested slots (21.875%) pass the existing strict format parser. There are no recorded generation exceptions or parsed over-budget candidate pools. This is format validity only; no route-quality claim follows.

| Complete-text result | Requests |
|---|---:|
| Valid JSON, exactly one route, exactly 24 points | 21 |
| Valid JSON, exactly one route, fewer than 24 points | 12 |
| Valid JSON, exactly one route, more than 24 points | 57 |
| Invalid JSON at the 512-token limit | 5 |
| Invalid JSON without reaching that limit (`Extra data`) | 1 |

All 90 valid JSON texts contain exactly one route. Their 2,263 points each contain four strict JSON integers, with coordinates within the existing ±10,000 mm parser range and event value 1. No floats, strings, booleans, missing point fields or invalid event values occur in those parsed texts. The six invalid JSON texts are not repaired or partially scored.

| Observed horizon among valid JSON | 22 | 23 | 24 | 25 | 26 | 27 | 29 | 34 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Count | 1 | 11 | 21 | 25 | 14 | 16 | 1 | 1 |

Thus wrong horizon accounts for 69/75 strict-format failures (92%). All 69 are already complete, parseable JSON; raising the token cap alone would not address this dominant failure category. Output-token counts are min 305, median 342, mean 353.0625, max 512. Five requests reached the cap. The remaining 91 ended with token ID 151645. The original generation summary gives 41.12 seconds median for a complete four-call scene, including all four encodings; no timing is removed.

Constant event value 1 is consistent with the narrow reach serialization but provides no evidence of learned general interaction events. Good syntax likewise does not certify correct units, targets, collision clearance or useful route diversity.

## Research decision and limits

The observed count errors justify testing a **standard syntax-constrained autoregressive baseline**: exact K routes, H24 points, four integers per point, the original coordinate bounds and binary event field. Apply it prospectively to both independent4 and whole4 with the same checkpoint, temperature/top-p and original aggregate token budgets. Its model-logit masking and all grammar processing time must be charged; no output repair, waypoint insertion, resampling, hidden candidate retry or posthoc truncation is permitted. Completion by the token cap remains a recorded failure. This is an engineering control, not the core methodological contribution.

The full original unconstrained whole4 run and independent geometry analysis remain authoritative and must complete unchanged. No conclusion comparing candidate quality between the two schemes is made here.

## Reproduction

`python reports/vlm_sft_independent_format_partial_v1/analyze_format.py`

This standard-library analyzer checks the exact journal SHA, all 96 unique scene/request pairs and original decoding scope, then reads complete JSON texts without repair. `diagnostic.json` contains aggregate counts, per-request hashes/categories/horizons, token counts and the copied-summary/analyzer hashes. Only this directory's journal and completed-method summary are input. No GPU or server computation was used for this diagnostic.
