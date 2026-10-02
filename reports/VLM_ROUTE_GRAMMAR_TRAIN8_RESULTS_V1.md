# Direct-VLM grammar TRAIN8 probe: real results

All eight requested paths now satisfy the unchanged H24 JSON format, but **none reaches the requested target within the original3cm tolerance**. This fixes one interface defect; it does not establish task planning. Following the predeclared decision, no constrained K4 or additional DEV run is launched.

Source `0eeeecbe3d02adac413083a9edc0df674aa53be2`, unchanged actual SFT best checkpoint `675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f`. Exact TRAIN parents272000–272007, each target0, K1 once, seed0, temperature0.7/top-p0.9, max512 tokens per request. No repeated trials, hidden candidates, output repair or new training occurred.

| Actual outcome | Value |
|---|---:|
| Completed requests / requested slots / charged slots | 8 / 8 / 8 |
| Strict H24 finite-format paths | 8/8 |
| Request errors / capped outputs | 0 / 0 |
| Requested endpoint within3cm | 0/8 |
| Minimum / median endpoint error | 23.484 / 48.877cm |
| Mean / maximum endpoint error | 57.372 / 89.372cm |
| Internal elapsed, including model startup | 87.1006s |
| Reserved GPU hours | 0.0241946 |
| Peak CUDA allocated | 4,410,898,944bytes |
| Median complete K1 request | 10.3273s |
| Actual prompt / output tokens | 6,194 / 2,683 |

The first measured request includes0.7571s grammar-vocabulary compilation. Summed logits-processor Python wall time is4.4122s; it is a component already inside total request timing, not an extra cost to add again. Learned Qwen features are not cached. Every path was generated from model logits under the original observation-only prefix and universal syntax/numeric bounds.

Before any model run,20 fixed-source CPU tests passed in1.60s. The actual pinned `Qwen2TokenizerFast` CPU preflight passed four synthetic complete K1/K4 H24 cases in1.0517s internal elapsed. It found70 supported ASCII fragments and checked exact decode concatenation; its fragment SHA is `2d71c22129e9964d14547e33b91cde382881ebf5009ddb2b1acb344be1ca4cb0`. Synthetic cases were never model inputs or robot results.

All eight generation outputs completed and their journal/NPZ/probe-summary hashes were verified before the independent analyzer opened original evaluation-only target metadata. It re-parsed unmodified texts with the same strict parser and verified equality to the saved predictions. No raw reference path, obstacle verification geometry or additional model call was used. This is a TRAIN endpoint diagnosis only; collision clearance, arm execution and generalization are unmeasured here.

The 15 original server artifacts were synchronized and SHA-verified: [artifact index](vlm_route_grammar_v1/artifact_index.json). Generation summary SHA `530c7621ff507669bef53252ccab31f8cf29d300347702c18722494869175e94`; raw requests SHA `1253e924ab680ed41e066647ddd05c1785579619faf2e8651ef8b1baac14d03f`; independent diagnostic summary SHA `651360bedc45bda288694ceae27123816c1839c29a70d218659721bd8e35bbc9`. Actual record PID391987 / child391988, exit0 at2026-10-02T15:10:26.495523+00:00. Frozen launcher SHA `db1c9e1a0517bed91b27824e30a2500e49b8a5e7403a04d8511d78e96ccaac2b`. GPU/CPU1 were released immediately afterward.

## Exposure boundary and next decision

The original SFT journal shows each selected target0 was used3–8 times, with only1–5 K1 exposures. The counts below are actual supervised requests, not generated trials:

| TRAIN parent suffix | Requests | K1 | K4 | Supervised route slots |
|---|---:|---:|---:|---:|
| 272000 | 8 | 5 | 3 | 17 |
| 272001 | 5 | 2 | 3 | 14 |
| 272002 | 4 | 2 | 2 | 10 |
| 272003 | 4 | 3 | 1 | 7 |
| 272004 | 7 | 5 | 2 | 13 |
| 272005 | 5 | 3 | 2 | 11 |
| 272006 | 3 | 1 | 2 | 9 |
| 272007 | 7 | 4 | 3 | 16 |

These instructions were observed during training, but that does not prove fitting. Total SFT exposure was3750 route slots. For the same96-parent obstacle regression baseline, the actual3000×32×4 run used384000 slots and its best1250 checkpoint corresponds to160000 slots. At1500×32×4, a regression run would use192000 slots. Thus equal step counts are not equal exposure, and current evidence cannot establish that SFT capacity is intrinsically inadequate. Architecture/input representation, optimizer and training costs also differ.

Next proposal is one bounded TRAIN-only teacher-forced coordinate audit, detailed in `VLM_SFT_TRAIN_DIAGNOSTIC_CARD.md`: verify unit/mask/shift correctness and separate syntax/coordinate/event NLL, then compare teacher-forced coordinate errors with these already-saved free generations. No long training or additional DEV generation is authorized by this report.
