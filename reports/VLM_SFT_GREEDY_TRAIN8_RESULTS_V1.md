# Same-checkpoint greedy TRAIN8 control: actual results

Greedy constrained decoding reduces mean requested-endpoint error from57.372cm to23.391cm, with lower error for all eight paired TRAIN scenes. **It still reaches0/8 requested targets under the unchanged3cm rule.** All eight outputs satisfy the same H24 format. Random sampling contributes substantially to these errors, but removing it does not establish useful task planning. The predeclared useful-control criterion (at least4/8 within3cm and all-eight mean<=15cm) fails.

This is one fixed TRAIN diagnostic using the original1500-step SFT best checkpoint, not a new training seed, variance estimate or held-out benchmark. No temperature, beam, repeated-seed or K4 search follows this result.

| Fixed parent suffix | Sampling error, cm | Greedy error, cm | Greedy minus sampling, cm |
|---|---:|---:|---:|
| 272000 | 56.931 | 21.463 | -35.469 |
| 272001 | 23.484 | 20.481 | -3.003 |
| 272002 | 89.372 | 41.925 | -47.448 |
| 272003 | 37.087 | 21.634 | -15.453 |
| 272004 | 39.231 | 25.388 | -13.843 |
| 272005 | 40.822 | 7.650 | -33.172 |
| 272006 | 87.179 | 21.577 | -65.603 |
| 272007 | 84.865 | 27.012 | -57.854 |
| Mean | 57.372 | 23.391 | -33.980 |

Greedy median error is21.605cm, minimum7.650cm and maximum41.925cm. Both compared pools contain all eight requested slots and eight finite strict-format outputs, so these means omit no failures. They use original evaluation-only semantic target centers, unlike the separate teacher-forced audit's first-reference endpoint distances.

## What was actually changed and verified

Both runs use the exact best checkpoint SHA `675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f`, identical observation hashes, fixed TRAIN8 target0 plan, Qwen revision/processor, grammar source/vocabulary, route budget and512-new-token cap. The paired analyzer verified these receipts before opening target metadata. It re-parsed the unmodified texts and checked equality with the saved prediction arrays. No reference route or verification geometry was used by this generation or its endpoint analysis.

The new run uses `do_sample=False`, one beam/one return, temperature/top-p/top-k unset, and `use_model_defaults=False`. The actual pinned-HF CPU preflight and recorded effective generation configuration both report `greedy_search`. This prevents inherited model defaults from silently restoring sampling. It is a standard decoding control, not a learned mechanism or a method contribution. The old default temperature0.7/top-p0.9 code path and old results remain preserved.

| Actual budget / validation | Value |
|---|---:|
| Fixed-source CPU tests | 26 passed in1.46s |
| Actual HF config preflight | Passed;0 forwards/0 raw inputs |
| Requested / attempted / charged candidate slots | 8 / 8 / 8 |
| Format-valid / exceptions / capped outputs | 8 / 0 / 0 |
| Unattempted time-budget slots / overshoot | 0 / 0s |
| Request-loop elapsed / boundary limit | 80.2598 / 180s |
| Elapsed including startup | 83.3914s |
| Reserved GPU hours | 0.0231643 |
| Peak CUDA allocation | 4,410,250,240bytes |
| Median complete request | 9.7972s |
| Prompt / output tokens | 6194 / 2656 |

Vocabulary compilation0.6615s and total grammar-logits-processor Python time4.3669s are already inside the measured requests; do not add them twice. No cached Qwen features or hidden extra candidates were used. The source boundary checks permit an in-flight request to finish and would retain remaining timeout slots; this run never reached that boundary.

## Evidence-driven decision

The earlier fixed teacher-forcing audit found no unit, mask, shift or tested-causal-prefix bug. Its coordinate tokens still have NLL0.9126 and top1 accuracy66.09%, contributing97.04% of total NLL. Apparently good conditional endpoints consume up to23 true prior route points and cannot demonstrate visual target grounding. Together with only3750 route slots and3–8 exposures for each selected instruction, these observations leave substantial spatial fitting and training-exposure concerns.

Prioritize a bounded continuation of the **same** training representation and objective before changing to endpoint-first serialization. This would test whether additional optimization/exposure helps while preserving a meaningful link to the original checkpoint. It is not evidence that more training is guaranteed to work. Endpoint-first would simultaneously change target order and the difficulty of teacher-forced prediction, making the present undertraining explanation harder to separate. Neither choice is a core-method novelty.

Root approved preparing, not launching, one explicit continuation to a declared total-step ceiling in a new output tree, restoring complete optimizer/scheduler/sampler/RNG/history and accounting for original versus added exposure. Original1500 outputs and normal strict resume semantics must remain unchanged. After that bounded run, only one fixed TRAIN8 greedy diagnostic is planned; any DEV expansion needs evidence. No additional generation or training has occurred as part of the present report.

## Reproduction and artifacts

Immutable source `2c47c2a4c354ecfeb2f73b8121476f5000e82a98`; original server directory `runs/vlm_route_greedy_train8_v1`. Job record PID424355 / child424356 completed exit0 at2026-10-02T16:18:14.811691+00:00. The independent CPU analysis completed exit0 at16:18:15.019963UTC. SSH70601 exited0; GPU/CPU1 were released. The frozen launcher contains the exact tested commands and source guards.

- Launcher SHA: `5c1ec0da5bc191faf30f835da175a37c7afd1c70b36943d85a465f05c725651e`.
- Generation summary SHA: `228506dbb94b80cad93ce5cc9d23c3841123447b43fbb52dade7963a5f1cc4c2`.
- Paired analysis SHA: `9a714fa1d563c47d67a12994fca4336badf85937518c0f0bf0e3db0a4d5f8c69`.
- Prediction NPZ SHA: `923e5e88aeff1b1b6d9183e151760131b478c90386f3d8a396bf4d683ac29507`.

All21 server files were copied and hash-verified; `reports/vlm_route_greedy_train8_v1/artifact_index.json` lists their provenance. The prediction NPZ remains in ignored local/server `runs/`; metadata, raw texts, status, config, source hashes and analysis are retained in the report directory. No base-model or checkpoint copy was created.

The actual frozen launcher first ran26 tests and `scripts.preflight_vlm_greedy_config`, then `scripts.evaluate_vlm_route_sft_constrained --scope train8_preflight --train8-greedy`, then `scripts.analyze_vlm_greedy_train8_pair` against the original stochastic probe. Inspect the saved launcher for absolute paths and environments; do not relaunch the completed experiment.
