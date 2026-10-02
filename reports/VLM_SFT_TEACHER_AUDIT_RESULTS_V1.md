# Fixed TRAIN8 SFT teacher-forcing audit: actual results

No coordinate-unit, serialization, supervision-mask, next-token shift or causal-prefix error was found in this bounded audit. The remaining prediction error is concentrated in coordinate tokens, and conditional readout is much better than the already-frozen free generations. **This does not prove correct visual target localization:** every conditional token receives the true preceding tokens, including earlier digits and up to23 true route points that already reveal the target direction. Exposure and representation remain unresolved.

The one authorized run completed from immutable source `1915de7dbe2c8cbc772250e092be87c83c7df786`, unchanged SFT best checkpoint `675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f`. It used exactly the fixed eight TRAIN parents272000–272007, target0, positive-reference index0. It made eight teacher-forced forwards and one late-token causal-control forward, with zero backward passes, optimizer updates or generation requests. No DEV/locked raw data or verification geometry was opened. The loader joined metadata and checked all selected original file hashes before decoding only those eight RGB/current/reference records; it never called the broad TRAIN/DEV route loader.

| Actual result | Value |
|---|---:|
| CPU tests | 10 passed in1.84s |
| Diagnostic forwards / permitted maximum | 9 / 9 |
| Diagnostic body / declared limit | 1.87552 / 60s |
| Internal elapsed including startup | 8.08515s |
| Reserved GPU hours | 0.00224588 |
| Peak CUDA allocation | 4,535,028,736bytes |
| Maximum meter→integer-mm→meter per-axis error | 0.49960mm |
| Maximum per-token-vs-original-loss difference | 4.61e-8 |
| Actual trainable parameters during audit | 0 |

The actual processor prefix was fully masked and its answer token IDs exactly matched the standalone answer-offset mapping. Every supervised token was assigned once; `hidden[t-1]` predicted `label[t]`. Original training helpers matched their recorded byte hashes on the server. No source guard was weakened for local line-ending differences.

## What the token loss measures

| Token category | Count | Mean NLL | Top1 accuracy |
|---|---:|---:|---:|
| Syntax only | 696 | 0.041079 | 98.7069% |
| Coordinate only | 1684 | 0.912570 | 66.0926% |
| Event only | 192 | 0.000969 | 100% |
| Mixed characters | 96 | 0.187972 | 93.7500% |
| Chat termination | 16 | 0.000098 | 100% |
| All supervised tokens | 2684 | 0.590012 | — |

Coordinate-only tokens contribute97.0432% of total NLL. This particular fixed K1 TRAIN/index0 audit is a different distribution from the original mixed K1/K4 DEV selection loss0.436727; these values are not a new train-versus-validation generalization estimate. Syntax/event predictions are easy here, whereas coordinate errors remain substantial. Mixed punctuation/sign tokens remain a separate category instead of being assigned arbitrarily to coordinates.

Of576 coordinate scalars,564 can be read as strict integer values from conditional top1 token fragments and12 are unparseable. Among those564 only, absolute error to the serialized integer label has mean16.842mm, median2mm,95th percentile100mm and maximum800mm;184 are exact. All24 first-point coordinates are exact to their integer labels. These parsed-only error statistics retain their explicit failure denominator; no missing scalar is repaired or replaced by its label.

## Conditional endpoint versus saved free generation

The conditional endpoint is parseable for7/8 samples. Those seven errors to the **same first positive reference** range0.040–2.057cm, mean0.861cm; the eighth endpoint is unparseable. Original grammar-constrained free generations have mean57.398cm error to those same reference endpoints, with all eight retained. This reference metric differs slightly from the previously reported semantic-target mean57.372cm and does not replace it.

| TRAIN suffix | Parsed coordinates /72 | Conditional endpoint error to reference, cm | Saved free endpoint error to same reference, cm |
|---|---:|---:|---:|
| 272000 | 69 | 0.683 | 56.986 |
| 272001 | 68 | 2.057 | 23.497 |
| 272002 | 70 | 1.038 | 89.231 |
| 272003 | 72 | 0.125 | 37.104 |
| 272004 | 72 | 0.964 | 39.241 |
| 272005 | 71 | unparseable | 40.971 |
| 272006 | 71 | 1.118 | 87.257 |
| 272007 | 71 | 0.040 | 84.895 |

This is evidence of a large gap between these conditional readouts and the saved stochastic rollouts. It does not isolate stochastic sampling, accumulated prediction error, insufficient fitting, or the depth representation as the cause. Conditional scalar fragments are not a coherent autoregressive trajectory. The 3750 SFT route slots and only3–8 supervised requests per selected instruction remain far below the same96-parent regression baseline's160000 slots at its selected checkpoint; current results do not establish that SFT is intrinsically unable to solve the task.

## Actual causal control

The last pure coordinate token in the first sample was changed from token20 to15 at input position1113. All1113 earlier positions and all151936 vocabulary logits per position were compared: **169,104,768 values, maximum absolute difference0 and zero tolerance violations**. Earlier hidden states also differed by0. Hidden states at/after the changed token differed by up to9, confirming the token change was not inert. The predeclared tolerance was `1e-3 + 1e-4*abs(original)` on FP32 comparisons of the BF16 head outputs. Exact equality was observed, so the conclusion does not depend on that tolerance. This is a check on this actual call, not a proof for every input or every possible runtime configuration.

## Provenance and reproduction

All21 original server files were synchronized and SHA-verified; see `vlm_sft_teacher_audit_v1/artifact_index.json`. No checkpoint or base weights were copied by the audit. The complete original masks/IDs, scalar readouts, token losses, interface receipts, source hashes and failures are retained. The original output is `runs/vlm_sft_teacher_audit_v1/train8` on the server. Job record PID408998 / child408999 completed exit0 at2026-10-02T15:42:39.845421+00:00; SSH session84542 also exited0 and GPU/CPU1 were released. No automatic retry or additional generation followed.

- Original summary SHA: `8b1a1fed1ad7d23b82178f7bba8143aabe4598a278a48400925237f4b3c195e4`.
- Frozen launcher SHA: `b8578f97ce0942cd58846d399af85415d77873a9f02e7e76e4fcd3235f046bbf`.
- Actual launcher, commands, status and imported-source hashes are preserved under `reports/vlm_sft_teacher_audit_v1/`.
- Independent CPU analysis, already executed locally without raw-data/model access: `python -m scripts.analyze_vlm_teacher_audit --input reports/vlm_sft_teacher_audit_v1/train8 --output <fresh-analysis.json>`.

The audit is complete. Do not repeat it or expand to DEV/K4 on the strength of these results. The next proposed finite diagnostic is a same-checkpoint greedy TRAIN8 control, described in `VLM_SFT_GREEDY_TRAIN8_CARD.md`; it is not yet implemented or run and would be a conventional decoding baseline, not a new mechanism.
