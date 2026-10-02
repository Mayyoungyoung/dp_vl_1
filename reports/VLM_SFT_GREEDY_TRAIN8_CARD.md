# One next control: same-checkpoint greedy constrained TRAIN8

Status: root approved this one control; implementation and local pure tests are ready for source review/freeze. No generation or retraining has run for this control. The existing stochastic TRAIN8 outputs and completed nine-forward audit are retained unchanged.

## Hypothesis and choice

The measured conditional coordinate top1 accuracy is66.09%, with97.04% of total token NLL in coordinates. Conditional endpoints are often close only when true preceding coordinates are supplied. Existing temperature0.7/top-p0.9 free rollouts miss every requested target. One directly testable explanation is that random token selection materially amplifies error beyond the current model's deterministic conditional mode. A same-checkpoint greedy comparison changes one decoding choice and needs eight calls, without confounding another training budget or output representation.

Choose this control before an endpoint-first SFT rewrite. Endpoint-first serialization would make the first target prediction depend on observation alone and could reduce an easy teacher-forcing shortcut, but it changes the sequence distribution and requires a fresh paired training run. Current evidence does not justify attributing the failure to route order rather than insufficient exposure or stochastic rollout. No endpoint-first implementation or second candidate is added now.

## Fixed protocol

- Same original best checkpoint SHA `675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f`, original Qwen revision/processor/LoRA and observation-only prefix. Same registered eight TRAIN parents272000–272007, target0. No teacher-forced answers in generation.
- Exactly one K1 request per parent: eight calls/eight candidate slots, H24, maximum512 new tokens each. Same exact integer JSON grammar, no beam search, retry, reranking, truncation, coordinate clipping or repaired point. EOS remains permitted only after a complete route.
- The only intended change from the previous constrained probe is greedy argmax under the same grammar mask (`do_sample=False`). Temperature/top-p are inapplicable and must be explicitly disabled in the actual generation configuration and reported, not falsely claimed as still0.7/0.9. Model weights and syntax bounds stay identical.
- Preserve input/checkpoint/source hashes, full text, failed slots, token counts and end-to-end request cost including preprocessing, encoding and grammar work. No unseen extra candidate is decoded. Input-byte reuse may verify identity; hidden condition features must not be cached between requests.
- All eight predictions must complete and be frozen before a separate endpoint-only analysis opens the original target metadata. Use the unchanged strict parser and target3cm threshold; no collision or robot execution claim. Compare paired endpoint errors and counts to the existing stochastic TRAIN8 probe; there is one stochastic realization, not a variance estimate.

Compute ceiling: one bounded eight-request run, expected roughly the previous87s including startup, GPU1/35%,CPU1. Each request remains limited to512 new tokens. Set a predeclared180s total request-loop ceiling with boundary checks; already submitted CUDA operations cannot be interrupted and any overrun must be recorded. No new training, checkpoint, warmup, repeat, DEV or K4 expansion. Actual time is unknown until separately authorized and measured after source freeze and tests.

The implemented boundary is before each complete request. If one in-flight request crosses180s, it finishes under its512-token limit and the actual overshoot is recorded; remaining requested slots are saved as unattempted timeout failures with NaN paths, not dropped. Thus the cap is cooperative, not a hard process kill. Requested/attempted calls and all eight charged slots remain separate. No timeout row causes a retry.

## Implementation and checks

The existing constrained-generation entry gets an opt-in `--train8-greedy` switch that rejects DEV, repeats or a different seed/checkpoint. The default sampling kwargs and request seeds remain unchanged. A cloned actual `GenerationConfig` clears temperature/top-p/top-k, sets `do_sample=False`, one beam/one return and512 new tokens. The same object is passed to generation and recorded with its greedy mode; the inherited model configuration is explicitly labeled as not effective.

`use_model_defaults=False` is essential in pinned Transformers4.57.1: its [official config preparation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py) can otherwise replace global-default fields in an explicit configuration with model-specific values, including a saved sampling default. The [official validation code](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/configuration_utils.py) permits unset sampling-only parameters for greedy generation. A separate CPU-only preflight invokes that actual config preparation implementation on the pinned model's config without loading model weights or observations, and rejects any drift from greedy512.

Twenty-four local pure tests passed (six new greedy/paired controls plus eighteen existing probe/grammar checks); the two existing grammar tensor checks and actual pinned-config preflight will run only after root freezes/deploys the source. Tests preserve the old sampling values, exact eight seed-matched calls, timeout/failure denominators, unmodified capped raw text, refusal of effective-config/information drift, and the absence of an all-eight quality mean when any output is invalid.

Paired CPU analysis runs only after both output pools are frozen. It verifies identical model/training config, observation hashes, plan, grammar source/vocabulary and512-token budget, then recomputes unchanged strict parsing and endpoint checks. Its small diagnostic selection criterion is separate from the unchanged3cm metric. No lower finite-only mean can make a run with a missing slot pass the all-eight criterion.

## What would change the decision

Report all paired distances and failed/parseable slots first. Predeclare a useful control result as at least4/8 endpoints within3cm and all-eight mean error at most15cm; this is a small TRAIN diagnostic criterion, not a new benchmark threshold or generalization claim. If reached, greedy decoding is a plausible baseline repair worth later independent validation. A smaller improvement supports only a partial sampling contribution. If all endpoints still fail and error stays near the current57.37cm mean, reject the claim that sampling alone explains current failure; do not add beam width, temperatures or repeated seeds to this control.

Even a strong improvement cannot prove the visual/depth input is correctly used, nor fix the original exposure mismatch. A negative result does not disprove SFT: the coordinate distribution may still be poorly fit. Further exposure or representation controls would require a new evidence-based decision, not an automatic continuation from this card. This is ordinary constrained greedy decoding, with no claim of a new set-generation mechanism.
