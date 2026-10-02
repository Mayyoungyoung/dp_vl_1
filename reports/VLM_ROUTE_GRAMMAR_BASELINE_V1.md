# Syntax-constrained SFT baseline and bounded TRAIN probe

This is a standard engineering repair for the direct-VLM baseline. Exact JSON syntax/count constraints, finite-state token masking and caches are not claimed as methodological contributions. No training changes, checkpoint changes, geometric corrections or posthoc route repairs are made.

## Evidence and current decision

The original unconstrained comparison is complete and preserved. Its independent4 journal has 69 complete JSON texts with wrong waypoint counts, five cap-truncated texts and one other JSON error; only21/96 slots pass strict format. The full whole4 arm likewise has substantial format failures. A separate, unmodified-text endpoint diagnostic finds that **all**90 independent and30 whole routes with complete JSON end more than3cm from their requested target: minimum9.70/12.99cm and candidate-mean59.46/71.58cm. Wrong-horizon JSON is included only in that descriptive endpoint diagnostic and remains failed under the original main evaluation.

Thus grammar is justified to remove a format bottleneck, but it is not evidence for correct geometry or conditioning. The next authorized model run is limited to **eight registered TRAIN parents272000–272007, target0 only, one constrained K1 request each**. This is eight calls/eight requested slots, at most512 output tokens per request (4096 total), seed0. It is a training-capacity/format diagnosis, not an independent4-versus-whole4 comparison or DEV result. Do not launch the full24DEV constrained comparison unless evidence and root's GPU queue decision justify it.

## Mechanism and exact constraints

`routeset/vlm_route_grammar.py` accepts canonical compact JSON matching the existing serialized training format: exactly K routes, exactly24 points each, exactly four integers per point. Coordinates remain model-generated in the existing parser range[-10000,10000]mm; event is0 or1. No scene geometry, target position, reference route, workspace-derived coordinate limit or type label enters the grammar. The only permitted syntax alphabet is `[],-0123456789`; whitespace is not emitted in this compact-format control. There are no explanations or additional candidate slots.

The finite state includes route/point/field counters and the current signed integer prefix. Each proposed tokenizer token advances the state through **its complete string**, including multi-character numbers, punctuation and fragments crossing field boundaries. This is not character-by-character model sampling. For the pinned Qwen ByteLevel tokenizer, the compiler verifies the decoder type, excludes special/added tokens, checks every retained token's actual decoded fragment and verifies concatenation stability. All individual grammar characters must be available. Unsupported decoders or a dead end fail explicitly.

The custom `logits_processor` sets invalid-token logits to negative infinity and preserves every legal model logit exactly. EOS is available only after the entire requested set closes. NaN/+infinity or no finite permitted logits cause a recorded failed request; no fallback token is fabricated. Negative infinity on individual already-masked tokens is preserved. The original temperature0.7/top-p0.9/top-k0 sampling then chooses among legal model probabilities. The original max-new-token cap remains; an incomplete capped prefix is saved unchanged and fails the same parser. No point completion, resampling, candidate truncation, retries or reference substitution is allowed.

Use the original SFT best checkpoint SHA `675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f` and Qwen revision`89644892e4d85e24eaac8bacfd4f463576704203`. Input helper hashes must still match the completed training record. No adapter/base parameter is updated.

## Timing and source separation

The grammar vocabulary is compiled during the **first actual measured request**, not during an uncharged warmup. Transition-mask caches are bounded to4096 CPU states per K and reused across that process; cache hits/misses and compilation time are recorded. Every request also records processor-call count and Python wall time. These component counters do not replace the outer CUDA-synchronized request latency, which includes image read/processing, full Qwen encoding, autoregression and grammar work. The independent4 mode still performs four complete encodings. No cached Qwen features are used.

The new entry `scripts/evaluate_vlm_route_sft_constrained.py` requires an explicit scope. `train8_preflight` uses the separate observation-only `routeset/vlm_sft_train_probe.py` driver and hard-checks the original eight-parent sampling table and16 current RGB/observation hashes. It does not relabel TRAIN as DEV to reuse the old evaluator. All eight raw outputs freeze before any semantic metadata may be read. `dev24_comparison` retains the original complete two-method driver and is implemented but not currently scheduled.

## Tests and preflight sequence

Local pure tests:18 passed in0.40s; the Torch test module is skipped only because the local Python has no Torch. These cover every two-fragment split of valid K1/K4 strings, arbitrary multi-character chunk sizes, exact coordinate/event/shape rejection, sign/leading-zero behavior, EOS/completion, cache bounds, dead ends, token history integrity, ByteLevel fragment checks, and exact eight-call TRAIN failure/cap accounting with no retries. Two actual tensor-mask tests must run in the existing server `.venv`; they verify unchanged allowed logits, excluded EOS before completion, unchanged input logits, nonfinite rejection and batch-expansion rejection.

Before the GPU probe, run `scripts/preflight_vlm_route_grammar.py` in the existing `.venv-qwen` with CUDA hidden/CPU1. It loads only the pinned tokenizer and checks synthetic full K1/K4 H24 sequences at numeric bounds and with varying coordinates/events; no observation, path label or foundation-model weights are loaded. Actual tokenizer acceptance is required, not inferred from fake-tokenizer tests. Keep any failure and fix into a new immutable release before retrying.

After root freezes the source, example command bodies (each through `record_job` and a frozen launcher):

```bash
# CPU only, existing environments; no package installation.
CUDA_VISIBLE_DEVICES=-1 "$P/.venv/bin/python" -m pytest tests/test_vlm_route_grammar.py tests/test_vlm_route_grammar_torch.py tests/test_vlm_sft_train_probe.py -q
CUDA_VISIBLE_DEVICES=-1 "$P/.venv-qwen/bin/python" -m scripts.preflight_vlm_route_grammar --training "$P/runs/vlm_route_sft_v1/seed0" --output "$P/runs/vlm_route_grammar_v1/tokenizer_cpu"
# Root owns the separate GPU1/35% launch; this script is not a launch authorization.
"$P/.venv-qwen/bin/python" -m scripts.evaluate_vlm_route_sft_constrained --checkpoint "$P/runs/vlm_route_sft_v1/seed0/best.pt" --scope train8_preflight --seed 0 --repeats 1 --output "$P/runs/vlm_route_grammar_v1/train8_k1"
```

After all eight predictions finish, report unchanged H24-format outcomes and endpoint errors separately. If TRAIN endpoints remain poor, stop expansion to DEV and investigate training exposure/spatial representation; syntax alone has not repaired planning. If TRAIN improves, that is still memorization/capacity evidence and does not establish generalization.

## Primary API verification

The pinned [Transformers4.57.1 generation code](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py) merges custom processors before temperature/top-p warpers; [the logits-processor contract](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/logits_process.py) returns modified vocabulary scores. The official [Qwen tokenizer implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen2/tokenization_qwen2.py) uses ByteLevel BPE; the actual fast tokenizer is additionally checked at runtime. This is API/source inspection, not a third-party constrained-decoding system reproduction.
