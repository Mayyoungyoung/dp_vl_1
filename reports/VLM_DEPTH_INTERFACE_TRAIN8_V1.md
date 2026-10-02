# Direct-VLM depth interface: predeclared TRAIN-only audit

Status: four fixed-source CPU tests passed; the first actual audit stopped at its metadata hash guard before opening raw observations or loading the processor. A minimally corrected plan is awaiting a new freeze. No training/evaluation source is modified. This is a diagnostic of the SFT input representation, not a new method or a claim that the representation caused a model failure.

## Fixed sample and information boundary

`configs/vlm_depth_interface_train8_v1.json` fixes exactly parents `obstacle_reach_272000` through `obstacle_reach_272007`, each lexicographically first observation (`target0`). It records the original SFT hashes for current `front.png` and `observation.npz`. All eight parents are registered TRAIN. Missing/changed inputs stop the job; there is no replacement parent.

The entry point reads the original SFT config/hash index and observation manifest metadata, then only those eight current RGB-D/calibration/gripper observations. It never opens a supervision manifest, future path, target/verification geometry, DEV raw input, or locked raw input. Hash-index path strings are metadata only. No foundation-model weights, adapters, generated routes or VLM forwards are used. The runtime must hide CUDA and use one CPU thread.

## Actual pipeline and comparator definitions

Use the original Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`, Transformers `4.57.1`, Torch `2.4.1`, `min_pixels=max_pixels=65536`. Call the unchanged SFT `prepare_prefix` with RGB followed by the depth image, K1/H24. Thus grouping and image processing match the training interface; the text prefix contains only observations.

The depth representation encodes metric millimeters into `(high byte, low byte, validity byte)`, with blue 255 for observed positive finite depth and zero for unknown. It rounds raw metric depth to millimeters before image processing. The frozen vision tower is not an analytic decoder for this custom format.

1. Recover processed RGB channels from the actual depth `pixel_values` by reversing the pinned patch permutation and affine normalization. Check repeated still-image temporal patches agree. Independently apply the actual processor's bicubic uint8 resize; recovered channels must match within 0.001 channel units. This verifies the audit's inverse, not a learned model capability.
2. Record raw depth versus raw byte decoding quantization error, raw/processed image sizes, channel distance to integer, and blue counts at 0, 255, or intermediate values. Decode recovered rounded bytes using the original formula. Also retain unrounded channel decoding so normalization roundoff is visible.
3. At the exact processed output size, compare decoded depth with (a) float32 metric depth resized by the same processor bicubic routine and (b) nearest-exact sampling of raw metric depth. Also compare (a) with (b). This separates the byte-format/nonlinearity discrepancy from ordinary spatial resampling; neither comparator is physical ground truth at newly interpolated pixels.
4. Report count/mean/median/p95/max absolute differences plus predeclared >1 mm, >1 cm and >10 cm counts. Preserve per-parent arrays and actual normalized depth patches in NPZ, source/runtime hashes, per-parent JSON, elapsed time and input hashes. No fitted threshold or model-selection rule is introduced.

Depth invalid entries are filled with zero solely to define the diagnostic resize comparator. Error summaries use blue=255, nearest-valid, and bilinear valid support >=1-1e-6. Bilinear support does **not** certify the wider bicubic stencil; therefore raw-invalid count and `all_raw_pixels_valid` accompany each result. With invalid inputs, these are descriptive image operations, not certified geometric error. Unknown pixels are never labeled free space.

## Interpretation and falsification

If raw byte encoding is accurate but processed decoding differs materially from the float-metric comparator, exact byte semantics have not survived that image interface. High/low byte boundaries, independent channel rounding/clamping, and validity interpolation are plausible observable sources. If the measured differences are negligible, reject that particular representation-distortion hypothesis for these eight inputs.

Reversing patch permutation/normalization recovers the **processed** image, not pre-resize samples. Interpolation with quantization/clamping does not generally offer an inverse restoring original millimeter values. This audit stops before the learned ViT patch embedding: it cannot establish what depth or spatial information the pretrained features retain, whether the language model decodes the custom bytes, or whether changing depth encoding would improve generation. Those would require separate finite matched condition tests or training controls. Do not change ongoing SFT/evaluation behavior based on this audit.

## Validation and reproduction

Local NumPy/Pillow tests: four passed in 0.39 s. They verify the patch/normalization inverse, identity quantization and unknown handling, a clearly labeled synthetic byte-carry/intermediate-validity case, and exact eight-TRAIN first-observation selection with role/count rejection. These tests do not claim actual processor execution.

Actual first execution from `6464b3fb329704a99d8f9076dd6a8c5ac6821363`: all four tests passed in 0.12 s. The subsequent `.venv-qwen` audit failed at `Registered parent reservation changed` before Torch import or any raw observation read. Its plan incorrectly expected the LF Git blob SHA `be14ab549554e587bfd77c18172e5ba4b187dca41d4e67efd4889e1f4978bf5a`; the actual deployed reservation preserves CRLF and has SHA `d9a8112d6f68efd13e785134fcd5435306e4e570fefbaf1c6c31468a9a8ca2e8`, identical to the original export receipt. Converting that Git blob's LF to CRLF reproduces the deployed hash exactly. The minimal correction changes only this expected byte hash. The original release, launcher, `runs/vlm_depth_interface_train8_v1` statuses and failure log remain unchanged; a retry requires a newly frozen plan and distinct job/output.

After root freezes a new immutable release, run the tests in existing `.venv` and the actual processor in existing `.venv-qwen` (`pytest` is absent there). Both jobs require `record_job`, GPU hidden and CPU1; do not install packages or launch a model. Example argument body, with `$SOURCE` set to that newly frozen release:

```bash
cd "$SOURCE"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=/home/wzy/dpvlm/route_set_v1
"$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/vlm_depth_interface_train8_v1" --run-id tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_vlm_depth_interface.py -q
"$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/vlm_depth_interface_train8_v1" --run-id actual_processor --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.audit_vlm_depth_interface --training "$P/runs/vlm_route_sft_v1/seed0" --output "$P/runs/vlm_depth_interface_train8_v1/actual_processor"
```

Use a frozen launcher, record actual source hashes and `CODE_COMMIT`, and require fresh output. The actual audit has no resume, no GPU hours, no parameter updates and exactly eight observations. A failed implementation guard is a recorded audit failure, not permission to silently substitute another processor.

The pipeline comparison follows the pinned [Qwen image-processor implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py) and its [resize/normalization implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/image_processing_utils_fast.py). These are inspected primary sources, not reproduced model-quality claims.
