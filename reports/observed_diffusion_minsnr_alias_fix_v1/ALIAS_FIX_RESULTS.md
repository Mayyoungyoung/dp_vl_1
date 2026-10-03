# Min-SNR CPU fork test: checkpoint alias diagnosis and minimal repair

The two failed fork/resume tests were caused by shared CPU Adam state in the test's reused parent checkpoint. They were not evidence of a stochastic resume mismatch, and no real-data Min-SNR training had started. The original 75586c5 failure is preserved as `original_failed_tests.log` and its original status.

On pinned Torch 2.4.1+cu121, `optimizer.load_state_dict` retained all six CPU tensor storages from this synthetic parent: `linear.weight` and `linear.bias`, each with `step`, `exp_avg`, and `exp_avg_sq`. The four-step branch therefore advanced the saved parent's optimizer while leaving its saved model and RNG unchanged. The split branch then restored an inconsistent mixture. The resulting mismatches were exactly `model`, `optimizer`, and `gradient_audit`; scheduler, RNG, all sampling streams, counters, and request journal still matched. The Adam step difference was 4, and the maximum weight differences were 0.0009380616 (uniform) and 0.0007981360 (Min-SNR).

The independent diagnostic used the unmodified immutable 75586c5 source. For each loss, its shared-parent case reproduced failure and its independently copied-parent case restored every state field, draw/noise chain, and journal exactly. The full per-tensor differences and storage checks are in `diagnostic/analysis/DIAGNOSIS.json`. No tolerance or digest scope changed.

The repair is one semantic line: `restore_components` deep-copies the optimizer state before loading it. This makes the helper preserve its checkpoint input even when two synthetic branches reuse one Python object. Production arms independently load their checkpoint files; no already-run real experiment was found or claimed to be contaminated. The model, loss, schedule, sampling, and experiment budgets are unchanged.

Two regression tests explicitly demonstrate pinned CPU aliasing and require restored optimizer tensors to be equal but storage-independent. Both original four-vs-two-plus-two tests now additionally require the complete saved parent digest to remain unchanged. The original full-state equality, four-stream/noise audit, and journal checks remain strict.

| Actual execution | Result | Wall/CPU cost |
|---|---|---|
| Original immutable 75586c5 tests | 14 passed, 2 failed; preserved | Original status/log retained |
| Independent synthetic alias diagnosis, CPU0 / CUDA hidden | Four cases: shared fails, isolated exact, for both losses; exit 0 | Body 3.758276 s; CPU 3.496833 s; outer 4.190574 s |
| Fresh fixed-helper/test snapshot over immutable 75586c5 dependencies | 18 passed, 0 skipped; exit 0 | Pytest 3.22 s; body 4.067384 s; CPU 3.313488 s; outer 4.566742 s |

Inner and outer wall times are nested, not additive. The two new outer jobs together took 8.757316 s. Both were synthetic CPU tests, with zero real dataset/checkpoint reads and zero GPU calls. The diagnostic executes 40 tiny optimizer steps across four cases; those are test work, not a Min-SNR experiment. Local tests separately gave 11 passed and 7 Torch-dependent skips because local Python has no Torch; they are not substituted for the actual server result.

The fixed validation used two hash-recorded files uploaded to a fresh incoming directory. All unchanged dependencies and the policy came from immutable 75586c5; only test module policy-path resolution was set to that frozen root. It did not modify the release or prior wrapper. Root will commit and freeze the local repair before any actual parent inspection or training. Archived test source has a `.py.txt` suffix solely to prevent pytest discovery; its bytes and original path are recorded in `INDEX.json`.

`original_source/actual_*` preserves the executed release's original CRLF bytes, while `original_source/git_*` preserves the 75586c5 Git blob's LF bytes. Their normalized source content is identical; both original forms remain indexed. The fixed incoming snapshots preserve the exact local bytes tested, rather than claiming their hashes are the future Git/deployment hashes.
