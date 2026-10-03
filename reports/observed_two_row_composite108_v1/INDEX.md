# Composite108 completed training evidence

Source: `71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c`. Both train and fixed-last-train completed with exit code0. This archive adds no model inference, raw-corpus reads or planning searches.

- [Results report](../OBSERVED_TWO_ROW_COMPOSITE108_BASELINE_RESULTS.md): sole comparison with original constant64/12000, fixed DEV36 and separate common/new TRAIN inputs.
- [REMOTE_ARTIFACT_INDEX.json](REMOTE_ARTIFACT_INDEX.json): 46 server-original files with paths, byte sizes and SHA256; two remote PT indices, no PT copied.
- [SYNC_RECEIPT.json](SYNC_RECEIPT.json): archive SHA and complete original-byte verification.
- [LOCAL_ARTIFACT_INDEX.json](LOCAL_ARTIFACT_INDEX.json): all locally retained files except the index itself; NPZ stored locally and Git-ignored. Also seals the external result report.
- [Training receipt](peak_seed0/composite_training_receipt.json), [summary](peak_seed0/summary.json), [48-point history](peak_seed0/history.json), [train status](train.status.json), [train log](train.log).
- [Fixed-last receipt](fixed_last_train/diagnostic_receipt.json), [request receipt](fixed_last_train/request_receipt.json), [CPU job status](fixed-last-train.status.json).
- [Saved-pool comparison](analysis/SAVED_POOL_COMPARISON.json): all 36 DEV condition pairs, all12 parent aggregates, old189/common189/new96/all285 TRAIN and exact cost/exposure fields.
- [Derived artifact index](analysis/artifact_index.json): comparison JSON, all12 original parent figures, history plot and all4 contact sheets.
- [QA.json](QA.json): full visual QA scope and limitations; no reclassification from images.
- `source/`: actual immutable scripts/model helpers, config and frozen wrapper bytes. These are evidence copies, not executable source edits.
- `archive_composite108_final.py`, `unpack_composite108_saved.py`, `analyze_composite108_saved.py`, `make_contact_sheets.py`: actual readonly archive/analysis helpers. `composite108_progress.py` reads compact status/log metadata only.

Actual remote output is `/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1`. Checkpoint files remain `peak_seed0/best.pt` and `peak_seed0/last.pt`; their hashes and sizes are in the remote index and training receipt. Do not relaunch the completed fresh-only wrapper.

Preparation and actual new96 Qwen encoding are separately archived in [the preparation family](../observed_two_row_composite108_preparation_v1/QUALITY_CACHE_RESULTS.md). Historical225 encoding is reused byte-for-byte, not charged as new encoding. No new extension DEV was opened. Old3 missing inputs remain missing.
