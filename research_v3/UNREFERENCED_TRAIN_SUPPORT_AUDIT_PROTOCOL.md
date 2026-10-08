# TRAIN-only unreferenced positive vocabulary audit

Both within-known-mode augmentation and frozen-input controls failed their
registered gates. The unchanged reference vocabulary deliberately excluded853
cross-valid TRAIN paths, treating them as positives outside teacher modes,
never as negatives. Before any further training, inspect this already-sealed
TRAIN evidence to determine what reference-vocabulary completion would change.

Read only verified_edit_support_v1/support.npz, records.json and receipt.json
from the closed local archive. Check file hashes and explicit1152TRAIN IDs;
every source/destination record must refer to these IDs. Do not read DEV,
TEST_LOCKED, score/calibration records, images or new oracle payloads. Do not
forward models or mutate support files.

Report all853 records, exact unique destination/path-event pairs, unique
destination/mode pairs, affected requests/families, all mode counts and all
family counts. Preserve full records. Compare each request's known-mode count
before/after adding these verified classes; report max and quantiles plus
unavoidable max(count-8,0) capacity deficit. No filtering by frequency, model
loss, outcome or distance. These are correlated model-generated but separately
geometry-verified positives, not independent demonstrations or exhaustive modes.

This is a local statistical bookkeeping audit from existing results, not a
new server experiment; record actual script/inputs/command hashes. Remaining
server command budget385.776200s is unchanged. No automatic new training,
novelty claim, default replacement or adjustment to existing gates follows.
