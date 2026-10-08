# Closed all-mode control: final TRAIN fitting attribution

All-mode completion gate failed: fixed-source retention-2.6279pp versus known-
mode augmentation. Teacher-known stratum loses67retained modes, unreferenced
stratum gains10; total-57. New validity81.8576% versus parent86.8056%. The
observed tradeoff is consistent with capacity competition, not proof of its
exclusive cause. Need TRAIN fitting evidence before a final diagnosis.

Run one fixed-final all_mode model forward over all1152TRAIN requests using the
existing seal-then-oracle full TRAIN audit. No updates or q fit. Then reuse sealed
parent and known-mode augmentation TRAIN predictions/labels, check their hashes,
and compute operational valid-mode sets for all three models against BOTH
original teacher vocabulary and expanded verified vocabulary. Report known/new
hits, per-request recall and capacity-bound deficits, every request/family and
semantic/collision/floor counts. Do not select cases or change prior gates.

Single job cap50s within71.104314remaining experiment command seconds. Immutable
export, GPU1/35%/4threads, serial7200s wrapper. Explicit TRAIN records only for
computation; shared support.npz also materializes permitted DEV arrays, unused.
No TEST_LOCKED, scorer roles or raw collector access. All receipts and failures
retained. No automatic follow-up training or budget expansion.
