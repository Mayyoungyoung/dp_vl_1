# Verified positives outside the teacher vocabulary

TRAIN-only sealed records contain853 unique destination/path-event pairs,
adding760 destination/mode classes across608 of1152requests and125 of128families.
All classes are geometry-checked over-passage variants: gap0|over249,
gap1|over321, over|gap0 11, over|gap1 157, over|gap2 22 unique request/classes.
These positives were deliberately excluded from the within-mode augmentation
control, never labeled negative. This audit reads no DEV, images or new oracle
payload and performs no model forward/update.

Expanded per-input mode counts range6-13 (previous6-9). Requests above M8 rise
768->772, while summed unavoidable missing known classes rise768->1422
(.666667->1.234375 per request). All1152 per-request counts and all853 records
are retained, including every family. Additional classes are not free coverage:
they introduce capacity competition, and witness vocabularies remain incomplete.

This motivates the separately registered ALL_MODE_EDIT_COMPLETION_PROTOCOL.md
ordinary data-completion control, not a new algorithm or new labels at inference.
The deterministic comparison must retain original-parent context and report
matching-group RNG divergence caused by the enlarged class sets. Local command,
script and input/output hashes are in PROVENANCE.json. Server budget unchanged.
