# Completed-uniform postvalidation recovery — ready, not executed

The original scientific uniform child completed all 500 steps and saved valid sealed outputs. Its wrapper then failed because this agent's textual `16 -> 18` substitution accidentally changed `additional_input_draws != 16000` into `!= 18000`. This is an orchestration error, not a scientific training failure. The original outer failed status, receipt, identity, log, and all inner outputs are retained. **Do not rerun any stage through the original f72 wrapper.**

Actual uniform metadata fixture: `.bootstrap/minsnr_uniform_postvalidation_failure/`. It records final step12500, 500 steps, 16000 input draws, 64000 target states, 791 geometry/11930 denoise calls, no new Qwen/DEV calls, inner process77.111573692s, child exit0 and outer exit1. All eight captured metadata files are pinned in the recovery manifest. No original PT was loaded or downloaded for this preparation.

New independent run: `/home/wzy/dpvlm/route_set_v1/runs/observed_diffusion_minsnr_probe_v2_recovery_v1`. Scientific source remains `f72e1aa11a37df2001877035fbfde9a4525c0aa0`; no loss, threshold, data, model, optimizer, 500-step or 360-second limit changes.

The new recipe has exactly three independently requested stages:

1. `seal-uniform`: reuse the original completed18-test and parent-inspection evidence by exact hashes, verify the exact completed-child/failed-postvalidation boundary, then read-only verify the original uniform summary, last PT hash, restore receipt, actual source hashes, all three issued-call chains, teacher and free-pool sealed artifacts. Save a **new** seal with all artifact hashes. No parent PT load, model, optimizer, checker, raw data, or uniform training.
2. `min_snr_5`: require the successful new seal and unchanged original artifacts. Start only the weighted arm into the new recovery directory, using the same original parent and original CPU inspection. No tests/inspection/uniform replay. Actual budget remains500/16000/64000 and cap360s. The corrected completed-arm validator checks actual counters and sealed outputs.
3. `compare`: require the new weighted receipt and uniform seal, then call the original fixed comparison with the **original** uniform path and **new** weighted path. It does not generate predictions or reselect a checkpoint.

Each stage is fresh-only. Any failure retains its issued work and stops; no automatic chaining/retry is present. Original uniform cost is recorded once, not charged again as new training. New sealing/comparison costs remain separate; nested wrapper/runner/child times must not be added together.

## Local evidence

`.bootstrap/minsnr_recovery_numeric_semantics.json` enumerates **all56 numeric AST constants** of original755, broken f72, and corrected reference. Exactly six 16-to18 changes belong to test identities/counts. The sole unintended change was16000-to18000. The new corrected reference differs from the original f72 runner in that single numeric literal only; it is loaded only as a validator, never executed as a launcher.

`.bootstrap/test_minsnr_recovery_recipe.py` uses actual completed uniform summary/status JSON (a clearly fake PT placeholder and explicit hash stub only for pure validator testing). It demonstrates the original guard rejects the real16000 result, the corrected guard accepts it, and altered budgets including18000 still reject. Tests also verify failed-boundary immutability, no uniform/tests/inspection command, exact new paths, full numeric proof, source hashes, and LF/bash syntax. **16 passed,0 skipped,0.47s** in `minsnr_recovery_local_tests_v2.xml`; first local test attempt had only a Windows slash assertion failure, preserved in `minsnr_recovery_local_tests.xml`, corrected with `Path` comparison. No server/model test was performed for this preparation.

## Immutable files and root execution

- `minsnr_recovery_f72e1aa.py` SHA `221c98c22aa886a2fce871753ded817ec44ba916e811cd9a9d133ac5401c89de`.
- `minsnr_recovery_f72e1aa_manifest.json` SHA `7ac551936241dd3d48bd4c4560439501cf4dc9b70f4691e124145a0c1c0af33f`.
- `minsnr_recovery_f72e1aa.sh` SHA `a72e8c88b1adc4bc02dab3342acafab87cd254eb765b62224a7024cee5fe854a`.
- `minsnr_runner_f72e1aa_corrected_guard_reference.py` SHA `a9951a36f2dc2f7e62fb89a0f0eeec8061404509ae88a0a97adb184287503bd5`.

Deploy uploads only those four **new incoming** files and checks original incoming hashes; it refuses any existing new destination or recovery run. It never modifies the release or original output, and performs no stage automatically:

```powershell
F:/ProgramData/anaconda3/python.exe .bootstrap/deploy_minsnr_f72e1aa_recovery.py
```

Then root issues each command separately, after reading the actual prior stage:

```powershell
ssh wzy3090 "bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/minsnr_recovery_f72e1aa.sh seal-uniform"
ssh wzy3090 "bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/minsnr_recovery_f72e1aa.sh min_snr_5"
ssh wzy3090 "bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/minsnr_recovery_f72e1aa.sh compare"
```

All run on CPU0/thread1. Seal/comparison hide CUDA; weighted uses original GPU1 UUID with original35% limit. Root controls timing and execution. No command above was executed during preparation.
