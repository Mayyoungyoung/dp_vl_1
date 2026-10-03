# eba0994 local deployment handoff — not executed on server

Source: `eba09945ecade4bdb9a5b4ab123a92da898283ed`.

Upload these six files to the existing `/home/wzy/dpvlm/route_set_v1/research_v2/incoming/`:

| Local `.bootstrap/` file | SHA256 |
|---|---|
| `layout_variation_eba0994.tar.gz` | `98251263bba95d8cb52a8fff15dcc23e1c905106ab36aede0f0e4093f5104f1a` |
| `layout_variation_sources_eba0994.json` | `6b779adf227e2822ecf224debb65161c035a2d010ef4c98051d70fc94ee86baf` |
| `layout_variation_runner_eba0994.py` | `6fbaeca615247e6bd76ca5d7fff10bace0265daf2c0d580d98305bce8f72e586` |
| `layout_variation_tests_eba0994.sh` | `4a0e9dccf6b4d96175db329ea09a9137db191697c0e5e33cd5ecfd32612a4ace` |
| `layout_variation_prepare_eba0994.sh` | `2a9fa9e17e371aa60302ae8d3cbadb5ee9b0ce2549565b1fa28765bf7684467b` |
| `layout_variation_pilot4_eba0994.sh` | `4d73bde38d7052ffe71430282c2daf31441e8bdef3c25a6bfbad9564ed323f6d` |

Archive: 1,105,902 bytes, 419 regular files, only `routeset/ scripts/ configs/ tests/ requirements.txt AGENTS.md` and this experiment's protocol report. No data/weights/old outcome reports. All six test suites use config metadata or synthetic fixtures; no extra historical report fixture is required. Manifest hashes actual tar-member bytes (including Git's line endings), not normalized Git blobs or worktree estimates. Extract only into the new immutable release directory after verifying tar SHA/member safety; do not overlay a running release.

Each command below is a separate root action after inspecting the previous completed receipt; this document does not run or chain them:

```bash
bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/layout_variation_tests_eba0994.sh
# Read tests.status.json, tests_receipt.json and exact114 JUnit first.
bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/layout_variation_prepare_eba0994.sh
# Read prepare.status.json, prepare_receipt.json and new registration/presence first.
bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/layout_variation_pilot4_eba0994.sh
```

All wrappers pin CPU1, hidden CUDA and numeric/LP threads1. Tests and coordinator use existing `.venv` Python3.8; workers use existing `.venv-sim`. Runner's pilot4 context creates an isolated Xvfb from `.observation-deps/xorg/usr/bin/Xvfb`, uses existing CoppeliaSim/software rendering settings, and cleans up only its own child. No global Xvfb/service/other job cleanup. TRAIN256 CPU2/3 is untouched. Root must keep total resource authorization satisfied before invoking any stage.

New corpus: `data/observed_layout_variation_train12_v1`. Family: `runs/observed_layout_variation_train12_v1`. Tests use fresh `/tmp/dpvlm_layout_variation_eba0994_pytest`, outside both ID-scan roots, preventing synthetic parent names from becoming false production ID collisions. Existing fixture root or stage status/lock is rejected. Common recipe lock prevents concurrent stages; active actual variable-layout coordinator/worker PIDs are checked without logging other command lines. Each stage has a fresh record_job with `--resume-strategy none`; this initial pilot4 recipe has no resume or pilot12 entry.

Budget preflight: new corpus+run logical bytes plus 512MiB must fit 8GiB; filesystem free and host MemAvailable must each be at least512MiB. This is not a virtual-memory limit or a total-host ownership claim. Immutable eba coordinator retains its already-reviewed single-parent256MiB reserve and45min parent-boundary worker-time cap; wrapper does not silently patch them. Internal parent outputs/logs and hashes remain as registered.

Local preparation only: source collection verifies114 exact testcase identities;11 synthetic recipe checks passed (exact/missing/duplicate/substituted/skip/suite count, incomplete prior stage, Python AST and three Bash `-n`/LF checks). `layout_variation_recipe_local_receipt_eba0994.json` records their hashes. These do not replace actual `.venv`114/0skip tests. No SSH, simulation, model calls, corpus registration or server ID availability check was performed during packaging.
