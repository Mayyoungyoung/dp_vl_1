#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=1915de7dbe2c8cbc772250e092be87c83c7df786
SOURCE="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_sft_teacher_audit_v1"
test ! -e "$JOBS"
mkdir "$JOBS"
cp "${BASH_SOURCE[0]}" "$JOBS/launcher.sh"
sha256sum "$JOBS/launcher.sh" > "$JOBS/launcher.sha256"
printf '%s\n' "$$" > "$JOBS/launcher.pid"
cd "$SOURCE"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cat > "$JOBS/expected_source.sha256" <<'HASHES'
fcacb71f4f7225b58c7dd4b7a30f24db7b0c54560540b0fffd4f7d8451510e30  routeset/vlm_sft_data.py
2001fc1b1e10d578058d9ef88989cc4e403b5cea1fa65c082a53a937b617126e  routeset/vlm_route_serialization.py
ed18ff29ec36fb2d93c3965ca349ac34cbc8d28f2abbfbc14d7abf39b458c12c  routeset/vlm_sft_loss.py
b67c38be821ad335d4080a6b9e55bc1f423451396bcd682459676da5de5d808b  routeset/observed_route_head.py
ce02085a9462bcdd4e20717b4a7e2a583ca001b475699a4ac08141ef908d7b70  scripts/train_observed_lora.py
6fd798ed9197905a46744cfa704a7f951d76025a0d2dc0d4891d912bf0b7810e  routeset/vlm_teacher_audit.py
d78b458fce211210fe1f348789d46f846dfc9a3deb02905b01d21cc7d731c436  scripts/audit_vlm_sft_teacher_forcing.py
a0da1da1ecdfce7e15a027316bd3d2bd96467db9f47e6b5898106f71b57add52  tests/test_vlm_teacher_audit.py
b5c7efb90b1722a72e32b39a6f34cef92e83840f08a496e0e83ae141a1c2fc38  tests/test_vlm_teacher_audit_torch.py
HASHES
sha256sum -c "$JOBS/expected_source.sha256" | tee "$JOBS/preflight.log"
sha256sum routeset/vlm_sft_data.py routeset/vlm_route_serialization.py routeset/vlm_sft_loss.py routeset/observed_route_head.py scripts/train_observed_lora.py routeset/vlm_teacher_audit.py scripts/audit_vlm_sft_teacher_forcing.py tests/test_vlm_teacher_audit.py tests/test_vlm_teacher_audit_torch.py configs/vlm_depth_interface_train8_v1.json > "$JOBS/source.sha256"
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_vlm_teacher_audit.py tests/test_vlm_teacher_audit_torch.py -q
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id train8 --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.audit_vlm_sft_teacher_forcing --checkpoint "$P/runs/vlm_route_sft_v1/seed0/best.pt" --generation "$P/runs/vlm_route_grammar_v1/train8_k1" --output "$JOBS/train8"
