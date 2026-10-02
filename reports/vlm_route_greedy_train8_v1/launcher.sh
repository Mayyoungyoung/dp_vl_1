#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=2c47c2a4c354ecfeb2f73b8121476f5000e82a98
SOURCE="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_route_greedy_train8_v1"
test ! -e "$JOBS"
mkdir "$JOBS"
cp "${BASH_SOURCE[0]}" "$JOBS/launcher.sh"
sha256sum "$JOBS/launcher.sh" > "$JOBS/launcher.sha256"
printf '%s\n' "$$" > "$JOBS/launcher.pid"
cd "$SOURCE"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cat > "$JOBS/expected_source.sha256" <<'HASHES'
6d5fc713962e6e37156c9113c5bd77e1c3c85f09de0bcad5b789231c4bcc33e9  routeset/vlm_sft_train_probe.py
9d63a362732c511a5e87c1024444cf7ccc21af6b7159cb91dcdb511d1612f5c9  scripts/evaluate_vlm_route_sft_constrained.py
d00b7d204f4273068d28e681965ee0589c0149c9077c0dac925000121ebad562  scripts/preflight_vlm_greedy_config.py
ee667b4d2083bc6306310ca32f0ca69324bc810a8869b0be53b442e0628f6635  scripts/analyze_vlm_greedy_train8_pair.py
7b21f944efe898af2f6bad214f834ce21668b3b625fa055e9e276f51318878ae  tests/test_vlm_sft_greedy.py
fcacb71f4f7225b58c7dd4b7a30f24db7b0c54560540b0fffd4f7d8451510e30  routeset/vlm_sft_data.py
2001fc1b1e10d578058d9ef88989cc4e403b5cea1fa65c082a53a937b617126e  routeset/vlm_route_serialization.py
ce02085a9462bcdd4e20717b4a7e2a583ca001b475699a4ac08141ef908d7b70  scripts/train_observed_lora.py
1d539b227a36945c088d2d28eb31d45d7edff192d3edeea56c43551426f8a9fa  routeset/vlm_route_grammar.py
HASHES
sha256sum -c "$JOBS/expected_source.sha256" | tee "$JOBS/preflight.log"
sha256sum routeset/vlm_sft_train_probe.py scripts/evaluate_vlm_route_sft_constrained.py scripts/preflight_vlm_greedy_config.py scripts/analyze_vlm_greedy_train8_pair.py tests/test_vlm_sft_greedy.py tests/test_vlm_sft_train_probe.py tests/test_vlm_route_grammar.py tests/test_vlm_route_grammar_torch.py routeset/vlm_route_grammar.py routeset/vlm_sft_data.py routeset/vlm_route_serialization.py scripts/train_observed_lora.py configs/vlm_depth_interface_train8_v1.json > "$JOBS/source.sha256"
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_vlm_sft_greedy.py tests/test_vlm_sft_train_probe.py tests/test_vlm_route_grammar.py tests/test_vlm_route_grammar_torch.py -q
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id config_cpu --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.preflight_vlm_greedy_config --training "$P/runs/vlm_route_sft_v1/seed0" --output "$JOBS/config_preflight.json"
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id train8 --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.evaluate_vlm_route_sft_constrained --checkpoint "$P/runs/vlm_route_sft_v1/seed0/best.pt" --scope train8_preflight --train8-greedy --seed 0 --repeats 1 --output "$JOBS/train8"
export CUDA_VISIBLE_DEVICES=-1
unset RESEARCH_GPU_UUID
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id paired_analysis --resume-strategy none -- "$P/.venv/bin/python" -m scripts.analyze_vlm_greedy_train8_pair --sampling "$P/runs/vlm_route_grammar_v1/train8_k1" --greedy "$JOBS/train8" --supervision-metadata "$P/data/observation_obstacle_reserved_development_v1/supervision.jsonl" --output "$JOBS/analysis"
