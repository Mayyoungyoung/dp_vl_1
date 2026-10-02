# Resource envelope — audited 2026-10-02 Asia/Shanghai

- Host: configured SSH alias `wzy3090`; account `wzy` (uid 1025).
- Existing project: `/home/wzy/dpvlm/route_set_v1`, verified, about 6.0 GiB.
- GPU 1: RTX 3090 24576 MiB, UUID `GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab`.
- No compute jobs visible at initial audit. GPU 1 showed 69 MiB; GPU 0 is outside our resource envelope.
- No evidence of higher resource authorization. Preserve 35% CUDA allocation cap (about 8.4 GiB), four CPU threads, sequential GPU queue.
- Host has 64 logical CPUs. This is not permission to use them all.
- Shared filesystem: 11 TiB total, about 1.3 TiB available. `quota -s` returned no usable quota information. Initial project growth budget: 30 GiB (Qwen cache, isolated environment and pilot data); check space before large acquisition.
- Existing environment: `.venv`, historical pinned PyTorch 2.4.1/CUDA 12.1. Preserve it. New Qwen dependencies use separate environment.
- No existing tmux session reported. New jobs must record PID, logs, status and exit code, retain optimizer and RNG checkpoints, and prevent duplicate launch.
- Small controlled training first: profile 100 updates, then paired 2000–4000 update runs. Record actual duration; do not claim a forecast as measured GPU hours.
- Network: SSH works. Local Git has no remote, server directory no Git metadata. Push is blocked on supplied existing authorized URL; local commit/export remains usable.
