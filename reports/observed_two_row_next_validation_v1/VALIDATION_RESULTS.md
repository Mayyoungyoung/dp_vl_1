# 冻结源码 ac6882c 的真实服务器验证

2026-10-02 20:26:38.987–20:26:44.413 UTC，`wzy3090` 实际执行 **73 passed in 4.81s，无 skip**，record_job 退出码 0。该结果包含真实 Torch 的恒定学习率前缀复现和恢复测试；不只是本地没有 Torch 时的纯逻辑测试。

- 不可变 commit：`ac6882c023e21dac92c5a79564f1721c0459f267`。
- 原始记录：`targeted_tests.status.json`、`targeted_tests.log`、`registry.jsonl`；PID 547683，子 PID 547685。
- 原始 `.bootstrap/two_row_next_tests_v1.sh` 与服务器 incoming wrapper 已逐字节核对一致，原样归档在 `source/`。该 wrapper 的 `bash -n` 成功后才进入所记录 pytest；没有单独伪造 bash 检查日志。
- 不可变 release 的 scripts、routeset、configs、tests 共 314 个文件仅生成字节/hash 索引，见 `source_index.json`；另保留本次 convergence driver、launcher、配置及测试原文件。测试副本仅将本地后缀改为 `.py.source.txt`，原字节未变，避免被默认 pytest 重复收集。
- 每个复制文件均核对服务器 SHA256、字节数与本地 SHA256；完整映射见 `artifact_index.json`，其 SHA256 为 `6eb4afbbe74119ebe23452e8f108c223e86f6b8cb7df9d5735d524fb966be20f`。

这份报告只证明指定源码测试通过，不预先声称真实 1500 步复现门禁或 6000 步收敛完成。归档过程只读服务器元数据/源码，没有启动训练、采集或推理，没有读取 locked 原始数据。
