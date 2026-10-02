# 固定 last1500 TRAIN 诊断归档

实际作业已完成（2026-10-02 19:23 UTC），本地归档没有启动新服务器任务。结论见 [拟合诊断报告](../OBSERVED_TWO_ROW_LAST_TRAIN_FIT_RESULTS.md)。

- `analysis/report.json`：原始完整报告，含各 192 条 best / last 匹配残差、原始来源 SHA、初末机械门禁。
- `analysis/artifact_index.json`：原始五文件字节索引，保留未改。五项 SHA-256 / bytes 全部通过。
- `analysis/last_train/metrics.json`、`per_scene.json`：48 请求、192 候选的实际评价。
- `analysis/last_train/predictions.npz`、`paired_language_predictions.npz`：原始二进制保存池，位于本地 ignored 文件，禁止纳入普通 Git；不删除、不改写。
- `analysis.status.json`、`targeted_tests.status.json`、同名 `.log`、`registry.jsonl`：实际命令、PID、退出码与 3 passed in 0.12s 的完整证据。
- `frozen_wrapper.sh`：`.bootstrap/two_row_last_train_743d9b2.sh` 947 字节原样副本；已完成的 fresh-only launcher 不可再次启动。
- `SOURCE_INDEX.json`：immutable 源码 LF/CRLF 字节核验、wrapper 索引、原始五文件核验、既有训练来源绑定。运行源码 SHA 以原始报告为准，不以正在扩展的当前工作树代替。
- `DERIVED_METRICS.json`：从原逐顶点数组派生的匹配 ADE / p95 / max、已完成作业时长；没有新增 forward。
- `LOCAL_ARTIFACT_INDEX.json`：本目录除自身外的文件与正式 Markdown 报告的字节索引，含 ignored NPZ；索引本身不自引用。

原 checkpoint 未重复下载；源码报告的 last.pt SHA 与既有服务器同步字节索引一致。三份原训练来源的本地副本（config、summary、best TRAIN NPZ）已重新读取核验。原 prefix28 报告和全局 STATE / LOG 未改。
