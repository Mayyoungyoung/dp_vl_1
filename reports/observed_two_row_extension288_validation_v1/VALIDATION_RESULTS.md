# Extension288：服务器测试与独立 prepare 实证

固定 source `5c8f8e4f5cd478c793a0e0d9640005deaf700973`，code archive SHA `6ae339f5e0f8d88b79340523fdc5bc9ef478a1b961d56813d9b1deae37298030`。本归档仅保存已经完成的 tests/prepare、注册、manifest 和实际源码；没有读取任何采集 raw、未闭合 TRAIN、旧 LOCKED 或新 DEV 图像/轨迹/验收内容。

| 阶段 | 实测结果 | record PID / child | UTC 起止 |
|---|---|---|---|
| targeted_tests | 158 passed in 34.10s，exit0 | 623161 / 623162 | 2026-10-02 23:16:18.078175 → 23:16:52.424163 |
| registration_prepare | completed，exit0；只准备注册，不启动模拟器 | 623692 / 623693 | 2026-10-02 23:17:31.457671 → 23:17:39.235982 |

测试实际 command、cwd、source、日志、注册器、launcher 原字节、起止与退出码均在 `tests/` 和 `prepare/`。`SERVER_ARCHIVE_INDEX.json` 逐文件列出服务器来源、字节数与 SHA：共 48 个服务器文件、1,442,723 bytes，复制后全部验证一致。源码副本加 `.txt` 后缀，避免归档测试被 pytest 重新收集。`LOCAL_ARTIFACT_INDEX.json` 覆盖本报告与其余本地归档文件。

prepare 后注册 SHA 仍为 `574167e8a818dc6ee3d6ac197c6436e8031489306b2f8514466bd764f0cac70e`。288 父、7,776 槽、256 TRAIN + 32 新 DEV、五个独立采集阶段与 8GiB 内部预算保持。`training_authorized=false`、`raw_dev_analysis_authorized=false`；prepare 没有自动继续下一阶段。

实际源码换行有可审计差异：服务器 `collect_two_row_extension.py` SHA `efca0619756c4859629462236daa1f89191c01e82cdb79194dda6dcf74ef3532`，含 295 个 CRLF；当前本地工作树 SHA `48f8971f616c472dbefe604386b85fa06d112a1b8321615ccacd745538f60a7d`，含 0 个 CRLF。两者 CRLF→LF 后字节完全一致。这是换行差异，不是算法改动。实际执行来源以服务器 SHA 和冻结 release 为准，不能用本地 SHA 冒充。

root 后来单独启动 train32 fresh，session `20261002T231813Z_train32_624110`、CPU2/3；该作业运行中的状态不属于这份已完成证据归档，也没有提前填入采集成功率或完整分母。32 父全部闭合前不做 raw 分析；后续 train64/128/256/dev32 均需另行明确启动。
