# ac6882c 首16 TRAIN 跨目标审计证据

全部已完成，exit0；只归档原有16父/48指令/285正参考及两池各192候选。联合预登记筛查失败，不启动跨目标辅助。

- `analysis/report.json`：完整原始6.84MB JSON，不压缩案例，不删除unknown、歧义、长弧或失败门槛。
- `analysis/artifact_index.json`：原始report SHA索引；已与复制字节重新核对。
- `analysis.status.json`、`analysis.log`、`registry.jsonl`：原始运行记录；record548227、child548228、实际20:27:14.993532–19.649714UTC。
- `SYNC_SHA256_INDEX.json`：5个原始服务器文件路径、实际字节数和SHA；全部复制后匹配。
- `source/` 与 `SOURCE_INDEX.json`：8个实际源码/配置依赖、对应测试、部署及本地wrapper、同commit预登记协议。协议属于Git元数据，未部署进code-only服务器release；一次只读复制缺文件已记录，没有隐藏失败试验。
- `validation/targeted_tests.log`、`targeted_tests.status.json`：同commit真实服务器73项相关测试通过，4.81s。共用验证记录的复制，不是额外重跑。
- `DERIVED_SUMMARY.json`、`summarize_frozen_report.py`：仅对封存report做本地描述聚合，逐槽互斥排除、原始screen、成本与未通过决定。无模型或原数据读取。
- `LOCAL_ARTIFACT_INDEX.json`：本目录所有文件（除自身）及上级正式中文报告的最终字节索引。

正式解读：`../OBSERVED_TWO_ROW_CROSS_GOAL_OPPORTUNITY_RESULTS.md`。全局STATE、MAIN、registry由root更新，本目录不改写它们。

复算本地描述：在仓库根目录运行 `python reports/observed_two_row_cross_goal_opportunity_v1/summarize_frozen_report.py`。原服务器作业为fresh-only且完成，不能重新执行其已用输出路径。
