# 传统观测A*实际结果索引

完整结论见 [32/64统一报告](../OBSERVED_TWO_ROW_ASTAR_SCALING_RESULTS.md)。本目录实际评估已完成exit0，仅同步已保存结果。

| 路径 | 内容 |
|---|---|
| dev_model/report.json | 原全部36请求/144候选指标、成本、源码/输入hash |
| dev_model/fitted_train_model.json | 仅TRAIN拟合的原型/workspace、原配置与来源 |
| dev_model/requests/ | 36个原result/status/generation seal；NPZ位于同名ignored runs目录 |
| dev_model/artifact_index.json | 原运行全部产物SHA |
| dev_model.status.json / .log / registry.jsonl | 真实命令、PID、时间、exit与日志 |
| input_metadata/ | 原已导出TRAIN/DEV的3个metadata文件；原hash绑定报告 |
| SYNC_SHA256_INDEX.json | 188项server/local路径、SHA/bytes；73个NPZ ignored |
| SOURCE_INDEX.json | 9项实际源码与不可变ac688字节精确核验 |
| SAVED_POOL_ANALYSIS.json | 全槽生成失败/有限错误/有效/unknown/重复统计；0新生成 |
| LOCAL_ARTIFACT_INDEX.json | 本目录与正式报告最终SHA |

逐条件三点对照和纯本地分析入口位于64family的 `SCALING_COMPARISON.json` / `analyze_saved_scaling.py`。原failed_candidate_slots只表示未产生完整路径；有限错误另列，原始报告字节不改。SelectedTipValid/全机器人执行保持null。
