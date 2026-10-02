# Prefix76 全部实际证据索引

正式结论见 [完整报告](../OBSERVED_TWO_ROW_PREFIX76_BASELINE_RESULTS.md)。所有缓存/训练/诊断/在线任务已完成exit0。此目录的归档与本地汇总没有新增模型调用、模拟器、优化器更新或服务器实验。

| 路径 | 内容 |
|---|---|
| training/ | 原config/summary/history、best/last/train逐场景、source/runtime、完整日志状态、初始化/曝光审计与固定36DEV特征收据 |
| preparation/ | 原CPU导出、TRAIN1108条容量/H24审计、固定DEV行和源SHA、缺初图与全部请求分母 |
| export_metadata/ | 已导出的本64TRAIN+旧12DEV manifest、观察/监督/尝试JSONL、父inventory；不含其他角色raw |
| cache_metadata/ | 实际225次Qwen的cache config/status/samples元数据；未复制特征数组 |
| diagnosis/ | 原保存池分析、原SHA索引、全部12父原图、CPU日志状态 |
| last_train/ | 固定last1500实际189次CPU head请求结果、全部756槽匹配距离、原索引/日志 |
| online/ | 固定best500真实36请求、每请求封存SHA、实际连续walltime、逐槽检查和原cache一致性 |
| validation/ | 已执行同源9e公共140 tests的原日志/status副本；不是重新执行 |
| wrappers/ | 两个root实际诊断/online本地wrapper原字节副本、已验证9e scaling launcher和公共test wrapper |
| visual_qa/ | 全12原图contact sheets/源SHA/查看记录、固定曝光三点比较图 |
| SYNC_SHA256_INDEX.json | 158项server path/SHA/bytes；156复制、45NPZ ignored路径、2PT只hash |
| SOURCE_INDEX.json | 22项运行源码与不可变Git字节精确LF/CRLF匹配；wrapper来源和SHA |
| OFFLINE_ANALYSIS.json | 本地保存池复核：16/32/64全部逐父/目标/颜色、四格失效、拟合残差、成本 |
| offline_saved_pool_analysis.py | 仅本地保存JSON/NPZ hash的复核入口，无模型/原始几何/网络读取 |
| LOCAL_ARTIFACT_INDEX.json | 本目录除自身外与正式报告的最终hash |

45个预测NPZ保存在 `F:/dpvlm/runs/synced_two_row_prefix76_v1/`，已验证Git ignored；不要force add。两个checkpoint只保留server SHA，未复制进Git。原服务器路径保持POSIX。

## 全部12父图

每图均含3目标、4候选、best/last、XY/XZ和全部接受参考；没有筛选成功场景。

- [283264 全部预测与参考](diagnosis/analysis/two_row_reach_283264_all_predictions.png)
- [283265 全部预测与参考](diagnosis/analysis/two_row_reach_283265_all_predictions.png)
- [283266 全部预测与参考](diagnosis/analysis/two_row_reach_283266_all_predictions.png)
- [283267 全部预测与参考](diagnosis/analysis/two_row_reach_283267_all_predictions.png)
- [283268 全部预测与参考](diagnosis/analysis/two_row_reach_283268_all_predictions.png)
- [283269 全部预测与参考](diagnosis/analysis/two_row_reach_283269_all_predictions.png)
- [283270 全部预测与参考](diagnosis/analysis/two_row_reach_283270_all_predictions.png)
- [283271 全部预测与参考](diagnosis/analysis/two_row_reach_283271_all_predictions.png)
- [283272 全部预测与参考](diagnosis/analysis/two_row_reach_283272_all_predictions.png)
- [283273 全部预测与参考](diagnosis/analysis/two_row_reach_283273_all_predictions.png)
- [283274 全部预测与参考](diagnosis/analysis/two_row_reach_283274_all_predictions.png)
- [283275 全部预测与参考](diagnosis/analysis/two_row_reach_283275_all_predictions.png)

[合页1](visual_qa/all12_page1.png) · [合页2](visual_qa/all12_page2.png) · [合页3](visual_qa/all12_page3.png) · [16/32/64比较](visual_qa/fixed_exposure_scaling.png)。全部已目视查看，原图hash不变。

复核本地归档：`python reports/observed_two_row_prefix76_v1/offline_saved_pool_analysis.py`。实际服务器命令保留在各status；完成的fresh-only whole launcher不可原地重放。若需另做复现，必须选择明确新输出并由root排队。
