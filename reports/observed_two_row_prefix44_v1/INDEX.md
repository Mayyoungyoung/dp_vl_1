# Prefix44 实际证据索引

结论见 [正式报告](../OBSERVED_TWO_ROW_PREFIX44_BASELINE_RESULTS.md)。所有服务器训练/生成/诊断已完成；此目录的归档与本地汇总没有启动新模型、模拟器或服务器计算任务。

| 路径 | 内容 |
|---|---|
| training/ | 原训练config/summary/history、best/last逐场景指标、实际source/runtime、完整日志状态、初始化与192k曝光、旧36DEV特征一致性收据 |
| preparation/ | 已执行CPU导出和全部TRAIN容量审计、固定DEV原始行与源SHA一致性、失败/缺输入分母 |
| export_metadata/ | 本次已导出TRAIN/DEV的manifest、输入/监督/尝试JSONL、父inventory；不含locked原始文件 |
| cache_metadata/ | 真实Qwen cache config/status/samples元数据，无缓存特征数组 |
| diagnosis/ | 原保存池分析JSON、原SHA索引、全部12父原图、CPU日志状态 |
| last_train/ | 固定last1500新增93次CPU head预测的指标、逐候选匹配残差、原索引与日志 |
| online/ | 固定best1000真实36请求的时间、封存SHA、逐候选判定、缓存比对、原始状态 |
| validation/ | 公共源码验证的原样日志与status副本，140 passed in17.16s |
| wrappers/ | 两个已有实际诊断/online wrapper、公共test wrapper原字节副本及immutable scaling launcher |
| visual_qa/ | 覆盖全部12原图的三页contact sheets、源图SHA与实际查看记录 |
| SYNC_SHA256_INDEX.json | 158项服务器文件SHA/bytes、156项已复制本地路径、45项NPZ ignored位置、两个PT只索引未复制 |
| SOURCE_INDEX.json | 实际源码SHA与immutable Git字节的LF/CRLF精确核验；wrapper字节索引 |
| OFFLINE_ANALYSIS.json | 保存池目标/颜色/父场景配对、四格错误、原匹配残差、成本等本地派生量 |
| offline_saved_pool_analysis.py | 可复算本地JSON/保存池哈希的入口；无模型/raw几何/服务器访问 |
| LOCAL_ARTIFACT_INDEX.json | 本目录除自身外全部文件及正式报告的最终SHA/bytes |

45个NPZ位于 `F:/dpvlm/runs/synced_two_row_prefix44_v1/`，保持ignored，不force add；原始预测封印和SHA不改。两个PT仍在服务器原运行目录，完整SHA在索引中，未复制到Git。

已结束的fresh-only wrapper不可原地重播；需复现时固定同一release并明确独立输出、由root分配资源。单请求时延只认online的连续walltime；缓存head吞吐不替代它。
