# 主张—证据对应表

| 主张 | 状态 | 实际证据 | 边界 |
|---|---|---|---|
| 历史集合回归优于现有两类扩散 | 已复核 | historical commit d0d97eb；multiseed_summary.json；audit_v2/historical_regressor_seed0_metrics.json | 固定三路、真几何/真终点、冻结CLIP；旧TEST/OOD已公开使用 |
| 已构建可变路线类型和多障碍探针 | 已实现并验证 | multigate.py；audit_v2/multigate_v1.manifest.json；1152父场景7131正例 | 两墙平面通道结构、恒定z、固定语言；不是机器人观测任务 |
| 随机K参考子集会造成路径平均 | 受控单种子支持 | v2_round1逐场景结果、配对区间、ROUND1_REVIEW.md | 已知多模态回归问题，不能单独作为创新 |
| 类型不足K时应允许有效重复 | 受控单种子支持 | v2_round2/saturation_summary.json；精确饱和匹配与穷举测试 | 当前开发探针达到可覆盖类型上限；尚无独立任务结论 |
| 补全覆盖记忆优于普通attention | 待实测判定 | 已启动配对2500步，固定参数/初始化/上下文/曝光；JOBS及server logs | 不能将[A,A]的幂等性写成attention没有的能力 |
| 真正使用Qwen3-VL | 权重/环境准备，训练证据待补 | OBSERVATION_READINESS.md；固定官方revision与文件hash | 未完成实际前向之前不得用CLIP或代码存在代替 |
| 同初态RLBench多路线 | 正在核验；失败保留 | observation采集attempts/logs | RNG重置不足的12次拒收不能当有效数据 |
| 研究初版核心已成立 | 尚未成立 | REMAINING_EXPERIMENTS.md | Qwen观测训练、机器人代表性数据、核心机制优势和多种子仍是核心缺口 |

所有开发集bootstrap区间只描述已选择开发数据上的配对差异，不能作为最终锁定测试的确认性推断。未实测字段在MAIN_RESULTS中留空。
