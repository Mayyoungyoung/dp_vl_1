# 主张—证据对应表

| 主张 | 状态 | 实际证据 | 边界 |
|---|---|---|---|
| 历史集合回归优于现有两类扩散 | 已复核 | historical commit d0d97eb；multiseed_summary.json；audit_v2/historical_regressor_seed0_metrics.json | 固定三路、真几何/真终点、冻结CLIP；旧TEST/OOD已公开使用 |
| 已构建可变路线类型和多障碍探针 | 已实现并验证 | multigate.py；audit_v2/multigate_v1.manifest.json；1152父场景7131正例 | 两墙平面通道结构、恒定z、固定语言；不是机器人观测任务 |
| 随机K参考子集会造成路径平均 | 受控单种子支持 | v2_round1逐场景结果、配对区间、ROUND1_REVIEW.md | 已知多模态回归问题，不能单独作为创新 |
| 类型不足K时应允许有效重复 | 受控单种子支持 | v2_round2/saturation_summary.json；精确饱和匹配与穷举测试 | 当前开发探针达到可覆盖类型上限；尚无独立任务结论 |
| 补全覆盖记忆稳定优于普通attention | 未获支持 | COMPLETION_THREE_SEED.md；三种子新增有效类型差+0.1641/+0.0156/-0.0469 | seed2反转；父场景bootstrap不能代替训练种子稳定性；不提升为核心方法 |
| 自产草稿补全优于一次联合生成 | 原配方不支持；修正正在训练 | v2_completion/rollout_seed0；严格总K4且不丢弃失败；selfdraft后期混合配对 | 初始2+2更慢且更差，不能只展示给定正确草稿的结果 |
| 真正使用Qwen3-VL训练/推理 | 已实测 | 官方固定revision；96真实RGB+语言前向；observed_frozen_v1；observed_online_v1 | 不是CLIP替代；真正接入不等于路线质量已达标 |
| 真实LoRA参数更新 | 已实测 | 末两层q/v共114688参数，8张量非零梯度+改变hash；observed_online_v1/lora_seed0 | 短训练无质量收益，不将“参数更新”解释为有效微调；缓存未复用 |
| 观测语义目标精确定位 | 未达标 | 23 DEV指令/8父，冻结头终点19.54cm、严格3cm正确率0；observation_head_diagnostic | 最近目标身份诊断81.5%不能替代严格成功；正在补齐观测几何编码 |
| 同初态RLBench多路线采集 | 已完成小批实采 | derived32:275/288成功、32父、restore/RGB0差；4原任务12/12采集成功 | 采集成功不是模型执行；未据轨迹距离定义不同类型，连续全身碰撞未认证 |
| 研究初版核心已成立 | 尚未成立 | REMAINING_EXPERIMENTS.md | 主要缺口是有效观测规划和稳定核心优势；剩余工作不只是扩大规模 |

所有开发集bootstrap区间只描述已选择开发数据上的配对差异，不能作为最终锁定测试的确认性推断。未实测字段在MAIN_RESULTS中留空。
