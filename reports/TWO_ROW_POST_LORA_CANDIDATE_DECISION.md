# 等待 frozen / LoRA 后的机制判断

2026-10-03。只读研究笔记；没有修改模型、运行新 forward / 搜索、读取 reserved 或 extension 新 DEV。本笔记写于 `f41be1ff1a35b0eab3d36edf4ca23197359f4731` 的 frozen 续训期间，不能当成两臂结果。**当前尚未建立值得直接实现的核心新机制。下一最有区分力的工作是补齐同观测条件的普通独立/集合扩散，而不是恢复覆盖奖励或继续修端点。** 实用方案另存 [观测扩散设计附录](TWO_ROW_OBSERVED_DIFFUSION_DESIGN.md)。

## 现有证据支持什么

| 已完成证据 | 对下一决定的约束 |
|---|---|
| composite108 ordinary：best 63/144 TipValid、34个已分类类型、1个有效已分类重复；last 56/144、26个类型、0个重复 | 神经模型首先缺有效几何。不能把 unknown 有效路线算重复，也不能用稀缺的有效重复支持覆盖分配机制。best/last 的语义正确但碰撞分别65/63条。 |
| 同一 last 的 TRAIN 989/1140有效，公共旧189输入85.05%，新增96输入90.10%；DEV38.89% | 有拟合与泛化差距。扩大人口带来 best 小幅收益、last 反降，不是已证明的新方法；颜色/父场景人口与逐输入曝光共同变化。 |
| native100k：edge/spatial同为96/144有效；已分类类型总和24→62，重复65→34 | 标准空间惩罚已经能同时保持质量、提高覆盖。它的训练定位器只用旧31实际父，不能与95父神经模型写成完全同训练信息。全部失败主要是44个附件失败及4个错目标，不是继续加排斥就能修复。 |
| TRAIN12直接通路交换改变路线约5cm、clear45→33；但去掉direct分支的普通控制未建立DEV收益 | 依赖性不是泛化因果。剩余几何网络仍有语义条件；不能把普通模块拆分包装为成功机制。 |
| TRAIN69正参考在固定观测grid下完整接纳为0；全部终点voxel拒绝，同时多数路径还有内部拒绝 | 不能把0解释成无路；端点附件、膨胀和不可见空间混合。直接给原图学习代价并不能开放其禁止的边。 |

数字来源：[composite108](OBSERVED_TWO_ROW_COMPOSITE108_BASELINE_RESULTS.md)、[native100k](OBSERVED_TWO_ROW_NATIVE_ASTAR_100K_RESULTS.md)、[通路及图支持审计](OBSERVED_TWO_ROW_ROUTE_CONDITIONING_RESULTS.md)、[no-direct](observed_two_row_prefix76_no_direct_v1/NO_DIRECT_RESULTS.md)。native的搜索2秒/100k与神经模型的单次解码不是相同计算预算，约1.22秒空间臂延迟也不能搬给神经模型。

历史停止决定继续有效：多查询强基线之上的补全/max-memory、局部复制门控、local refiner、同场景重复参考假设、共同失效覆盖、跨目标对应screen、立即DTW修复均未建立实施依据。此前 TRAIN768 的响应参考最优/几何最远对/DPP在可评价父上无区别，也不因改名而消失。见 [共同失效审计](MULTIGATE_CLOSURE_OPPORTUNITY_RESULTS.md)、[覆盖风险卡](REFERENCE_COVERAGE_RISK_CARD.md)、[跨目标失败screen](OBSERVED_TWO_ROW_CROSS_GOAL_OPPORTUNITY_RESULTS.md)、[曲线对应复核](TWO_ROW_CURVE_CORRESPONDENCE_REVIEW.md)。

## 只保留一个待区分的问题

**当前观测模型是否需要学习条件可行路径分布，而不只是有限个确定性坐标解？** 这只是待检验问题，普通扩散或Flow Matching不是核心贡献。

有两种仍未分开的解释：一是冻结特征或全局几何摘要的泛化不够，二是确定性路线解码/训练对多正例条件的表达不够。代码中 `routeset/observed_geometry.py:138` 将点特征压成 attended+mean，`:192` 将直接Qwen条件、状态与几何context相加，`:204` 在预测anchor参考线上输出完整坐标。另一方面，`routeset/train_v2.py:23` 的 saturation 已经是全正例匹配，不是先把正参考平均成一条真值。32父6000步曾拟合到97.04%有效；单正参考条件也曾碰撞。因此“多正例坐标平均导致当前失败”**尚无因果证据**。

先等待固定frozen/LoRA的全部12次选择与fixed-last285池。只读分解旧189/新增96 TRAIN和旧36 DEV的语义×净空四格、未知/已分类有效类型及重复；按父比较，不能挑某条候选或某个step替代整池。如果LoRA主要修复语义或几何，先保留它作为强普通基线，不给新方法独享这个骨干收益。若仍以语义正确碰撞为主，附录的观测扩散是更直接的下一控制。

扩散筛选可能得到三种判断：

1. 独立与集合均提高有效性，但彼此无可靠覆盖差：支持分布模型这一普通基线，仍无集合核心机制。
2. 集合在相同独立噪声/正例曝光下提高有效类型而不靠增加失败：有后续集合机制研究空间；现有跨候选attention本身仍不新，后续须胜过相同数据的成熟多样化方法。
3. 两者合理收敛后仍主要碰撞：不追加PG、更多去噪或新种子保住方向。保留负结果，将下一问题定位于观测约束表示/数据覆盖，不能凭这一次比较证明感知是唯一原因。

不能用较低MSE、单个最优采样repeat或仅更大AnyValid判定成功。要求整池Valid—Unique/ReferenceCoverage—成本一起看；未知有效仍计Valid，但没有类型证据就不增加已分类Unique。

## 一手近邻把哪些说法排除

| 核验来源 | 本轮相关边界 |
|---|---|
| [ModeSeq，CVPR2025原文](https://arxiv.org/html/2411.11911v2)，§3.3–3.6 | 顺序模式记忆、EMTA、可变模式数已有直接先例。读取前面候选不构成新贡献；本轮仍未找到可确认的官方实现入口，没有声称复现。 |
| [DSF/DPP，ICLR2020原文](https://arxiv.org/html/1907.04967)，§4.2；[SetPO，ICML2026](https://proceedings.mlr.press/v306/li26jk.html) | 质量加权相关采样、集合边际效用都不是新概念。改成路线、换一个核或加正例覆盖项尚不足以成为核心；当前神经池的重复机会又很少。 |
| [GoalFlow，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_GoalFlow_Goal-Driven_Flow_Matching_for_Multimodal_Trajectories_Generation_in_End-to-End_CVPR_2025_paper.pdf)；已读官方 [goalflow_model_navi.py](https://raw.githubusercontent.com/YvanYin/GoalFlow/main/navsim/agents/goalflow/goalflow_model_navi.py) 与 [goalflow_model_traj.py](https://raw.githubusercontent.com/YvanYin/GoalFlow/main/navsim/agents/goalflow/goalflow_model_traj.py) | 目标词表的距离/可行域评分、目标条件flow已经存在；代码还有teacher目标与student目标配置，不能直接把其接口搬成无答案泄漏观测输入。此次只读，未执行。 |
| [Ready, Set, Plan!，CoRL2023](https://proceedings.mlr.press/v229/pavlasek23a.html)及[原文](https://openreview.net/pdf?id=5JMGq83yf1N) | 从目标样本建立目标分布并做联合轨迹推断已有先例。“不固定唯一终点”本身不新；本次未核验到官方实现，不声称复现。 |
| [Motion Planning Diffusion，作者原文](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JoaoCarvalho/2023-iros-carvalho-mpd.pdf)，§III–IV | 整条轨迹生成先验及代价引导已有先例；该文已知起终点/障碍代价、关节路径与本项目RGB-D语言任务路线的信息不同。不能借原文成绩或几何引导给新基线加答案。 |

这次短核验没有产生一个已论证的新核心。可行路径分布与约束表示值得研究，但要先填补实际缺少的观测生成强对照，并用下一轮结果决定真正需要改变哪一项；不是同时实现目标集合、学习代价场和约束投影。
