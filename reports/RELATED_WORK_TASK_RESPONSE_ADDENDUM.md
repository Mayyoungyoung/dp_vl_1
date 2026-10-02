# 任务变化响应与集合预算：原文核验增补

2026-10-03。增补[原近邻矩阵](RELATED_WORK_MATRIX.md)与[当前方法](METHOD.md)，不更改历史结果。以下均为阅读原论文/官方论文集的机制核验，**没有在本项目复现这些外部方法**。

| 一手来源与状态 | 原文具体机制 | 本轮候选的差异与边界 |
|---|---|---|
| [ModeSeq原文v2](https://arxiv.org/html/2411.11911v2)，[CVPR2025论文集](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.html) | §3.3–3.6：先前模式嵌入作为Memory Transformer上下文，逐个预测模式；EMTA优先监督最早匹配，重复模式让位；层间重排与模式数外推。 | 读取已有路线、动态K、顺序模式、减少重复本身已有直接先例。当前审计只问约束改变时的共同失效；不重写一个相似解码器。未核验作者实现来源，不声称无代码。 |
| [LookOut原文v3](https://arxiv.org/html/2101.06547v3)，[ICCV2021正式论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_LookOut_Diverse_Multi-Future_Prediction_and_Planning_for_Self-Driving_ICCV_2021_paper.pdf) | §3.2式7–8：通过不同预测交通未来所诱导的自车计划距离奖励采样器，用REINFORCE处理规划不可微；§3.4输出共享初始动作和不同长期应变计划。 | “对下游决策有用的多样性”并非新概念。自身任务路线在语言/几何变化后能否保留，区别于预测交通未来，但仅更换领域或相似度不能证明创新。此次未核验可追溯作者代码，未把第三方LookOut同名库当官方实现。 |
| [DSF/DPP原文](https://arxiv.org/html/1907.04967)，[ICLR2020作者机构存档](https://www.ri.cmu.edu/app/uploads/2020/04/Yuan_DPP_forecast.pdf)；[DPP基础原文](https://arxiv.org/html/1207.6083) | DSF学习场景条件相关潜变量集，经固定生成器获得多条预测，以DPP期望基数衡量多样性；DPP基础包含质量、固定大小与条件子集。 | 改成路线响应核仍可能只是传统多样化适配。审计采用单位质量RBF K2 MAP作为明确传统控制，数学上等于最远对，不冒充DSF神经训练复现。 |
| [SetPO正式论文](https://proceedings.mlr.press/v306/li26jk.html)，[原文](https://arxiv.org/html/2602.01062v1)，[作者代码](https://github.com/chenyili0818/SetPO) | ICML2026；集合核效用的leave-one-out多样性边际作为GRPO/GSPO/DAPO优势修正。 | 集合边际贡献不新。普通确定性路线头没有可直接使用的序列策略概率；当前不引入DPO/SetPO，也不把最大覆盖重加权更名为策略优化。 |
| [Finding disjoint paths in networks with star shared risk link groups，出版社](https://www.sciencedirect.com/science/article/pii/S0304397515001097)，[作者INRIA记录与原文链接](https://www-sop.inria.fr/members/David.Coudert/Biblio/Year/2015.complete.shtml) | Theoretical Computer Science 579:74–87，2015；明确以同组资源同时失效定义SRLG，k-diverse routing寻找两两SRLG不相交路径。这里核验出版社摘要及作者记录，不声称已复查全部复杂性证明。 | “若干不同路线不能被同一物理变化一起破坏”已有成熟算法问题。将开口作为共享失效资源或用最大覆盖近似，不构成本项目创新。 |

“共同失效更少”也与传统共享风险路径选择相近：仅两条路径不同，并不意味着它们不依赖同一失效资源。[IETF路径多样性规范草案](https://datatracker.ietf.org/doc/html/draft-ietf-pce-association-diversity-12)明确区分链路、节点和shared-risk-link-group路径多样性。此处只用其解释成熟概念，不把网络协议当机器人算法实证。

当前唯一行动是[TRAIN机会审计](MULTIGATE_CLOSURE_OPPORTUNITY_CARD.md)：检验实际joint4、其明确未适配prefix2、传统几何参考池选择与标签辅助共同失效最优二元组之间是否有空间。响应最优只是有限参考池上的最大覆盖控制，**并未形成新方法**。若传统几何选择已经达到它，应该记录这项否证；若仍有缺口，也必须先训练真正K2 saturation强基线，不能靠截断K4建立优势。

特别是当前两墙完整笛卡尔门对，几何最远对很可能已选出各墙两端不同开口，达到任一可解单门关闭的AnyValid上限。这是待正式TRAIN审计核实的结构性推断；若成立，应判为当前设置不能区分拟议机制，而非训练神经目标击败弱随机选择后宣称创新。

以后若要提出实质机制，需要证明如何从不完备正参考与真实观测推断“当前已保留路线仍缺什么”，并在相同完整候选数/时间/信息下超过这些成熟控制。当前数据只允许证明受控机会或否定它，不能跳到真实机器人解集补全已成立。
