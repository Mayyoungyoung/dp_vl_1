# 预算条件集合：有限设计审查（2026-10-02）

本次审查只保留一个低成本候选：**显式K条件的普通集合回归**。它由共享解码器、K嵌入和已有正例集合匹配组成，应当作为强基线，不构成新方法。先补齐多样化生成对照，再判断是否存在尚未被普通模型解决的预算分配瓶颈。本次没有安装、运行或复现外部方法。

## 最接近的方法及核验边界

| 近邻 | 原文/官方来源与实际机制 | 对本项目的约束 |
|---|---|---|
| MTR | [NeurIPS2022原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf)，[官方实现](https://github.com/sshaoshuai/MTR)。已读固定tree `a5ba7bdafa09a1a355cc34f8a895499a2b14ddb3` 的decoder/config；配置用64个意图中心，最终NMS留6，decoder先生成全部轨迹。 | 意图查询、局部细化和去重均已存在。原始64→6必须记64个生成候选；缩为K查询是需要重新训练并披露的适配，不能借原论文成绩充当严格K基线。 |
| LED | [CVPR2023原文](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf)，[官方实现](https://github.com/MediaBrain-SJTU/LED)。已核验tree `aae048a85292a24c2218de5fe9b4d20d3542bf04` 的initializer；联合预测均值、方差及相关样本位置，默认 `k_pred=20`，随后少步去噪。 | 相关候选与学习分配已存在。把K条件加入联合初始化器仍是组合基线；20条再选K条不得报K预算。 |
| DSF/DPP | [ICLR2020作者原文](https://arxiv.org/html/1907.04967)，第4.2节明确N是采样预算；DSF输出N个相关潜变量，DPP期望基数用于优化质量与多样性，原核使用欧氏距离。 | “固定小预算覆盖少数模式”已是原问题。换成路线类型指标是必要评价，不能自动成为新机制。本次未核验到作者发布的DSF独立代码；检索到的第三方改写不当官方来源。 |
| DIVA | [原文](https://arxiv.org/html/2302.03462)，[作者机构PDF](https://cedric.cnam.fr/~thomen/papers/ICPR22_Calem.pdf)，ICPR2022。针对普通DPP只增加纵向差异的问题，设计横向末端差异核并惩罚离开可行区域。 | 任务特定多样性核加几何有效性也已有近邻，不能只把欧氏排斥改名为语义多样性。未运行其实现。 |
| ModeSeq | [CVPR2025原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.pdf)，[可读原文v2](https://arxiv.org/html/2411.11911)。Memory Transformer读取前序模式，顺序生成候选；EMTA优先最早匹配；支持改变解码次数并展示6→24模式外推。 | 是“读取已提路线后继续补全”和动态候选数的直接强近邻。EMTA的其它模式负标签服务单未来预测评分，不能照搬成“不匹配参考=路线无效”。原文有层间重排，不应声称其多层版本具有严格跨K前缀不变性。已查原文、作者主页与公开仓库信息，本次未定位可核验的作者ModeSeq实现；不据此断言代码不存在，也不称已复现。 |
| 嵌套/前缀预测 | [Nested Dropout, ICML2014](https://proceedings.mlr.press/v32/rippel14.html)训练可截断的有序表示；ModeSeq是更直接的模式序列近邻。这里“嵌套多假设”仅指集合前缀约束，并非已核验某篇同名论文。 | 共享前缀、随机K训练、前K输出、多尺度损失不应单独作为新意。嵌套输出限制不同K只能追加而不能重新分配；允许重新分配具有可测试差异，但K嵌入本身非常常规。 |

## 唯一预算候选：K条件普通集合基线

输入仍为同一观测条件或受控条件，再显式加K=1/2/4/8的嵌入，只实例化K个查询，输出K条完整路线。推理不生成隐藏候选，不后选大池，不读参考条数。训练每轮使用相同父场景、同已知正例池、同K日程、同目标槽数；仅比较有/无K嵌入，保留已有饱和正例匹配。不完整参考中的未匹配输出不被训练成不存在，也不用示范条数预测真实总解数。

相对MTR/LED的实际差异是显式把请求预算交给同一个解码器，且不先产生固定大集合；相对严格嵌套序列，允许同一查询在不同K时改变路线；相对DPP，没有额外排斥或势函数。以上均不足以构成论文贡献，只作为后续新机制必须战胜的对照。

可证伪假设：在同日程和槽数下，显式K条件改善低K的有效率，同时高K保留覆盖；若相对不加K的同一动态查询模型没有稳定配对收益，则删除K嵌入，不将普通变长输出包装成方法。开发选择只能看DEV_MODEL，旧固定K4权重的K1/2截断只能是零适配诊断。

不建议现在堆叠预算路由、前缀一致性、DPP、覆盖记忆四个模块。当前受控普通回归已强，观测数据仍受语义/路径有效性和不完整模式覆盖限制。近期最有区分力的动作是补齐同数据、同目标槽的独立扩散与集合扩散，并在同独立权重上测试清楚命名的Particle-Guidance-inspired适配；具体实现卡见 [DIFFUSION_MULTIGATE.md](DIFFUSION_MULTIGATE.md)。
