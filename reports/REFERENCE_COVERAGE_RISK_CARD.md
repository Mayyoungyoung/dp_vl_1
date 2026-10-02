# 有限预算覆盖：先核实遗漏来源，再考虑一个常规目标对照

2026-10-02。状态：**尚未证明新的核心机制；不启动训练。** 本卡只提出一个可证伪候选。TRAIN 已保存预测审计现已完成，详细证据见 [新96审计报告](OBSERVED_REFERENCE_COVERAGE_TRAIN96.md)。入口 `scripts/audit_observed_reference_coverage.py` 不加载模型、不搜索路线、不访问 DEV/locked 原始样本。

实际决定：新96共393个有效已分类重复候选，但只有28/288指令、27父同时缺少已知正参考类型；特权参考替换机会总30个类型（0.10417/指令），不等于模型可实现提升。560条参考全部Tip有效，192未知，同指令已分类重复参考仍为0。48个匹配到遗漏类型参考的候选有40个走成另一有效类型，说明普通坐标拟合与类型覆盖可能错位；不证明无监督该参考、matching跳变或坐标均值因果。按同场景重复参考去重的前提被否定。覆盖风险仍降级为成熟基线候选，未实现或训练附加目标。

## 已纠正的前提与停止决定

旧32 TRAIN 的181条参考确实有频数不均：negative_x48、positive_y57、positive_x6、未知70。但94个有参考指令的参考条数分布为1/2/3条=28/45/21，全部少于K4；同指令内**已分类重复参考数为0**。已知不同类型数0/1/2/3对应21/38/32/3个指令。70条未知类型分布在51个指令，不能合并成一个类型，也不能认为是70种不同路线。跨场景采集频数不均不能推出同场景重复参考挤占匹配槽。

`routeset/train_v2.py::positive_assignment_loss` 的 saturation 分支在参考少于K时已经要求每条参考至少匹配一个候选，多余候选可匹配最近正例。它没有用未匹配候选监督不存在。因此“换成正例集合损失”本身不是未实现的机制。

旧32 peak last1000 的单参考指令仍有44/112候选碰撞，其中39个终点语义正确；多参考均值不是这些碰撞的必要解释。两参考指令72/180碰撞，三参考48/84碰撞。这些只是分层关联，不能证明模型在取目标均值：共享参数跨场景欠拟合、表示不足、优化、分配跳变都尚未分离。原诊断 SHA256 为 `2e3420bb5e99b61f63c481aec381d7d665ae6fcf62ac94e9ef69831354642ebf`，原始结果不改。

普通局部 refiner 分支已停止。新96同step3000，local草稿Tip46.88%/Unique0.958高于原peak last43.75%/0.875，而更新后回到43.75%/0.875（修0坏3）。best只有修1坏0及总计+1类型；两阶段ADE下降没有稳定有效性收益。没有依据把失败归因为联合训练草稿退化，也不为这个常规模块追加冻结基座训练。

## 近邻核验与可主张差异的边界

下列均阅读一手原文；没有在本项目复现这些外部方法。“balanced subset coverage”在本卡指一类标准控制，不冒充已核实同名论文。

| 工作及状态 | 与当前问题直接相关的机制 | 此候选的差异与不能创新部分 |
|---|---|---|
| [ModeSeq原文v2](https://arxiv.org/html/2411.11911v2) §3.3–3.6；[CVPR2025论文集](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.html) | 顺序模式记忆、EMTA按最早匹配监督，层间重排，模式数外推。 | 下一候选读已有路径、动态K、让重复让位，都已有直接先例。本卡不增加顺序解码器；讨论多条不完备正参考的覆盖风险。未定位可核验作者实现，不等于宣称没有代码。 |
| [DSF/DPP原文](https://arxiv.org/html/1907.04967) §4.2；[ICLR2020作者机构存档](https://www.ri.cmu.edu/app/uploads/2020/04/Yuan_DPP_forecast.pdf) | 场景→预算N个相关潜变量，DPP期望基数优化；其潜在质量项主动避免偏好高频模式，并允许模式少于N。 | 频数不均和质量—多样性取舍不是新问题。拟议项衡量对已知正参考的覆盖，而非输出之间的排斥；这仍需DPP-inspired控制。没有把第三方代码当官方实现。 |
| [SetPO原文](https://arxiv.org/html/2602.01062v1) §3.2–3.4；[ICML2026/PMLR306](https://proceedings.mlr.press/v306/li26jk.html)；[作者仓库](https://github.com/chenyili0818/SetPO) | kernel局部质量、leave-one-out集合多样性边际，加入GRPO/GSPO/DAPO优势；已读作者[训练入口](https://raw.githubusercontent.com/chenyili0818/SetPO/main/scripts/run_setpo.sh)。 | 稀有候选边际贡献不是新概念。确定性坐标回归器没有该策略概率，不能直接称SetPO/DPO。此卡只有已知正例支持的可微损失，不声称策略优化复现。 |
| [Lin & Bilmes，ACL2011原文](https://aclanthology.org/P11-1052.pdf) §4.1–4.2 | facility-location覆盖、饱和覆盖和簇内边际递减，在有限预算下优化代表性/多样性。 | 按类型封顶、先覆盖未覆盖簇、贪心边际收益都是成熟概念。仅用路线替代文档或神经网络近似这些目标，不足以成为核心方法。 |
| [Multiple Choice Learning，NeurIPS2012原文](https://proceedings.neurips.cc/paper_files/paper/2012/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf) | 对每个监督目标由最合适预测器承担损失，是最小距离覆盖的直接概念对照。 | 本项目已有多查询匹配，另加min-over-candidates项仍可只是MCL/Chamfer式目标重加权；需要同监督强基线证伪，不能靠更名主张。 |

## 唯一候选：正参考支持的集合覆盖风险（先降级为成熟基线候选）

具体不足假设：在同一场景已采到多个有效类型时，平均坐标匹配误差继续改善常见/容易路线，但实际K4输出中某已知类型仍失效或遗漏，多余槽重复其他有效类型。这里的“遗漏”只涉及已证实正例，不意味着参考以外无解。

候选保留当前K4头、输入、一次输出和原质量损失，只考虑一个附加的**按已知正例类型封顶的覆盖风险**。设训练已知类型组为G，整条参考为r，输出为p_j，a(p_j,r)是预声明的整条路径相似度，则

`L_coverage = mean_{g in G} [tau - max_{j<=K} max_{r in g} a(p_j,r)]_+`。

仍保留所有未知类型正参考的原质量损失；未知类型不进入类型封顶项，不被判无效，不被用于监督总解数。类型来源若使用采集/验收标签，所有相关对照获得相同训练标签，推理不得输入类型、guide、箱体真值或目标坐标。相似度只辅助优化，不替代原Tip有效性和物理绕行类型评价。

该形式对同组完全重复参考幂等，已覆盖组不持续增加覆盖收益。它在概念上属于有阈值的facility-location/MCL覆盖目标，**目前不能成为论文核心贡献**；相比普通全参考matching的实质变化主要可能是难正例/类型重加权。若采用它，必须先认真比较相同类型标签的普通balanced subset/等类型加权质量损失。给新目标更多标签、更多完整路线或更多曝光会使结论无效。

尚未设定tau/相似度尺度，不能从DEV选择。当前已知参考每指令少于K、未知多，而且saturation已覆盖每条参考，故没有理由立刻实施这个附加项。若审计发现大多数重复场景已覆盖全部已知类型，就直接否定现数据上此候选的可检验空间，不强行惩罚合理同类有效变体。

## 已完成的最小TRAIN审计

范围固定为已注册的新96 TRAIN父272000–272095、全部288指令，包括无正参考指令；使用ordinary peak best原始保存的 `train/predictions.npz`，K4/H24。本地该预测SHA256为 `4d8fa5ff90d5fa4c4261c003df40df167b41ce7c43dfcf721a48386a3c94aa58`，服务器必须一致。读取预测并核验完整ID、父ID、shape、hash后，物理几何和参考仅进入诊断。

1. 分开统计有效已分类重复、未知有效，以及“有效重复且存在已知未覆盖正参考类型”的场景与父场景。用 `min(重复槽数, 已知遗漏类型数)` 给出参考替换的标签辅助机会量；这不是可实现模型增益，也不覆盖未知全部解。
2. 复现原saturation最小代价指派，计算第二个不同指派的代价差；另外保存每候选第一/第二参考误差差，以及是否来自两个已知不同类型。连续差值不设DEV阈值，不把低差值直接判为mode averaging。
3. 对同一指令正参考，在原H24对齐下固定alpha=.25/.5/.75做线性插值，分别报告不同已知类型、相同已知类型、未知关系的碰撞探针。原始两条参考必须分别有效；探针绝不进入预测K4或训练。插值穿障只说明坐标平均可能危险，不能证明模型采用了该均值。
4. 以单参考指令碰撞、无参考指令、已知类型已全部覆盖但重复等分层，排除把所有碰撞都归因于多参考冲突、把所有重复都归因于分配失败的叙述。

本地5项测试已通过（0.75s）：精确指派与穷举一致、单阶段重采样与实际训练函数逐元素相等、未知类型边界、两条有效路线的穿障插值、完整预测shape/父身份拒绝。固定5fb74源码的服务器5测也已实际通过（0.21s），CPU1正式审计exit0、内部2.18727s、模型前向0、GPU小时0。完整job/source/input/复制哈希在报告索引中；没有新增训练。

固定release中的实际审计命令如下（原输出已存在，复算须换新的output；完整record_job包装见结果索引）：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python scripts/audit_observed_reference_coverage.py \
  --data /home/wzy/dpvlm/route_set_v1/data/observation_obstacle_reserved_development_v1 \
  --run /home/wzy/dpvlm/route_set_v1/runs/observed_obstacle_reserved96_v1/peak_seed0 \
  --reservation configs/observation_partition_reservation_v1.json \
  --expected-prediction-sha256 4d8fa5ff90d5fa4c4261c003df40df167b41ce7c43dfcf721a48386a3c94aa58 \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_reference_coverage_train96_v1
```

下一项决定必须来自这份审计：若有大量标签支持的替换空间，先做成熟balanced质量目标的强对照；若指派已明确而指定候选依然撞障，优先承认生成质量瓶颈；若未知参考使原因不能辨认，报告不可判定并改进数据证据，不把不完备解集硬补成完整分类。任何进一步机制都应提供超过这些成熟对照的可证伪差异后才训练。
