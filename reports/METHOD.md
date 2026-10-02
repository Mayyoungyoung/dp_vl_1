# 方法实现与证据边界

2026-10-02 20:23 UTC：新两排同图三目标语料上的16/32/64父普通集合回归均已实际完成1500步等曝光训练，结果仍混合；32父的固定last TRAIN只达到36.02%尖端有效，不能将全部误差归因于泛化。下一步是独立的[6000步普通收敛控制](OBSERVED_TWO_ROW_CONVERGENCE_PROTOCOL.md)，以及只读取已冻结first16 TRAIN池的[部分正对应机会审计](TWO_ROW_CROSS_GOAL_OPPORTUNITY_PROTOCOL.md)。二者都不构成新核心方法。后者必须同时具有稳定正参考关系和预测侧实质漂移缺口；同类型距离较近本身可能由类型定义保证，不能据此宣布机制成立。协议/源码已经准备，服务器正式执行结果尚未产生。

2026-10-02 18:20 UTC 更新：固定十二个旧开发父上的三种子迁移已完成。原规则 best 的平均 ADE 改善4.74%，仅2/3种子改善；固定末步 ADE 退化1.49%，不能据此建立核心机制。v6四个静态初态均通过，原失败布局取得23/27有效参考，但仍是旧开发几何。下一实际实验为一次登记的116个新父场景与普通集合回归，先用真实 TRAIN 参考核对表示容量，再读取独立 DEV 预测决定机制。跨目标路线对应关系只保留为条件假设；T-MPC已有不同目标间拓扑类别对齐和身份维护，不把对应、缓存或类型分配本身称为创新。详见 [固定迁移结果](OBSERVATION_LEGACY_DEV_TRANSFER_RESULTS.md)、[近邻核验](POST_K2_CROSS_GOAL_NEIGHBORS.md)、[正式采集协议](OBSERVED_TWO_ROW_FORMAL116_PROTOCOL.md)。

2026-10-02 17:35 UTC 更新（后文较早状态为历史）：Qwen6000步续训完成，best NLL降至.416633；同8TRAIN贪心端点仍0/8在3cm内，均值23.391→27.769cm，停止这组扩训及DEV/K4扩展。四布局v5完成108请求槽，61条有效参考（49已知类型、12unknown）、20路线失败、27setup未尝试槽；6/12目标条件有≥5已知类型，但仅为DEV_COLLECTION局部布局可采性。详见[物理结果](OBSERVED_TWO_ROW_LAYOUT4_V5_RESULTS.md)、[续训结果](VLM_SFT_CONTINUATION_6000_RESULTS.md)。下一实际工作是有界静态初始化修复与已冻结三种子模型在另一批12父上的前瞻迁移；未得到这些新结果，核心A–H仍未完成。

2026-10-02 16:09 UTC：支持事件位置的普通辅助基线已完成三个配对训练种子，原规则best宏ADE9.8548→9.0827cm、末端17.9283→15.6893cm。固定1500路径误差仅2/3种子改善，杯子均退化；没有把该辅助当新集合机制。9次真实Qwen条件前向审计通过单位/mask/shift/因果检查，但坐标占97.04% NLL；真前缀条件末端准确不能证明视觉定位。下一唯一SFT控制是同权重TRAIN8贪心生成。详见 OBSERVATION_MULTITASK_LANDMARK_THREE_SEED.md、VLM_SFT_TEACHER_AUDIT_RESULTS_V1.md。

当前新增普通基线修复：对每条正TRAIN参考，在首次open→close后的点与最终末端之间，选择离初始有效观测点云较近的一点，仅监督已有空间attention；无close或平局用末端。固定weight .02、sigma .025，不丢难参考、不改变free_offset末端标签、不把taskID或目标点传入forward。原普通头与辅助头参数、真实首输入初始输出及48k采样流已核验一致；实际收益须读配对训练结果。它是标准定位辅助，不是集合生成创新。

直接SFT的完整自由生成未产生正确路线；通用语法约束使固定8TRAIN全部格式有效，但目标仍0/8正确。该约束只限定K/H24/整数和event，不读几何、目标或参考，不做路径修复。下一诊断只分解已有模型的teacher-forced坐标拟合和因果对齐，不能把teacher-forced预测当作观测条件推理。

当前补充（2026-10-02 14:30 UTC）：真实Qwen直接序列分支已正式训练，K1/K4轮换、整集合随机顺序、不把未采到的模式当负例；答案token与观测prompt严格隔离。它是SFT对照，不是创新。相同checkpoint的independent4/whole4生成按全部失败/重复/超量槽及4次/1次完整编码计费；几何验收独立发生于完整产物落盘后。六任务普通头使用不受表面5cm边界约束的自由终点，避免空中任务被结构性排除；同DEV扩数据收益属于普通基线泛化。已测试的覆盖max、局部门控和局部草稿refiner均未建立稳定核心优势，不继续以换名保留这些主张。

当前决定（2026-10-02 16:50 UTC）：尚无获验证的核心集合机制。历史补全、局部编辑、区域分配和局部几何更新的负结果保留；最新 TRAIN768 单门关闭审计中，传统几何最远对/DPP 与响应参考最优全部达到100%条件 AnyValid，机制机会为零，因此也停止该设置的 joint-risk 方案。不能把 K4 的前两个查询当作训练过的 K2 强基线。实际 K2 普通训练已单独登记，评价器与旧结果不改，见 [审计](MULTIGATE_CLOSURE_OPPORTUNITY_RESULTS.md)和[预算基线](MULTIGATE_K2_BUDGET_BASELINE_CARD.md)。

目前保留的观测基线是真实冻结 Qwen + RGB-D + 当前状态的普通三维集合回归。六任务使用 free_offset 允许离开物体表面的终点；常规事件位置 attention 辅助在三种子原 ADE 选模下改善宏 ADE9.855→9.083cm，但固定末步只2/3改善，杯子仍有退化，不能成为核心创新。直接 Qwen 集合 SFT 已实际训练且自由生成失败；1500→6000同目标续训是一次有界基线修复，额外训练预算单列，未得到新质量结果前不称有效。所有参考路径和验收字段仍在模型前向之外。

当前新数据工作解决模式机会不足：物理 v4 单父中央目标得到8/9有效参考，其中5种实际侧向关系、3条有效 unknown；后者不自动变为负例。只有在独立布局、同图多指令的采集与普通模型预测完成后，再依据有效重复、缺失类型和成本确定新的候选机制。现阶段没有把物理可采性或传统部件组合称为论文方法。下文保留早期方法及其随后证伪记录。

更新：2026-10-02，覆盖受控实验 round1/round2。当前已经验证的是普通集合回归的监督目标纠偏；拟议的有效覆盖补全机制尚未建立优势。观测任务层另见 [OBSERVATION_READINESS.md](OBSERVATION_READINESS.md)，不得把本文件的 oracle 几何结果写成 Qwen 或机器人结果。

## 任务、条件和输出

研究问题是在相同任务信息、训练数据、候选数 K 和计算预算下，输出语义正确、几何有效、存在实际选择价值的路线集合。路线是高层末端路径，不等同于关节控制策略或机器人执行成功。

当前实现分为两种信息条件：

- 历史受控数据：冻结 CLIP 图文条件 1024 维，加真实起终点和几何 12 维；三类固定绕行路线。
- `multigate_v1`：34 维真实起终点及两面多开口墙参数；本轮路径头没有图像、语言编码器或 Qwen 输入。

`routeset/models.py::SetRegressor` 沿用历史结构：条件 MLP、K 个学习查询、3 层候选 self-attention/条件调制块、线性路线输出。宽度 192、4 个 attention heads；输出 24 个三维路点，其中 22 个内部点预测相对起终点线性插值的残差，首尾由已知起终点固定。multigate K=4 模型有 1,171,074 个参数。当前没有事件预测、由观测推断终点或可学习的显式预算分配。`k` 仅截取查询，现有结果来自固定 K=4 训练，不能据此称为跨 K 泛化。

学习查询、候选交互和 Hungarian matching 都是成熟集合预测组件，参考 [DETR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/832_ECCV_2020_paper.php)、[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) 和 [MTR](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)。本项目保留该模型为强基线，不把它重新命名为新方法。

## 已实现的正例分配纠偏

同场景有 R 条已知正参考 y_j、K 条预测 p_i。R 表示参考池规模；它不是一般连续规划问题的真实解总数。令 C_ij 为内部路点残差的均方误差。以下三种目标使用同一模型、同一正参考池和相同的逐对代价计算。

1. **subset**：随机选至多 K 个参考；不足 K 时随机重复参考补足，再做方阵一对一匹配。这是监督失配诊断对照，不应作为论文唯一强基线。
2. **positive**：R≥K 时对 K×R 代价做矩形匹配，让模型选择 K 个互异的已知正参考；R<K 时仍随机重复参考补到 K。
3. **saturation**：R≥K 时保持 positive；R<K 时求覆盖每个已知正参考至少一次的最小代价多对一分配。多余预测可选择任何已知正参考，不随机规定各模式重复多少次。

第三种目标可写为：

```text
R >= K:  min_a (1/K) sum_i C[i,a(i)],  a 为从 K 到 R 的单射
R <  K:  min_a (1/K) sum_i C[i,a(i)],  a 的像覆盖全部 R 个已知正参考
```

`routeset/train_v2.py::positive_assignment_loss` 对 R<K 精确求解该目标：先求每个预测最近参考的基础代价 b_i=min_j C_ij，再在转置后的机会代价 `(C_ij-b_i)` 上匹配 R 个不同预测作为各参考的代表，其余预测沿用最近参考。匹配索引停止梯度，选中的 C 反传；每条预测恰好计一次损失。它不会把未匹配预测判成不存在或无效，也不惩罚未采到但可能有效的解。

机制解释是避免给确定性查询施加互相冲突的随机路线子集或随机重复数。在 R≥K 时，随机子集的条件均值可能位于不同通道之间；在 R<K 时，随机重复数又可能推动多个查询在不同有效模式之间折中。纠正这两项是加强基线的必要步骤，不能据此宣称新的集合学习理论。

## 两轮实测及公平性

两轮均为 seed 0、batch 64、3000 次更新、AdamW 学习率 3e-4、权重衰减 1e-4、梯度裁剪 1，FP32。场景抽样 RNG 与分配 RNG 分离，保证同种子配对场景序列。每 500 步在同一 128 个 DEV_MODEL 父场景评估，按 `UniqueValid + 0.05*Valid` 选 checkpoint；锁定划分未用于选择。

| 目标 | Valid@4 | AnyValid@4 | UniqueValid@4 | ReferenceCoverage@4 | 选优步数 | 实际训练及周期评估耗时 |
|---|---:|---:|---:|---:|---:|---:|
| subset | 52.5391% | 75.7813% | 1.7734375 | 54.9967% | 2500 | 127.524 s |
| positive | 89.6484% | 100% | 3.2890625 | 72.2873% | 3000 | 134.846 s |
| saturation | 100% | 100% | 3.3984375 | 75.9332% | 1500 | 120.564 s |

数据、逐场景诊断见 [ROUND1_REVIEW.md](ROUND1_REVIEW.md)，结果来源为 `reports/v2_round1/results.json` 与 `reports/v2_round2/saturation_summary.json`。round1 代码为 `478c4bb6ae9e65a032d6b75266a8f19af56938c7`，round2 为 `1a1b6bb7600419504f5c1b2f5534435de7a1cf6e`。round2 从物理隔离的 development 数据读取，数据内容仍属同一 multigate_v1。

三项各反传 768,000 个目标槽；这不等于参考池访问量完全相同。positive/saturation 的分配可从全部 R 个参考选择，而 subset 先抽子集，这正是消融自变量，须明确披露。各配置均计算填充至 16 个参考的代价，round2 记录实际参考访问 1,175,991 次、逐对代价 12,288,000 个。费用另记训练实际秒数/GPU 占用时段，不能只用目标槽数宣称计算完全相同。

positive 的 53/512 个无效候选都来自 R=2/3；saturation 选优模型修复这一开发失效并达到当前 `mean(min(R,4))=3.3984375` 的类型上限。其第 3000 步最终模型 Valid 为 99.8047%，表格报告的是开发选优模型，不能将每步训练都表述为零失效。单种子、反复观察的 DEV 结果尚不支持确认性泛化结论。

saturation 的 batch=1 路径头中位延迟为 1.3405 ms、p95 为 1.3928 ms，测量包含已在设备上的几何条件下的前向/解码，不含 Qwen、图像处理、完整几何校验或评分。它不是端到端请求时延。

## 待验证的核心：有效覆盖条件下的路线补全

下阶段集中检验候选 A：模型读取已有路线，区分有效覆盖、近重复和无效草稿，再生成预算内的补充方案。拟议模块是任务条件路线编码、逐路线有效性门控与对精确重复幂等的 max 覆盖记忆；以相同路径上下文的普通 attention 补全为直接对照。详细机制、证伪条件与预算规则在 [METHOD_CANDIDATES.md](METHOD_CANDIDATES.md)。本文件不将设计图当作实现或结果。

必要检验包括空/单条/近重复/无效上下文、去门控与更换聚合、模型自生成草稿，以及固定总候选数和固定请求时间。受控层若使用精确几何门控，所有对照获得相同信息；观测层只能从 RGB/RGB-D、语言、相机和当前状态估计有效性。普通集合 attention、[条件 DPP](https://arxiv.org/abs/1207.6083) 与相关生成已有近邻，因此“读其他路径”或“减少重复”本身不构成新颖性。

本文件上述数值截至 round2；之后的补全实现、单种子结果及固定 checkpoint 干预见 [COMPLETION_REVIEW.md](COMPLETION_REVIEW.md)。该审查已观察到积极开发信号，但也证明普通 attention 在当前 `[A,A]` 情境同样幂等，不能将收益直接归因于这一性质。

后续观测链路已完成真实 Qwen 前向、32 父场景/275 条采集参考和四任务小批可采集性核验，详见 [DATA_CARD.md](DATA_CARD.md)。在线末两层 q/v LoRA 实现见 `scripts/train_observed_lora.py`：114,688 个适配器参数，实时 RGB+语言前向，无冻结 hidden 缓存；CPU 小型真实 Qwen 架构测试验证梯度、冻结基座不变和 optimizer/RNG 精确恢复。这些实现/采集证据不能代替实际 2B LoRA 训练与方法优势实验。

独立机器人设置、三种子新机制主结果和观测任务层的核心优势仍须分别核验。当前受控任务已接近饱和；后续价值应来自更难的条件补全/观测任务和清楚的质量—覆盖—成本取舍，而非继续把同模板分数作为论文成熟度证明。

## 后续实测决定

三种子coverage增益未稳定，详见COMPLETION_THREE_SEED.md。后期混合实际模型草稿显著改善两种补全头，但同模型2+2仍不超joint4，并付出两次前向；保留joint4作为空集合默认。没有把max、自产草稿混合或已有集合回归重新命名为新方法。

观测链路目前采用真实冻结Qwen mean+last特征、RGB-D反投影点和当前夹爪状态。观测点MLP与任务query产生空间注意力，anchor是实际可见点的学习加权坐标，普通集合头预测完整3D路线和open事件，终点为anchor加有界残差。所有新模块训练；RGB-D仅显式读取当前depth和相机字段。真实语义目标坐标仅用于评价。

`positive_endpoint_attention_loss` 将训练正参考终点附近观测点构成高斯混合分布，监督空间注意力，权重0.02、sigma2.5cm。混合不把多个不同终点平均到中间；teacher停止梯度，标签不进入前向，不按真值筛推理点。相同头/数据/1000步在三个种子分别使严格3cm正确率提高33.33/28.13/21.88个百分点。这是定位基线纠偏，不是路线集合贡献。详细公平性和边界见OBSERVATION_GROUNDING_THREE_SEED.md。

局部约束变化实验已经实现三个同规模对照：自由集合补全、最小编辑配对监督的完整预测、同监督的局部门控复制。累计候选按4+b记录，旧候选中的失败和重复不从预算删除。2000步首轮和公平续训至4000步均未支持门控：最终对b1/b2分别少0.06537/0.11800个有效类型，新增路线有效率也更差。放弃当前门控，保留full_free；全部失败、预算及恢复实证见CONSTRAINT_UPDATE_V2_RESULTS.md。

真实观测侧的定位能力仍不足以作为有效多路径系统。已加入TRAIN正参考监督的颜色原型RGB-D单终点对照，在已见精确指令上严格正确19/24与20/24，同时保留严重背景离群。这个专用控制不替代Qwen方法，不属于完整路线生成。新的在线RGB-D对照从同一个64父训练头出发，冻结/LoRA两臂同时训练全部新模块、对齐新增曝光；真实Qwen每请求重新前向，公共预训练成本独立列明。

该在线配对已完成，两臂都选共同step0，没有得到微调收益。随后针对注意力跨物体时软坐标均值落在空白处的问题，仅把anchor前向改为最高attention的有效观测点，反向沿原soft期望梯度；无额外参数、标签、修复或候选。natural64三种子严格定位从17.36%升至83.68%，参考ADE也降低。obstacle32三种子原选模结果不稳定，参考ADE更差。峰值坐标读取仅作为更强普通定位基线保留；完整方法贡献仍需在候选有效覆盖和成本上建立。详见OBSERVATION_PEAK_ANCHOR.md。
## 当前研究决定（2026-10-02 12:00 UTC）

以下历史补全/max/gate机制均保留为已证伪或不足的实验，不再作为待交付核心。当前仅检验[观测前段区域候选卡](OBSERVED_BRANCH_ALLOCATION_CARD.md)的表示前提：先从观测栅格构建区域，再用TRAIN正参考审计可分性；未实现联合预算分配。若区域没有稳定多分支，停止该表示。现有RGB-D局部读取不足首先应作为普通基线问题修复，不能把MTR式局部特征当创新。

随后两项实际结果均不支持继续该实现：72个观测边界审计全部单连通分支，放弃区域表示；等参数local/global草稿修正配对没有稳定质量—覆盖—成本优势，放弃局部更新。完整负结果见OBSERVED_DEPARTURE_REGION_AUDIT与OBSERVATION_REFINEMENT_PAIR。继续使用原普通peak和传统观测A*作为参照。当前下一机制尚在针对TRAIN预测中真实重复/缺失类型的诊断阶段，未声称已实现或有效。

多任务普通头兼容工作与机制贡献分开：抓取/抬升/取盖任务的最终末端位置可能不在初始观测表面，不能直接沿用reach的表面±5cm终点约束与终点attention标签。正在增加可选的无表面终点限制表示、按任务/父场景均衡训练及宏平均参考指标。没有语义目标验收标签时保留null，不用示范终点接近替代任务成功。
# Current implementation status — 2026-10-02 13:37 UTC

Six-task ordinary regression is now actually trained with real frozen Qwen, current RGB-D/calibration and state, using free endpoint offsets and task-parent balanced supervision. This repairs representation capacity for lifted endpoints; it is not a new set mechanism. Best500 of1500 steps gives TRAIN/DEV macro ADE1.783/17.121cm on12/12 parents, so more representative training is required. All unsupported semantic/validity/execution metrics remain null. See `OBSERVATION_MULTITASK_BASELINE.md` and actual artifacts.

TRAIN96 duplicate/reference diagnostics rule out same-scene repeated reference supervision as the current failure explanation. Only28 of288 instructions have valid duplicates plus known missing types; no mature MCL/facility-location reweighting is promoted to core novelty. Direct VLM whole-set/independent sampling is being built as a strong conventional baseline; real teacher-forced preflight is working after an equivalent chunked-loss memory repair, but full training and generation evidence are still pending.
