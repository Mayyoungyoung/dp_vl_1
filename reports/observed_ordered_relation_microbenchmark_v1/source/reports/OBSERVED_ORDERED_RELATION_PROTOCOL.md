# 有序观测关系损失：数学实现与三臂预登记

2026-10-03。**当前只完成独立损失、合成测试和资源微基准入口，没有训练器、服务器执行或新模型结果。** 单一候选是“完整正参考内部的有序局部观测关系参与集合匹配”；它尚不是已建立的新颖核心。标准 DTW、Soft-DTW divergence、Hungarian/saturation、局部距离描述均不能单独称创新。原 segment v1 保持 stop_underpowered，不扩其父集合或改门槛。

## 三个臂及精确尺度

原生成器、Qwen 缓存、观测点编码器、K4/H24、端点/事件表示和 grounding 损失不变。A 调原 `routeset.train_v2.positive_assignment_loss`，不改历史函数。B/C 只产出 `[B,K,R]` 完整路线代价，再用与原 saturation 相同的指派；R<K 时覆盖每个已知正例后允许重复，R>=K 时在全部正例中矩形指派。unknown 类型正例保留，不将参考条数当解数，不给未匹配预测存在性负标签。

原 `train_observed_geometry.py` 实际用 `xyz[:,:,1:]` 和 `.2*open[:,:,1:]` 的四维均方误差；已知起点不再次监督。因此固定 XYZ 权重3/4、事件权重1/4，事件继续只比较后23个位置。B/C 的完整24点单调路径从(0,0)到(23,23)，只允许右、下、对角步；一次对齐绝不跨两条参考拼接片段。

令 s=0.10m、γ=0.10，

`c_B(i,j) = mean_xyz[((p_i-r_j)/s)^2]`。

用相同递推 `Sγ(C)`，三组成本均独立构造：

`D(p,r) = Sγ(Cpr) - 0.5*Sγ(Cpp) - 0.5*Sγ(Crr)`；

`L_kr = (3/4)*s²/23*D(p_k,r) + (1/4)*mean_{i=1..23}[(.2*(open^p_i-open^r_i))²]`。

这样在起点误差为零、仅取对角对应时，XYZ及事件的系数与原四维米制MSE相等；**没有声称整个Soft-DTW目标与原MSE相等**，软路径长度与self correction本来就是B的区别。尺度s对应米制局部代价温度γs²=.001。B/C同s、γ、归一化和事件项，不扫温度/尺度/权重。XX两侧预测都参与梯度，不能detach任何一侧；YY及参考描述是固定监督，允许detach。

## C 唯一新增项与信息边界

每条H24路径取23个相邻段中点及末端，共24个有序描述位置。在每位置加 `[0, ±x, ±y, ±z]` 七个固定世界坐标偏移，偏移长度0.05m。每个查询到**全部当前可见有效RGB-D点**求最近平方距离 d²，描述分量为 `exp(-d²/(2*.05²))`。中心查询距有效点<=0.15m标“附近有已观测表面支持”；这不是查询点可见性、障碍内部/外部、free-space或碰撞证书。未观测区域为unknown。

`c_C(i,j) = c_B(i,j) + 1.0 * support_p(i)*support_r(j) * mean_7[(v_p(i)-v_r(j))²]`。

Cpr、Cpp、Crr各用自身两侧的支持函数计算，不能将cross mask塞入self项。未知mask只去掉描述差，XYZ始终保留。硬支持半径与最近点切换使C只**分段可微**；Gaussian值到预测坐标的梯度真实保留。选择最近点索引在no_grad块进行，然后对选中点重算平方距离并求导，这是远离最近点切换处的同一梯度，不是detach预测描述。无下采样/近似近邻/地图填充。

描述入口只接 `world_xyz, valid_mask`，不会接真箱体、mode、guide、验收标签或目标坐标。观测点来自原白名单backprojection；预测/参考只是该损失的两端，参考永远不进入生成器条件。描述不使用Qwen特征和可学习网络，预测端每步重算，不能复用旧预测缓存。

允许在TRAIN损失侧缓存与模型无关的逐输入参考描述。`prepare_reference_descriptors` 绑定完整参考/有效mask/观测点/策略SHA；`batch_reference_descriptors` 按已抽取ID组装且逐行核原字节，重复抽到同输入可复用，不另抽样。缓存不得用于DEV forward或生成条件。归档缓存构建成本及完整源hash。

**限制必须实报：** C可能通过移动到低支持区域减小辅助项；这不自动表示自由空间。报告预测/参考支持率、负代价数量和所有原指标，不以辅助损失下降过研究门。有限七方向表面响应不是拓扑完备描述，也不能保证同描述意味着同通道或免碰撞。共同已知起点与参考终点仅作为原训练路径监督，推理端不增加真实终点桥。

## 原文与官方实现核验

| 已核验的一手来源 | 机制及本实验边界 |
|---|---|
| [Soft-DTW，ICML2017](https://arxiv.org/abs/1703.01541)，[作者递推实现](https://raw.githubusercontent.com/mblondel/soft-dtw/master/sdtw/soft_dtw.py) | 软单调对应是成熟工具，本模块以反对角波前向量化实现同一递推，无新对齐算法主张。 |
| [Soft-DTW divergence，AISTATS2021论文](https://proceedings.mlr.press/v130/blondel21a/blondel21a.pdf)，[官方numba_ops](https://raw.githubusercontent.com/google-research/soft-dtw-divergences/master/sdtw_div/numba_ops.py) | B采用XY减半XX/YY的标准self correction；原文Table1/Prop4只给普通平方欧氏X=Y驻点，其普遍非负性未在该条件下证明。Prop3的特殊cost/1D结论不能挪给本B/C。C带pairwise硬mask，更无一般divergence定理；两臂均不clamp负值，不宣称严格度量。官方梯度保留完整XX两侧。 |
| [DILATE，NeurIPS2019原文](https://arxiv.org/abs/1909.09020)，[作者loss](https://raw.githubusercontent.com/vincent-leguen/DILATE/master/loss/dilate_loss.py) | 已组合可微shape alignment和时间扭曲项；本轮不另加其时间penalty，不把顺序/时序监督称新颖。 |
| [Action-Constrained Imitation Learning，ICML2025](https://proceedings.mlr.press/v267/yeh25a.html) | DTWIL以MPC对齐受动作约束的替代示范；已是约束与轨迹对齐先例。本轮不生成替代示范、不做MPC。此次已读正式原文入口摘要，未运行其代码。 |
| [Trajectory Learning using HMM and DTW，2012原论文入口](https://ieeexplore.ieee.org/document/6166903/)，[RGB-D composite SDF，ICRA2025作者项目](https://stalhabukhari.github.io/icra25-sdf-dyn-nav/) | 前者已有机器人示范对齐，后者用观测几何距离梯度优化路径；距离场/障碍关系本身不新。后者并非本候选的完整正参考集合训练损失。本次为有限邻近核验，不宣称排尽同机制或已复现。 |

同时沿用已有MTR/BAAT核验：[前一决策卡](NEXT_CORE_AFTER_BUDGET_CEILING_DRAFT.md)。MTR已有route-local attention；若C只胜旧全局头，仍缺同信息强局部生成控制。当前没有证据确认完全相同的“masked observed relation + full-reference self-debiased set loss”已被排除；应将其视为一个可证伪的辅助目标候选，不能提前写方法贡献成立。

## 实际形状、内存与微基准门

原composite配置 `geometry_preprocessing.sampled_points=12544`，来自224×224 RGB-D stride2；归档TRAIN285行参考总数1663、Rmax=9。**不是512点。** 合成微基准固定B32/K4/R9/H24/N12544/float32，三臂各2次warmup+3次记录；全量点参与，query chunk64。最近点搜索不保留完整N轴autograd图；选中距离再求导。DP包含全部BK R的XY、BK的XX、BR的YY及CPU saturation指派和backward。参考描述预计算单列实际时间/峰值，缓存身份校验仍在每次C计时内。

入口 `routeset.observed_ordered_relation_loss.synthetic_microbenchmark(device)` 返回结构化测量，不加载数据/checkpoint/model，不做优化/模型forward，不自动训练。CPU/GPU/线程/35%显存由root冻结wrapper施加，所有源、实际Torch、外层时间及异常记录。必须在真实Torch下先过全部数学测试且0skip，再测真实GPU峰值/时间；只报tiny通过不足以放行。每臂返回3个原始耗时、峰值allocated/reserved、有限梯度和loss，3000倍耗时仅是**损失计算投影**，不含生成头、grounding、optimizer、DEV或启动，不能冒充全训练预计。若越内存/时间预算，先记录失败再做保持数学/输入等价的实现优化，不能自动缩N、换γ、减参考或减少任一臂曝光。

当前本地Python无Torch：纯配置/索引测试3 passed，真实Torch数学17 skipped；py_compile通过。真实资源成本尚未知。建议root读微基准后才决定是否实施三臂各3000步，不预填GPU小时。

## 后续训练器必须保留的公平与判据

三臂从同一composite last12000模型/Adam/实际存在的scheduler/采样器/RNG恢复到新输出，不能用旧A结果代替本轮A。全部285TRAIN/1663正例、同批ID/参考mask、同384000路径状态/臂，保留所有unknown与长正例；每250步共12次旧36DEV选择，固定last主要比较，best另保留。固定起点既往成本单列不三次计；参考描述构建与DP训练增量完整计时。同曝光不等于同FLOPs/时长，不能宣称同时间优势。新variable数据、extension新DEV、score/calibration/locked均不读。

A使用原loss；B/C `ordered_relation_costs -> saturation_loss_from_costs`；grounding项权重/正例endpoint监督保持原值。新函数没有任何参数可优化、没有额外候选或检查器调用。未来trainer需记录每步XYZ/self/event/grounding与支持统计、实际梯度/参数更新、完整checkpoint与请求账本，不静默重发失败训练。

固定last继续门保持原决策卡：C−A及C−B TipValid至少+5pp、已知Unique不降、语义下降不超过2pp，目标正确子集TipClear改善；报告全285TRAIN及12DEV父逐场景差、unknown有效和全部失败。这是单seed开发筛选而非显著性标准。B≈C则降级普通曲线控制；只TRAIN改善不支持泛化；只增加unknown且已知覆盖下降不支持质量—覆盖主张。通过后仍需对应打乱消融、强局部生成对照及多种子/独立设置，不在本轮叠加模块。
