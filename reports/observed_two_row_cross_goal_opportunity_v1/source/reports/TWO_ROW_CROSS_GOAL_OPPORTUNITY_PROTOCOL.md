# First16 TRAIN 跨目标正对应机会审计（执行前协议）

2026-10-03。本协议只登记一个必要性检查，不登记新模型或训练授权。当前代码仅经本地纯 NumPy 测试，尚未在服务器执行审计。

## 为什么检查，什么不构成证据

固定 first16 的 last1500 在 TRAIN 有168/192条 TipValid，DEV33/144条；训练正例可明显拟合，但泛化不足。DEV有效已分类重复为0，因此去重不是当前神经基线的主要解释。A*的有效重复是另一生成器的缺口，不能移作神经机制证据。新增 first32 固定1500步 TRAIN 有效率只有36.02%、saturation loss约.00298756，存在明显欠拟合；不能将其全解释为泛化，普通收敛控制另行实施。本审计不打开 first32/64 新 TRAIN 数据，也不利用新结果修改门槛。

唯一问题是：在同一初图/current、只改变目标指令时，原正参考是否存在可复用的空间关系，而普通模型的目标正确路线是否出现超出参考正常变化的漂移，并与碰撞共同出现？这只是描述性机会检查，不能证明目标改变造成碰撞，也不能证明某种训练机制能修复。

## 固定范围和真实性

- 仅父283200–283215，48个已存在TRAIN输入、285条全部原正参考。每父三个目标形成3个无向对，共48对；有向是96对。不得把48输入、48无向对和96有向对混称同一分母。
- 保存的普通 best500 TRAIN192槽和 last1500 TRAIN192槽都完整读取，原checkpoint、config、summary、driver receipt、last审计report及两个NPZ精确SHA固定在 `configs/two_row_cross_goal_opportunity_v1.json`。不加载模型、不forward、不调用规划器、不新增候选。
- 原export_manifest精确SHA固定；输出metadata四文件整体只做字节hash。JSONL先读取规范的id首字段，只解码固定48个TRAIN行；其余行保持不透明，不解码DEV监督内容。原始文件只允许位于对应已选 `parents/TRAIN/<parent>` 内，跨父、越界、符号链接逃逸拒绝。
- 复用collector的机械注册/布局门禁和两排原评估器；可读取其他角色的机械receipt，但不打开其图像、轨迹、目标或模型指标。不能调用会hash全部DEV raw的 `verify_export` 全量路径。
- 对固定TRAIN读取current、evaluation-only箱体/目标/route config和正参考，核原manifest SHA、几何readback及原类型函数。预测后的检查仍是tip线段2cm、目标身份且3cm、起点5mm、reach恒定事件；独立重算best/last TRAIN三个核心指标须逐项对齐原记录。
- 同父三个目标须共用相同图像字节和同一current文件。末尾复核所有已读来源SHA和机械gate。CPU1、CUDA隐藏、GPU小时0，输出fresh目录；失败留record_job日志，不能绕过检查。

## 空间描述和歧义保留

参考分别报告原raw与现有 `resample_event_segments` 的model-H24。官方分类函数对每种表示重新分类，同时保留原raw类型以及重采样是否改变类型；原raw必须与冻结标签一致。H24异常不删除参考，也不偷偷更新原标签。无论known/unknown，每条正参考都保留长度、最大高度、原tip检查结果和空间描述。

穿越描述用轨迹与两排真实row plane的交点，不用guide编号、query槽或H24相同下标。记录全部交点的行号、方向、顺序、XYZ和自身累计弧长比例；共享顶点交点只计一次。平面内线段、切触、缺行、反向/多次穿越、首末点未跨越两排各有明确标志。仅每排恰好一次向前穿越、顺序0→1且无歧义时，才计算两个交点的横向y/相对柱顶z差。其余保留并报告不可评价，不将unknown当负例或重复。

raw或H24长于2m只加 `long_arc` 描述标志，报告其总数和仍可进行单次穿越比较的数目；不删、不截断、不改采样。另按自身弧长各取33点，扣除各自首末点弦线后计算形状差异，**仅描述**，绝不将残差拼接回另一目标或做路径修复。弧长对齐也不是已证明的物理对应，因此此量不进入主要机会筛查。

同父同目标对内，先按已知类型对汇总其全部可评价轨迹对，再对类型对等权平均；最后保留每个目标对和父等权汇总。避免某种类型示范更多导致其组合数主导结果。unknown×任何类型不作同类或异类监督，但其数量和全部组合数明确保留。类型少于/多于K不是全解数；没有共同已知类型不等于不存在可复用路径。

## 预声明两重筛查及否证

**参考结构筛查仅为必要条件。** 原来的same/different空间距离差部分由类型定义保证，因此通过它本身没有机制证据。raw与H24均须满足：

1. 至少24/48目标对能比较同类及异类；覆盖至少12/16父。
2. 至少12个可评价目标对有两个及以上共同已知类型。
3. 目标对等权的同类/异类穿越距离比中位数≤0.75；至少2/3可评价对的同类距离更小。

每对同时报告同类型**绝对**穿越偏移（m）、横向/高度分量、偏移相对2cm clearance及两排较窄middle opening净宽的比值。middle净宽只是固定有限通道尺度，不能代表外侧/over通道宽度。上述比值不能当安全证书。大绝对偏移、多次穿越和长弧剩余分母必须与比率一起解释。

**预测侧缺口是必须额外成立的条件。** 每条候选只用实际空间类型匹配正参考，不追踪query编号。对于目标正确、单次向前穿越、实际类型在本目标及另一目标均有已知正例的候选，计算它到另一目标同类参考的平均穿越距离，减去两个目标间同类正参考距离分布的p95，得到`excess_m`。如两个其他目标均可评价，先在候选内等权平均，再做父等权统计。无自身同类参考、unknown、多次穿越等全部保留不可评价原因与192槽总分母，不用最近类型强配。

固定last1500必须满足以下额外筛查；best500仅报告阶段对照，不从两者择优：

- 至少4条可评价且目标正确的碰撞候选、8条可评价且目标正确的clear候选；两组各覆盖至少4父。
- 碰撞组父等权excess≥0.02m，且比clear组父等权excess至少高0.02m。
- 碰撞组至少一半候选超出参考正常变化p95包络。

2cm是预先选定的实质漂移筛查尺度，**不修改2cm碰撞规则**，也不声称这些位移能造成/消除碰撞。以上样本量门槛只是小成本否证门槛，不是功效分析或统计显著性。组间不同父构成可能混杂，完整逐父/候选记录必须检查；没有预测侧缺口或样本不足就停止本机制试验，不能借类型分组天然距离差放行、转看DEV补机会或事后放宽。

即使所有screen通过，也只进入实施讨论，不自动证明因果、可学习性或新颖性；若漂移只反映已定位终点的合理变化，仍应停止。现数据少、颜色覆盖和普通收敛都是竞争解释。

## 若有机会，下一步最小配对（尚不执行）

同一已收敛普通基线、同一新数据与已知类型监督，比较同目标关系辅助与同父跨目标部分正对应辅助。两臂参数、初始值、配对数量、请求曝光、原路径loss、端点辅助和选模完全相同；未知/缺参考不作负标签，不强迫K槽一一对应。推理仍每请求单次K4，不输入旧路径、不额外生成上下文，因此没有隐藏8条完整路径状态。

只有跨目标配对本身超过同目标强控制才支持关系监督的作用；若二者相同，归因于普通辅助监督并停止跨目标核心主张。常规对比loss、Hungarian耦合或同类型query编号本身都不是论文贡献。具体训练步数需待普通收敛控制确定后另行冻结，不能在本审计后偷改成有利训练预算。

## 最接近的一手来源（阅读核验，均未在本项目复现）

- [MTR，NeurIPS2022原文](https://arxiv.org/html/2209.13508)、[官方decoder](https://github.com/sshaoshuai/MTR/blob/master/mtr/models/motion_decoder/mtr_decoder.py)：意图点、终点最近意图正分配、轨迹局部map读取及迭代细化；不能把这些组件当区别。默认64→NMS6的预算不能伪装成直接K6。
- [ModeSeq，CVPR2025正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.html)、[原文](https://arxiv.org/html/2411.11911v2)：顺序模式记忆、EMTA、动态输出模式数。读取已有路线/补全不新；本次未确认官方实现来源，不断言无代码。
- [T-MPC，T-RO2024原文](https://arxiv.org/html/2401.06021v2)、[官方guidance实现](https://github.com/tud-amr/guidance_planner/blob/main/src/global_guidance.cpp)：跨目标同伦比较、过滤及历史类别传播已经存在。现官方实现面向已知目标/障碍预测的2D动态环境，不是RGB-D/语言的同信息机器人基线；row-crossing标签也不是完整机械臂同伦类。
- [Goal-conditioned VAE trajectory primitives，SN Computer Science2020](https://link.springer.com/article/10.1007/s42979-020-00324-7)、[全文](https://arxiv.org/pdf/1912.04063v2)：目标条件和离散/连续形状码分离、约束投影；本轮未核验作者实现。[GoalFlow，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_GoalFlow_Goal-Driven_Flow_Matching_for_Multimodal_Trajectories_Generation_in_End-to-End_CVPR_2025_paper.html)、[官方代码](https://github.com/YvanYin/GoalFlow)：场景选择目标再条件flow生成，不能把目标条件/transport名称当创新。
- [HAMSTER，ICLR2025作者页](https://hamster-robot.github.io/)、[官方beta](https://github.com/liyi14/HAMSTER_beta)；[3DWay原文](https://arxiv.org/html/2609.08224v1)、[官方代码](https://github.com/ziqin-h/3DWay)：前者单路径分层迁移，后者多视角对应二维点三角化单路径。3DWay本轮核到2026预印本及作者ECCV2026声明，未独立确认会议录。均不能作为本项目跨目标集合对应的已复现证据。

可能区别仅是从不完备正例学习条件间**部分**关系并证明固定预算价值；有限检索不是新颖性证明。完整集合transport所需的双射/质量守恒当前没有依据，暂不实现。

## 执行和当前验证状态

源码冻结部署后，使用新输出路径，经 `record_job` 包装执行：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0 /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  -m scripts.audit_two_row_cross_goal_correspondence \
  --config configs/two_row_cross_goal_opportunity_v1.json \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_cross_goal_opportunity_v1/analysis
```

必须在root指定的不可变release cwd执行；此处不是已经执行的命令。输出原参考描述、全部48目标对、两池各192候选、全部hash、CPU/耗时和screen结果。原数组与历史结果不改写。稳定源码当前本地纯测试15项通过（0.36s），py_compile通过；测试未导入Torch。服务器正式审计尚未启动。
