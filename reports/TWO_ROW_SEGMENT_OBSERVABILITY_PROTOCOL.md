# TRAIN 内线段可辨识性诊断：预登记 v1

2026-10-03。这是普通表示前提的低成本证伪实验，不是新方法、路线有效率提升或未见场景泛化实验。只准备独立代码，尚未执行服务器/真实模型。不得据本地契约测试宣称假设成立。

## 保存池与调用边界

使用封存 `observed_two_row_composite108_v1/fixed_last_train/predictions.npz`，SHA `5b490ddf8fd4b0a0b1bff8abe961a4b9513e3a91f916c5f7cc14b8633ddc29d7`，匹配普通 constant12000 的 last 权重 `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`。固定选择原首16个TRAIN父283200–283215，每父三个目标共48输入，原K4共192候选；同时使用全部285已采正参考，按原事件分段重采样到H24。未知类型、错误目标、失败和长弧不筛除。正参考少不等于解数少，不新增不存在监督。

已核本地真实归档的训练receipt也索引了 `peak_seed0/train/predictions.npz`（best TRAIN）；但本次依root最终收窄指令只使用fixed-last池，不调用或混入best。初版候选草案提及best/last的范围在此明确替换为last-only。最新普通模型不是旧prefix28模型。

原保存池没有128维几何context或逐线段碰撞标签。Qwen最终mean/last缓存、RGB-D、相机及当前state已有封存来源，因此无需新的Qwen或路线头forward。只构造原 `ObservedGeometryEncoder`，严格载入last checkpoint的 `geometry.*`，eval/no_grad/CPU，每个输入一次：**48 geometry-only forward，0 Qwen、0路线头、0新候选**。CPU float32使用原权重与同源函数；没有已存GPU context可逐值比较，不声称跨设备逐位等价。

所有几何/目标真值只由原评价器构造标签、路线分组和诊断指标；它们不进入几何encoder、局部描述符或probe特征。路线坐标是被判别对象，允许作为探针输入；这是路线碰撞可辨识性，不是生成器的无未来条件推理。图片语言条件来自已有真实冻结Qwen缓存，不加载Qwen。

## 固定拟合与留出

- 拟合：父283200–283211，12父36输入。留出：父283212–283215，4父12输入。不能根据碰撞数量换父。
- 这些父全都曾用于普通生成器训练。这里只是**探针按父留出**，不能称生成器OOD或新测试集。留出不参与标准化、正则选择、阈值或模型选择。
- 原完整285 TRAIN池可读取容器及身份，但只取固定48行；新DEV/旧DEV原图、路径和label均不打开。导出JSONL按id首字段筛选，未选行payload不解码；全文件SHA可机械校验。原collector全局layout gate只读机械metadata。
- 原2cm扩张AABB完整线段检查产生每条H24的23个碰撞标签；并逐路线核 `any(segment_collision) == not original_tip_segments_clear`。没有修改原碰撞/语义/事件/分类阈值，也不把box-only当全身执行有效。
- 请求477条已有路径、10971条线段。有限但错误目标/unknown/无效路径照常拟合和评价。非有限线段原样保留在records和请求分母，不伪造坐标或标签；单列feature unavailable。没有生成替代路线。

## 三个同维度探针

固定闭式ridge线性回归碰撞二元标签，lambda=.01，不调参、不迭代选checkpoint。各臂188输入＋bias，189个系数；预测截到[0,1]仅便于Brier与固定0.5分类，**不是校准成功概率**。共3次CPU监督拟合/线性求解，明确是诊断训练，不称“零训练”；没有Torch optimizer或基础参数更新。

共同输入为实际context128＋线段坐标20：当前末端相对两端点6、端点平方6、delta3、中点3、长度1、零长度1。没有ID、目标编号、类型、真几何、成功标签或来源类别作为特征。

| 臂 | 额外40维 |
|---|---|
| global | 全零padding，保留同一参数数组尺寸；有效特征秩可能较低 |
| aligned | 五个有序线段位置的观测局部描述，各8维 |
| shuffled | 与aligned完全相同的局部描述分布，但在同一输入内按仅依ID与行数的确定循环错位，至少错1行、没有固定点；不读label来匹配或分层 |

每个局部位置使用：最近可见点距离1、其相对xyz3、最近32点中落在5cm内的比例1、相机深度差1、可见射线自由flag1、ray-known flag1。空间和深度差截±25cm，距离最多25cm。ray-free仅表示该采样点在当前有效视锥内且比观测表面靠近至少2cm；未知/出视野保持known=0，不能当free。负fx/fy保留。没有分割真值/开口/箱体/guide输入。

这五点离散描述不是连续净空证明、不是规划图、不是路线修复，也不复用原保守图对正例的否定。相邻线段和同父样本相关，不能按10971线段当独立样本。

**容量边界明确：** 零padding不会给global提供额外信息，因此不能仅凭aligned好于global归因空间对应；必须同时好于具有相同40维真实描述的shuffled。这个控制仍只检验所选固定特征对线性探针的可用性，不排除更强非线性global探针可追平，不证明128维表示在信息论上丢失拓扑。

权重按父→输入→来源（候选/参考）→路线→线段逐层等权，均不使用标签。标准化均值/方差仅由拟合父加权计算；三个臂按同一拟合行与权重。正参考重复和长度不能以行数压过模型失败；全参考仍保留。

## 预定判据

主集合是全部留出父的generated-last线段，包含错误目标与未知类型；目标正确子集、参考、错误目标、未知类型及无效路线仅分解报告，不替代主判据。报告AUROC（ties平均秩）、Brier、0.5 balanced accuracy、逐父结果和父平均。

进入后续设计讨论的必要条件：至少2个留出父都有两类标签，合计至少20碰撞线段及20clear线段；aligned相对global和shuffled两者，父平均AUROC均至少提高.05，父平均Brier均至少下降.01。若支持不足则 `stop_underpowered`，不扩父、不补负例、不换seed；充分但未达到则 `stop_no_alignment_advantage`。该门槛是现在预定的决策屏障，不是显著性或新生成有效性的证明。若通过，也只支持下一普通空间解码配对的必要前提，尚未授权新训练模块。

## 来源、执行与失败

新文件仅脚本/config/tests/本协议；原模型/collector/评价源码不变。配置锁导出、缓存config/samples/receipt、last checkpoint、固定池、原诊断receipt全部SHA。每个被选原图/NPZ/routeconfig/参考按export源SHA检查；当前几何与原评价函数必须匹配训练receipt来源。预先与最终机械gate均通过，期间输入再SHA核验。

执行须root冻结新release后单核CPU、显式隐藏GPU、数值库线程1，fresh output。没有自动重试或恢复；失败保存实际已发geometry调用、原样错误、耗时及部分证据，不重放已发预算。3次CPU线性拟合另计。geometry ledger先issued再complete；保存逐段features/labels/index、全部477路线结果、三臂拟合参数与scores、完整指标、screen、status和文件SHA。特征包含监督标签，因此仅用于诊断，不能成为未来生成条件缓存。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c <root分配单核> .venv-qwen/bin/python -m scripts.audit_two_row_segment_observability \
  --config configs/two_row_segment_observability_v1.json --output <fresh-run>/analysis
```

运行通过现有record_job记录PID/source/日志/退出码，root负责实际launch。此文没有提前声称実测结果。

最近直接近邻是[MTR](https://arxiv.org/abs/2209.13508)按预测路线读取邻近map polyline；[官方decoder](https://github.com/sshaoshuai/MTR/blob/master/mtr/models/motion_decoder/mtr_decoder.py)已实现动态map收集。局部对应/attention不是本项目新贡献。该诊断只判断是否值得建立更强的普通观测空间控制，不重启旧半程refiner或原max记忆。

本地纯契约测试初轮19 passed、1 skipped（Windows创建symlink权限不具备；Linux冻结测试必须覆盖），2.51s。没有实际Torch几何重放、真实数据诊断或服务器调用；这些待root单独执行。
