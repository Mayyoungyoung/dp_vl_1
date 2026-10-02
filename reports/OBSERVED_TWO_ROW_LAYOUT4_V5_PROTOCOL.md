# v5 四布局泛化可采性：执行前登记

原执行前登记如下；现已从固定 9324efc 唯一执行完毕，完整结果见 [v5 实测报告](OBSERVED_TWO_ROW_LAYOUT4_V5_RESULTS.md)。数据角色全部 `DEV_COLLECTION`，不训练、不跨 split、不建立正式 TEST。四父闭合并不表示全部成功；原登记范围和失败保留。

## 固定范围与非自适应采样

配置为 `configs/observed_two_row_layout4_v5.json`，layout RNG 为 NumPy RandomState(283099)，四父采集 seed 固定 283100–283103。每次所有四组数值先一次性生成并写 registration/source SHA，之后才启动任何 worker；没有按前一父成功率采下一父，没有几何重抽或失败父替换。执行前的实际四布局已保存在 [采样登记](observed_two_row_layout4_v5_registered.json)。新 seed 本身不是物理布局独立性证据。

| 参数 | 分布/固定值（米） |
|---|---|
| 第一排 x | Uniform[0.135,0.145] |
| 第二排 x | Uniform[0.335,0.345] |
| 第一排两柱 y | 独立 Uniform[-0.115,-0.105]、Uniform[0.105,0.115] |
| 第二排两柱 y | 独立 Uniform[-0.155,-0.145]、Uniform[0.145,0.155] |
| 每个目标 x | 独立 Uniform[0.45,0.47] |
| 三个目标 y | 独立 Uniform[-0.175,-0.145]、[-0.015,0.015]、[0.145,0.175] |
| 所有目标 z | 0.84 |
| entry x/y | 独立 Uniform[-0.005,0.005] |
| entry/guide z | 0.865 |
| 两段准备动作 | 同 entry x/y，先 z=1.25，再 z=0.865 |
| 柱截面/高度/base | 0.035×0.035 / **0.14** / 0.755，固定 |
| guide y / row offset | -0.25/0/+0.25 / 0.07，固定 |

每父三个球保持不同可见颜色，三个目标指令共用完全相同的初始 RGB-D/camera/current state。每个目标分别收九个提案，共 27 槽。这里只是围绕已证实可采布局的保守局部几何变化，并非广泛场景泛化数据；后续宽分布/正式规模须另登记。姿态策略与所有有效性阈值保持 v4；不扫描高度/朝向。

## 几何、还原与重复门禁

每个采样配置先检查完整 27 条理想 tip guide 折线及 H24 的碰撞/实际 crossing 定义；仅检查已定布局，失败则记该父 27 槽未尝试，不重新抽样。它不是 IK/整机可达性证明。物理加载时继续核验真实柱位置/尺寸、全部球可见性、RGB-D 到柱面审计、setup/低位碰撞。每条路线之前使用同一父的真实 native snapshot，world、inventory、RGB、depth、相机、pose/open 必须严格零差。

跨父独立性同时报告预登记和实际物理几何指纹。实际主 hash 使用 full world audit 中四根柱的实际 pose/bbox 与实际三个目标中心，分别保存 float 精确 hash 与 1 mm 量化 hash；**重复 gate 不包含 seed、颜色、机器人关节或 setup FK 微小漂移**。layout 加 actual entry 的 hash、初态 RGB hash、完整 world JSON hash另外保存，仅为补充证据，不能用 dummy/velocity/颜色变化掩盖相同布局。若几何重复，保留全部父和失败，明确 block 后续跨 split 使用，等待 group-aware 新划分，不删除重复个体凑数量。

重复检查只在这四个全开发父内进行，本版本不读取任何既有 locked 样本，也不新分配正式角色。所有物理坐标、模式、guide、mask 仍是验证/监督字段，不进入观测生成器的五字段 JSONL 输入。

## 预算、失败和恢复

总预算为 **4 请求父 × 3 指令 × 9 提案 = 108 路线槽**；collector 显式四次 setup 上限 8 次 get_path，路线最多 972 次，显式合计最多 **980 次**。事后源码审计补充：框架 task.reset 的 Task.validate/_feasible 另调用 waypoint.get_path，未插桩，980 不是全系统规划调用上界；内部 IK/OMPL 配置搜索同样未逐项插桩。完整 worker walltime 包含全部开销。每段最多 1000 模拟步，失败即停该槽；完整 raw/H24/事件/partial、规划/模拟/严格恢复成本、exit/PID/log/source hash 均保留。

父 worker 通过 `record_job` 顺序运行，每个使用独立新数据目录，线程上限 CPU1，不并发启动模拟器。协调器为每父保存闭合摘要，再将所有语言/成功参考/失败 attempt join 到批次级 JSONL；setup 失败不虚造图像/参考，仍留请求分母。

`--resume` 要求注册清单和源码 SHA 完全相同。已闭合父只读取；已有 partial 数据或 worker status 的父绝不重新生成提案。若 worker 仍活着则拒绝重叠；已中断且末条未落盘时明确给出已尝试/未尝试上下界（最多一个在途槽），缺失耗时记 unknown，保留后再处理后续预登记父。闭合失败父同样不替换。总状态的 complete 表示四父均已闭合，不表示全部成功。

## 数据判断与界限

每个目标分别报告五类以上有效实际 signature 是否成立，lateral-only 与包含 over 的完整已知类型并列；unknown/重复/H24 拒绝仍保留。原规则不变，既有 v1/v2 negative 不改。四父全部结果读完再决定正式采集范围，不能挑一个成功布局宣称泛化。若出现≥5真实类型的一部分目标，这是对多路线任务可采性的进一步证据，仍不是新生成机制对强基线的优势。

新增协调器与显式目标门禁测试与已有 48 项合并。v4 明确只接受 `[1]`，v5 明确只接受 `[0,1,2]`，即使把预算同时改成匹配长度也拒绝错误版本目标；缺失目标字段同样拒绝。覆盖确定性预登记/独立几何/全108预算、指纹不靠 FK 漂移、setup失败与中断上下界、拒绝重复 ledger、拒绝错源码恢复、已失败闭合父不重跑，以及 starting/running 活 worker 不被错标中断或重启。固定四布局的 108 条理想 raw/H24 几何路径全通过；实际机器人成功率尚未知。
