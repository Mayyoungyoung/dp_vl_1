# 数据卡：受控几何与观测任务分层

## 当前实际进展 — 2026-10-03 00:26 UTC

本段覆盖下方旧状态。首32新增TRAIN完整质量报告及全部96图/32front已核验、归档：reports/observed_two_row_extension288_quality_v1/TRAIN32_RESULTS.md。555正例/309失败、247unknown原样保留，未发现系统采集错误；仅窄±5mm同分布，不称复杂场景或OOD。主报告包括每槽失败、19色分布、长弧和全部hash。

composite source71cf0c1实际70测试0skip通过，1663/1663TRAIN正参考通过H24/端点容量；真实新增96条Qwen编码完成，旧225份NPZ（含36DEV）容器字节全部保持，三阶段cost来源已归档。见 reports/observed_two_row_composite108_preparation_v1/QUALITY_CACHE_RESULTS.md。cache外层21.775545秒/.0060487625GPUh，编码body11.501351秒含加载，嵌套不相加。没有过时LoRA条件缓存或新DEV数据进入这轮冻结Qwen控制。

root独立启动同总12000步普通模型训练：source71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c，PID652623/child652624，00:17:54.174724UTC开始，SSH64194，CPU1/GPU1/35%；00:24进度7750/12000、31/48原DEV选择。当前仍运行，不用中间结果作新主结论。输出 /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1；完成后必须单独fixed-last-train诊断，再读原189TRAIN公共子集/新增96TRAIN/固定36DEV配对。初始化/总1536000路径状态同原constant，数据人口及每输入曝光不同，不能称方法贡献。

另按已读质量独立启动train64 resume，仍为原5c8f8e4 collector，session20261003T001613Z_train64_651720，SSH57754，CPU2/3，00:16:13UTC开始；仅追加注册indices32..63，不重采已闭合前32。该阶段全64闭合前不读取新增raw结果，不自动启动train128。新DEV继续封存。服务器实际job路径/PID/命令以JOBS与registry为准，禁止根据旧running段重复启动。

MAIN271保持。JOBS快照 2026-10-03T00:25:15.049030+00:00 共549条，registry同步最新转移。尚无核心方法优势；当前活跃会话继续实际训练后的分析和研究决定，没有会话结束后自动思考服务。

## 当前执行 — 2026-10-03 00:13 UTC

本段覆盖下方旧状态。source1fccf48的88项服务器测试通过，两臂各36次真实Qwen/K4已完成exit0，72份特征字节及288候选检查决定均与原best一致。完整请求中位constant/no-direct为74.5681/74.5213ms，P95为86.1414/91.2371ms；未显示可靠加速。原质量59/144对52/144保留。MAIN271，原269字段未改，新两行只记在线成本；见 reports/observed_two_row_12000_online_v1/ONLINE_RESULTS.md。

extension288首批train32已于00:02:21UTC结束，两shard exit0，原SSH89814结束。source1a3的完整TRAIN32分析于00:07:08UTC exit0：864已发槽、555accepted、309failed、247有效unknown、0已知重复，864严格状态恢复全部通过；无缺初始观测/blocked/跨父重复。96条件中20个已知类型数超过4，96全图正在独立视觉核验和归档。没有将提案失败隐藏或将unknown视无效。

source1fcc的composite108 readiness/export已独立完成，export于00:08:26UTC exit0；321输入=285TRAIN+36原DEV，1868正参考=1663TRAIN+205DEV。原225行字节保留，旧3缺失输入不替换。新DEV封存。下一步是新冻结pipeline的真实Torch测试、完整TRAIN表示/端点容量检查、仅新96条Qwen编码，然后同总12000步普通模型训练；所有阶段分别启动，当前尚未运行后四阶段，不把export成功称训练完成。pipeline已独立复核，本地22pass，3项Torch待服务器验证。

JOBS快照 2026-10-03T00:11:37.242427+00:00 共534条；registry实际同步。当前没有GPU训练或采集作业运行。待读完32质量后才单独启动train64 resume，使用原5c8f8e4固定collector，不重采前32。核心新方法优势仍未成立，数据扩大是普通基线控制。恢复需先核实际PID/status/immutable source；不重启上述已完成作业。

## 2026-10-02 23:24 UTC 实施更新

TRAIN12条件探针已完成36 K4前向，未建立泛化/新方法优势；详见[原始诊断结果](OBSERVED_TWO_ROW_ROUTE_CONDITIONING_RESULTS.md)。全部69正参考被grid拒绝不能解读为路线类型不存在，原判定未改。下一普通去direct分支对照尚在实现；Qwen冻结、geometry与剩余decoder仍训练，减参是必须披露的混杂。

独立extension288的158服务器测试和prepare已完成，首批32TRAIN/864槽在CPU2/3采集，未提前报告质量。256TRAIN+32新DEV一次登记，不替换失败；新DEV后采并封存，旧12DEV保持reused。全部32父闭合后才做全槽质量核验与后续导出。详见[登记协议](OBSERVED_TWO_ROW_EXTENSION288_PROTOCOL.md)及[实际验证](observed_two_row_extension288_validation_v1/VALIDATION_RESULTS.md)。本次扩数据不构成核心机制，论文A–H仍未满足。

## 2026-10-02 21:20 UTC：formal116采集机械完成

原冻结collector2626487会话20261002T183023Z_485548在21:10:27 UTC结束，两shard退出0；116个登记父均有closure文件，角色计数64TRAIN/12DEV_MODEL/12DEV_SCORE/12CALIBRATION/16TEST_LOCKED。仅核验保留区标记存在、登记元数据与不透明SHA，没有打开保留区closure内容、观测、轨迹或指标。完整采集标记不等于所有路线成功。

已获允许的TRAIN/DEV导出仍为63实际TRAIN父189输入1108正参考、12DEV父36输入205正参考；缺失父283220的3不可用输入与27未尝试槽保留，不替换。新完成的SCORE/CALIBRATION/TEST_LOCKED数据尚未用于模型选择或评估。原始作业、登记、源与机械审计见[完成档案](observed_two_row_formal116_completion_v1/COMPLETION_DATA_CARD.md)。无需重启已完成采集。

2026-10-02 19:30 UTC：两排观测普通基线已完成，best/last DEV均TipValid22.92%，有效已知重复0；固定末步TRAIN87.5%，表明当前主要是泛化差距。真实Qwen在线36请求中位73.15ms；传统观测A*为52.78%有效但平均1.31个已知重复槽、中位779.90ms。主表248行保留全部失败、相同checkpoint成本去重。下一步已确定前32/64TRAIN等曝光普通基线，不把扩数据或现有控制称为核心方法优势。

2026-10-02 17:35 UTC：新增四个DEV_COLLECTION物理布局、同图三目标；108请求槽中81尝试/61有效/20路线失败/27初始化未尝试。已接受参考中49已分类、12unknown；已知类型数1–8，6/12请求目标条件至少5种。原失败、全部尝试、严格恢复和数据hash见[完整v5报告](OBSERVED_TWO_ROW_LAYOUT4_V5_RESULTS.md)。这些是约±5mm局部几何变化，尚未升级为正式TRAIN/独立测试；没有把示范数当真实总解数。

最新第三点为prefix108：96请求TRAIN父（95正参考）、12原DEV父，321/324正参考，432语言输入；TRAIN保留3条无参考输入，只对381条可评价输入计算参考指标。所有48DEV记录、60原文件及48Qwen缓存hash与旧快照一致。该快照数据已用于真实1500步普通头训练，详见[数据清单](OBSERVATION_PREFIX108_DATA.md)与[实测结果](OBSERVATION_MULTITASK_PREFIX108_RESULTS.md)。其余score/calibration/locked角色没有用于这次训练和选模。两排新24-query只是端点配置诊断，0路线/0新参考；不要计入轨迹规模。

最新实测补充（2026-10-02）：六任务prefix24/prefix60快照均已封存，分别12/48TRAIN父与完全相同的12DEV父，72/180条正参考，96/240个语言输入。父划分遵守原注册和跨批次机械布局门禁，全部选中请求父保留；无语义、类型或模型执行标签时相应评价为null。每父语言改写不跨角色。collector用parent_index对应variation，因此扩TRAIN同时增加颜色/variation，固定DEV不是IID样本；不得把普通头误差变化归因于纯数据量。详见[学习曲线](OBSERVATION_MULTITASK_LEARNING_CURVE.md)。两排低柱仍是DEV_COLLECTION单父可采性试验：v1/v2各自27槽，18/4接受，跨版本关节初态不同，不能当作54个独立父或配对机制证据。原始采集失败、未知类型和全部预算见[第二轮报告](OBSERVED_TWO_ROW_PILOT_V2_RESULTS.md)。

更新：2026-10-02。本文记录实际已有数据与当前建设边界；参考集条数不作为连续规划的真实解总数。观测采集状态以 [OBSERVATION_READINESS.md](OBSERVATION_READINESS.md) 及对应原始日志为准。

## 历史受控数据（回归检查）

原数据 SHA256：`34f4e6944f66c7e2716239830b22f0ad1885bf80a9f9dc6be606e7883d04374d`；生成 seed 17；1152 个场景，TRAIN/VAL/TEST/OOD 分别 768/128/128/128。每场景 12 条 24×3 路点参考：固定 negative_y、positive_y、over_top 三类，各四个变体，总参考数 13,824。

模型条件是冻结 `openai/clip-vit-base-patch32`（revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`）的图文特征，加真实几何和已知起终点。数据、模型及三种子结果保留于历史代码 `d0d97eb22ff1190f4f89d4d2fd7100db116c8a57`、服务器 `runs/main` 和历史报告。不能由此证明 Qwen、开放词汇定位或机器人执行。历史 TEST/OOD 曾被用于本次后续研究判断，现在标记为历史/开发证据；这不改写原始实验当时没有用它们拟合评分器或选 checkpoint 的事实。

## 层级 A：multigate_v1

生成器：`routeset/multigate.py`；seed 2718；全归档 SHA256 `7259cd30218d44a668ce8ec8d4bec993b3efcc66f6b492cd304a1a9c94e7658a`。元数据在 `reports/audit_v2/multigate_v1.manifest.json`。

| 划分 | 独立父布局数 | 用途 |
|---|---:|---|
| TRAIN | 768 | 模型及后续评分训练来源 |
| DEV_MODEL | 128 | 模型、损失及方法决策 |
| DEV_SCORE | 64 | 评分器/筛选设置选择，尚未用于本轮拟合 |
| CALIBRATION | 64 | 最终评分映射拟合，尚未用于本轮拟合 |
| TEST_LOCKED | 64 | 待方案冻结后的独立确认 |
| OOD_LOCKED | 64 | 待方案冻结后的预设分布变化确认 |

共 1152 个父布局，当前每父布局只有一个目标版本。划分单位是父布局；将来同布局的目标变化、相机变化、语言改写必须继承该 split。各 split 使用独立 RNG 子流，使扩充 TRAIN 不改变既有锁定场景。

场景是两面平行、贯穿 z 方向的墙；每墙 1–4 个开口。起点在 x=-0.92，目标在 x=0.92，y 随机，z=0.5。开口数量、位置、宽度和墙位置变化。所有跨墙开口组合在该构造中可行，每个组合构造一条中心线正参考；R 可取 1/2/3/4/6/8/9/12/16。路径 24×3；一个场景最多 16 个参考。模式由经过两面墙的开口索引对定义，而非欧氏抖动。

参考路径单调沿 x 前进，保留墙前/墙中/墙后肩点，z 恒为 0.5。网络输出并未被硬编码为 x 单调。输入为真实 start3、goal3 与两行墙/开口参数，总 34 维；模式编号和参考路径不是推理条件。指令目前固定为一条英文到达目标描述，且本轮模型不读取该语言。因此这是三维坐标表示的平面通道选择任务，不是复杂三维绕行或语义规划。

### 几何与类型检查

检查所有相邻路点形成的**完整闭线段**与 clearance=0.02 扩张 AABB 的相交，不只检查离散点；扩张按各坐标轴实施。另检查有限数值、工作空间 `[-1,-1,0]` 到 `[1,1,1]`、首尾容差 0.06、总长度与直线距离比不超过 3.5。墙贯穿高度避免未定义的上方绕行。

类型检查记录每面墙的开口穿越；若回穿同一墙却使用不同开口，路线可独立通过有效性检查，但类型记为未分类，报告 `unclassified_valid_rate`。有效性和类型识别并非同一判断。ReferenceCoverage 仅以已知参考类型作分母。这里可构造全部开口组合，并不意味着任意机器人数据都知道全部解法。

### 物理隔离与已使用范围

服务器路径：`/home/wzy/dpvlm/route_set_v1/data/multigate_v1_partitions/`。

| 文件 | 包含划分 | SHA256 |
|---|---|---|
| `development.npz` | TRAIN、DEV_MODEL，896 父场景 | `f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057` |
| `scoring.npz` | DEV_SCORE、CALIBRATION，128 父场景 | `558dd162e8fceefc50ac15aecf5621d5dab77db64ba29aca73ba48c2e2009c9b` |
| `locked.npz` | TEST_LOCKED、OOD_LOCKED，128 父场景 | `fb0634087b426aa14706b71e36d3710f411ff4a05bacf17e6931ae753c1e6cc5` |

round1 加载含所有 split 的原归档，但训练索引和指标仅使用 TRAIN/DEV_MODEL；不能说 round1 进程从未读取锁定数组。round2 改用 development 物理分区。两轮未使用锁定预测或指标作方法选择。OOD 的预设变化为墙位置更外移、开口更窄；其最终表现尚不在本文。

### 局限

该数据超过了固定三路，但仍是固定两墙、全开口组合可行、无接触事件的强模板。缺少不规则多障碍、不可兼容开口组合、不同接近方式、变化约束和真实观测误差。K=4 saturation 在当前 DEV 已达类型预算上限，应将其用于机制诊断和回归检查，不能仅靠扩充同模板父场景支撑机器人方法贡献。

## 层级 B：RLBench-derived 观测任务（已完成首批采集与真实 Qwen 特征核验）

Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203` 的官方文件哈希已通过，2,127,532,032 参数 BF16 模型已真实加载。正式 pilot 的 96 条 RGB+语言输入均完成在线前向并缓存 2048 维 mean 与 2048 维 last hidden，缓存时全部基础参数冻结；有限值、图像/输入哈希已检查。峰值 CUDA allocated 约 3.99 GiB。模型前向成功仍不同于路线训练、语义泛化或 LoRA 效果。

`observation_derived_reach32_20261002` 有 32 独立父场景、32 唯一初图，24 TRAIN/8 DEV_MODEL；每父同图 3 个颜色目标，共 96 个观测—指令对。288 次采集尝试产生 275 条成功轨迹和 13 条规划失败；全部 288 次完整状态/RGB 恢复差为 0。275 条保存轨迹均有限，末端通过 3 cm 终点复核，平均误差 0.677 mm、最大 10.844 mm。这些是**采集轨迹**验收，不是学习模型预测性能。

成功参考分布为 88 个指令各 3 条、5 个各 2 条、1 个 1 条、2 个 0 条。无成功参考不等于无解。原v1 loader排除这两条，已作为实际评价错误修正：v2保留全部96观察，训练正例采样仍为71 TRAIN指令，DEV语义按全部24条、参考误差按23条。所有旧checkpoint都重评v2并保留v1报告。275条参考（208 TRAIN/67 DEV_MODEL）不变。在线LoRA不读取冻结缓存特征。

审计与全文件哈希清单位于 `reports/observation_reach32_audit/`，数据哈希清单 SHA256 为 `b6cb09f782d8bd8d3755ec33671cf6399bf7d5c2a57c7a47b52e6b3d9c93e81b`。采集 Python 完成且产物通过审计，但外层 shell 因运行中修改脚本导致最终退出 1，状态保留为 `collection_complete_wrapper_failed`；不把这一控制层失败隐去或记成 exit 0。具体修复记录见观测链路报告。

| 审计任务 | 可变化的自由运动部分 | 尚未满足的研究条件 |
|---|---|---|
| reach_target | 到颜色目标前的接近运动 | 原场景无明确多通道；轨迹弯曲不自动是新类型 |
| pick_and_lift | 抓取前接近；保持原抓取/抬升段 | 抓取与事件正确性需真实验收 |
| push_button | 按压前接近 | 单按钮/指令改写不足以验证同图不同目标 |
| take_lid_off_saucepan | 抓盖前接近 | 单目标；同图目标/约束配对尚待构建 |

试采称为 **RLBench-derived free-approach pilot**，保留原成功判断并插入自由运动方案，不作为原 benchmark 原生多解标注。四原任务各 1 父场景、每父 3 次尝试，12/12 通过各原任务成功条件，12/12 状态与 RGB 恢复差为 0；这只是四任务可采集性验证，不是学习方法在四任务上的结果。当前仅逐模拟步检查新增自由段碰撞，未完成连续碰撞认证；未定义任务路线类型前 `unique_valid_route_types=null`。1 cm 近重复数为 0 也不等于发现了不同拓扑类型。

输入契约为 RGB/RGB-D、语言、相机参数、当前末端/夹爪状态。真实目标坐标、完整仿真障碍几何、未来路径、task_low_dim_state、验收标签、采集引导点和模式编号均与生成输入分离。当前 Qwen 缓存 JSONL 严格限定 `id,parent_id,split,image,instruction`；深度/相机/状态另存为观测条件。终点必须由这些观测条件学习推断，不能沿用层级 A 的固定真实端点解码。

同初态采集显式恢复模拟器/任务状态、物体姿态、关节、速度、颜色与机器人状态，统一物理稳定化后核验 RGB；单独重设 RNG 和连续 `get_demos` 均不够。早期恢复失败与零恢复差但终点失败的尝试已保留，未放宽容差；修复后才采集上述正式 pilot。记录全部尝试、恢复失败、采集失败、验收失败及近重复，模式/引导标签只留审计。

实际Qwen+观测RGB-D头已训练并有三种子定位修正结果，仍缺多任务规模及路线集合核心优势。训练结果另列checkpoint和预测指标，不混入采集器验收。

## 第二批正在采集与学习曲线快照

自然布局批次`observation_reach_fast256_20261002`预定256父（192TRAIN64DEV_MODEL）、每父同图3目标×3自由运动提案，总计划2304次。种子262000–262255，与旧32父和单父pilot261900分开。正式采集必须保留每一步arm/gripper碰撞检查、状态/图像严格恢复、端点和失败原因，具体已完成数量见作业快照，不能将计划总数写成成功总数。无障碍自然布局仍不因弯曲或欧氏差别宣称不同离散路线类型。

`observation_learning_curve_new32_v1`是完成前32个新TRAIN父后生成的不可变快照，加入明确反复使用的旧8DEV父；120观察、96TRAIN监督、24DEV语义/23参考。输入manifest SHA256=`b10909644f66dd86e3275b181e86318b2c459bbc2e77869f8b8d4765beda1f11`。仅纳入已写完整3观察/3监督/9次尝试的父，源文件逐个hash二次核验，失败和零参考指令保留。它不改变正式256划分，也不是独立确认集。同内容别名快照未另训练，不增加样本规模。

随机障碍批次`obstacle_reach_safe_randomized_v3_128_20261002`预定128父（96TRAIN32DEV）、每指令四个±x/±y提案；种子272000+。实际物理障碍的xy/z/尺寸及三个目标的安全区排列和连续位置变化，类型按实际穿过障碍中平面的绕行侧定义，混合/回穿不一致则类型null，不能强行标类型。它是RLBench-derived扩展，不是原reach_target基准。

四父开发pilot seeds271100–271103：48次尝试、20成功、17可分类且无分类重复，21规划失败+7真实碰撞；48/48恢复严格一致。固定双restore中间必须显式渲染并丢弃暖帧，避免CoppeliaSim首次渲染旧mesh。所有箱体mask深度点对实际AABB表面的双向距离必须≤1cm，四父实测最大2.356mm。mask/几何仅用于采集审核和评价，生成器仍只用RGB-D/语言/相机/当前状态。

`observed_obstacle_tip_eval_v1`独立检查完整末端线段对物理箱体的2cm扩张AABB、原3cm语义终点、5mm当前起点、reach事件序列；报告TipValid和已分类类型覆盖。它不保证全身/IK/执行成功，也不把未知类型有效路线当无效。失败/NaN全部占K，缺分数Selected为null。类型参考不完备，无已知类型时覆盖率为null。正式六分区、多任务与独立观测确认集仍需后续建立。

进一步闭合快照：new64自然布局包含前64个新TRAIN父、192监督指令和旧8父DEV；输入SHA43205853a21731b3be3f133210031f50016d1ca9208d5c50e26b23a3c9252c31。obstacle_new32包含前32个请求TRAIN父272000–272031、96指令181条正参考，保留2条零参考；70条路径未分类仍是正例，35/96指令有至少两种已知类型。已知类型频数-x48/+y57/+x6，没有采得-y不能推断-y不存在。同4父开发pilot的12指令20参考反复用于开发。输入SHA878ae1935499a5c5e6e64c4922f2d7cd6c1ffb80a0b0329af6e3b37efd914dfe。快照不改变原批次父划分、不以补采成功父替换失败父、不读取未闭合父。

2026-10-02补充：自然256父批次已实际完成并exit0，2304次尝试中2281成功、23失败、1近重复，全部2304次严格恢复通过，0父初始化失败。该总计是获准的机械采集审计，不是生成器结果；近重复之外的路径也不因此算不同类型。实际耗时6324.26秒。

原采集器中的192TRAIN/64DEV字段保持原样；10:10:31UTC预注册的 `configs/observation_partition_reservation_v1.json` 定义后续派生数据的精确父角色，取代全64父都用于开发的旧计划。`export_observation_roles.py` 先按parent_id过滤，再解析所选记录、打开所选父文件，拒绝锁定/评分/校准角色与跨父路径。实际导出192TRAIN+16freshDEV（624输入，576/48，均有参考），输入SHA `aa17ecef147f73ada2e902ac30f7de39be0acc47a25ced5959607619bf945495`。源文件未变；Linux测试真实拒绝符号链接跨父。新DEV会用于开发泛化检查，不能称最终锁定测试。评分、校准和锁定父内容未被打开用于选方法。


预注册障碍角色已从原128父collector独立导出：96TRAIN＋8DEV_MODEL、312个观测指令，285/288 TRAIN与23/24 DEV有至少一个参考，4个零参考输入保留。只读过滤在JSON载荷解析之前按父进行；score/calibration/locked未进入导出。input SHA d24e7ec52948a6b521498a72885a5c880c4a50b0e295f65386e692d67feee072。初次导出因旧attempt记录不含split而exit1；修复只从原观测/监督取source split，新目录完整exit0。

312条真实Qwen冻结特征完成，22.952秒含3.688秒加载，峰值4279495680字节；全部特征和图像SHA逐条核验。该耗时是整批提取，不能冒充单请求延迟。
