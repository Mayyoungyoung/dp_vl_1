# 相同初始路径头的冻结 Qwen / LoRA 续训：完整配对结果

本轮没有获得稳定的原始 best 优势。LoRA best 的 TipValid@4 从 62/144 降至 58/144，AnyTipValid 和已分类有效路线数持平；固定 last3000 则从 53/144 提高至 62/144，TRAIN 拟合也提高。保留两种普通强基线和全部代价，不把单种子、重复开发集上的 last 改善写成方法贡献或稳定泛化优势。

下一研究决定是完成预先固定的普通独立/集合观测扩散对照，检验生成目标和训练形式；本轮不追加 LoRA 参数搜索或以此增加核心模块。0f 扩散首次启动因历史 receipt 字段位于 budget 内而失败，尚未构造模型/发出调用；ROOT 正独立修复并冻结新源。本报告不把该启动视为扩散训练结果。

## 固定条件与真实更新

两臂从相同 composite108 的 last12000 权重续训，使用同一 285 个 TRAIN 输入、1663 条已知正参考、RGB/语言/观测 RGB-D/当前状态。288 个请求 TRAIN 输入中的 3 个不可用输入保留；没有加入新 DEV、SCORE、CALIBRATION 或 LOCKED 数据。共同初始权重 SHA256：`ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`。

每臂新建 AdamW，固定 3000×32（B1 梯度累积）、K4/H24、96000 个相同抽样、384000 个新增训练路径状态。头学习率 3e-4；LoRA 仅最后两层 q/v，r8/alpha16、学习率 1e-5。两臂仍训练全部原头和观测几何参数；LoRA 额外增加 114688 个可训练参数，是容量差异。

每臂 60/60 个 head tensor 均实际改变，625 个冻结基础模型 tensor 的字节保持；LoRA 的 8/8 adapter tensor 实际改变，最后一次梯度均存在、非零。冻结臂训练参数 1231965，LoRA 臂 1346653。共享抽样 fingerprint：`b3e0826f1f09d197d1599b6df3fc53851925d987b0d8b7258603281e55acc863`。两臂实际 issued 账本均为 195864 条，摘要 SHA 相同：`11f32842970019a98566be05931b8c5d30de9f4b83858aaa7d2e6f2127eb7d9e`。

固定每 250 步保存全部 36 个旧 DEV 指令，12 次原始选择；选择分数是 UniqueClassifiedTipValid + 0.05×TipValid，仅严格大于时更新，平分保留较早步数。冻结臂 best1000、LoRA best2000；两臂均保留固定 last3000。未按本报告重选 checkpoint。

## 全部 36 DEV 指令、12 父场景、每视图 144 候选

| 臂/原阶段 | 步数 | TipValid | AnyTipValid | 已分类 Unique/条件 | known-reference coverage | 语义正确 | TipClear | 有效 unknown 总数 | 已分类有效重复总数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen best | 1000 | 62/144 (43.06%) | 30/36 | 28/36 = 0.7778 | 5.5556% | 124/144 | 73/144 | 34 | 0 |
| lora best | 2000 | 58/144 (40.28%) | 30/36 | 28/36 = 0.7778 | 6.0185% | 124/144 | 65/144 | 29 | 1 |
| frozen last | 3000 | 53/144 (36.81%) | 28/36 | 20/36 = 0.5556 | 2.3148% | 119/144 | 63/144 | 32 | 1 |
| lora last | 3000 | 62/144 (43.06%) | 29/36 | 25/36 = 0.6944 | 5.0926% | 127/144 | 70/144 | 35 | 2 |

覆盖率仅针对已采到的已知正参考类型，36 个条件均可评价、共 205 条已知正参考；不代表所有可能路径的覆盖。unknown 有效路径仍算 TipValid，不当负例。AnyTipValid 是集合 oracle 上限，没有逐路线评分选择或机器人执行认证。

| 臂/阶段 | candidate ADE (cm) | reference ADE (cm) | candidate endpoint error (cm) | event state/sequence |
|---|---:|---:|---:|---:|
| frozen best | 11.3702 | 14.9469 | 4.4884 | 100%/100% |
| lora best | 11.2335 | 14.8183 | 4.4814 | 100%/100% |
| frozen last | 11.3127 | 14.8598 | 4.5637 | 100%/100% |
| lora last | 11.1854 | 14.6983 | 3.7026 | 100%/100% |

两种 ADE 和末端误差的参考分母均为 36；末端误差针对匹配参考，不能等同真实目标中心误差。该 reach 数据的 open 序列正确不表示接触/执行成功。

## 失败分解与逐父差异

| 臂/阶段 | 语义对且净空 | 语义对但碰撞/无效 | 目标错但净空 | 目标错且碰撞/无效 |
|---|---:|---:|---:|---:|
| frozen best | 62 | 62 | 11 | 9 |
| lora best | 58 | 66 | 7 | 13 |
| frozen last | 53 | 66 | 10 | 15 |
| lora last | 62 | 65 | 8 | 9 |

全部四池的坐标/事件均有限，起点和事件序列检查均通过，所以这里的 collision_or_invalid 实际由 tip 线段碰撞造成。best 中两臂均有 124 个目标正确候选，但其中 62/66 个仍碰撞；已知有效重复仅 0/1 个。剩余主要缺口是有效几何生成，不能仅凭本轮结果把重复压制作为充分解释。该分解不证明 LoRA 的因果作用。

LoRA−frozen 的 TipValid：best 为 −2.7778pp，按父胜/平/负 4/3/5、按条件 5/25/6；last 为 +6.25pp，按父 8/1/3、按条件 11/19/6。原 best 的 known-reference 平均覆盖略增，但按父仅 1 胜/9 平/2 负，不能概括为全面覆盖改善。

下表按父给出 LoRA−frozen 的实际计数差；每父 3 条指令、12 条候选，无筛选。ΔAny 是有至少一条有效候选的指令数差；ΔUnique 是三个指令已分类不同有效类型数之和差。

| 父 ID | best Δvalid/12 | best ΔAny/3 | best ΔUnique | last Δvalid/12 | last ΔAny/3 | last ΔUnique |
|---|---:|---:|---:|---:|---:|---:|
| two_row_reach_283264 | +1 | +0 | -1 | +1 | +0 | +0 |
| two_row_reach_283265 | -3 | +0 | -1 | -4 | -2 | -1 |
| two_row_reach_283266 | -1 | -1 | +0 | -1 | +0 | +0 |
| two_row_reach_283267 | -1 | +0 | -1 | +0 | +0 | +0 |
| two_row_reach_283268 | +1 | +1 | +1 | +1 | +1 | +2 |
| two_row_reach_283269 | -2 | +0 | -2 | +2 | +0 | +1 |
| two_row_reach_283270 | +1 | +0 | +2 | +1 | +0 | +1 |
| two_row_reach_283271 | +1 | +0 | +1 | +2 | +0 | -1 |
| two_row_reach_283272 | -1 | +0 | +0 | +3 | +1 | +2 |
| two_row_reach_283273 | +0 | +0 | +0 | -2 | +0 | +0 |
| two_row_reach_283274 | +0 | +0 | +2 | +1 | +0 | +0 |
| two_row_reach_283275 | +0 | +0 | -1 | +5 | +1 | +1 |

保存池中的具体反例：283265 三目标 best 有效数从 6/12 降至 3/12，last 从 5/12 降至 1/12；LoRA last 仍有 12/12 个语义正确端点，说明目标定位正确没有消除中段碰撞。283271 则 best 从 6/12 增至 7/12、last 从 5/12 增至 7/12；283272_target0 的固定 last 从 0/4 语义正确、0/4 有效变为 4/4 语义正确、2/4 有效。全部其他条件仍在原分析中，未选这些例子替代总体结果。

## 固定 last3000 的 TRAIN 诊断

| 分组 | 条件/候选 | frozen TipValid | LoRA TipValid | frozen candidate ADE cm | LoRA candidate ADE cm |
|---|---:|---:|---:|---:|---:|
| old189 | 189/756 | 656/756 (86.7725%) | 692/756 (91.5344%) | 2.0100 | 1.4964 |
| new96 | 96/384 | 327/384 (85.1562%) | 344/384 (89.5833%) | 2.0792 | 1.5108 |
| all285 | 285/1140 | 983/1140 (86.2281%) | 1036/1140 (90.8772%) | 2.0333 | 1.5012 |

两个 TRAIN 分组均改善，说明此预算下 LoRA 提高了训练集拟合；DEV best 未稳定转化为更高有效率。共同初始 last12000 的 DEV 为 56/144 有效、27/36 Any、26/36 已分类 Unique，TRAIN 为 989/1140 有效。续训后的这些变化同时包含额外更新、新 Adam 状态和选择历史，原初始模型只能作上下文，不能当同新增更新预算对照。

## 成本与恢复：共享费用单列

每臂训练 issued：96000 tail replay + 96000 head + 3000 optimizer + 432 DEV tail + 432 DEV head，共 195864。每臂固定 last285 另有 285 tail + 285 head，共 570，产生 1140 个诊断路径状态。12 个 DEV 池总计 1728 个路径状态/臂。行政 pause2 包含在原 3000 步内，不是额外训练曝光。

| 臂 | 训练累计主体秒 | pause2+resume 外层秒 | 固定 TRAIN 诊断主体秒 | 固定 TRAIN 诊断外层秒 | 两阶段外层 GPU h | peak allocated bytes |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 1830.295408 | 1835.531759 | 69.637119 | 74.023181 | 0.530431928 | 4323820032 |
| lora | 2352.707010 | 2357.685516 | 73.652842 | 76.541939 | 0.676174293 | 4343086080 |

共享 prefix 的 321 full + 321 tail：主体 80.600073 秒 / 0.022388909 GPU h，外层 81.866961 秒 / 0.022740823 GPU h，只计一次。两臂训练+固定 TRAIN 诊断+共享 prefix 的外层 GPU 占用合计 **1.229347043 h**。这些主体数是嵌套成本，不能再与外层相加；共同 12000 预训练的 0.161166143 GPU h 已在历史结果计过，本轮不重复收费/计账。这里的 GPU h 是作业预留墙钟，不乘以 35% 显存限制。

LoRA 训练外层比冻结臂多约 28.45%；这不是在线推理时延。所有训练与评价通过当前尾层重放产生实际 Qwen 条件，未复用过时最终特征；本轮没有真实完整 Qwen 单请求 E2E 测量。CPU 零 forward 配对分析实际主体 59.764082 秒、外层 60.385129 秒，单独记录，不混入 GPU 训练成本。

| 阶段 | child PID | 开始 UTC | 结束 UTC | 外层秒 | exit |
|---|---:|---|---|---:|---:|
| frozen_train_pause2 | 687613 | 2026-10-03T01:43:09.776422+00:00 | 2026-10-03T01:43:51.020626+00:00 | 41.244204 | 0 |
| frozen_train_resume | 688762 | 2026-10-03T01:44:42.991123+00:00 | 2026-10-03T02:14:37.278678+00:00 | 1794.287555 | 0 |
| frozen_fixed-last-train_full | 703799 | 2026-10-03T02:15:31.095887+00:00 | 2026-10-03T02:16:45.119068+00:00 | 74.023181 | 0 |
| lora_train_pause2 | 704845 | 2026-10-03T02:17:14.572016+00:00 | 2026-10-03T02:17:56.457928+00:00 | 41.885912 | 0 |
| lora_train_resume | 705966 | 2026-10-03T02:19:28.269555+00:00 | 2026-10-03T02:58:04.069159+00:00 | 2315.799604 | 0 |
| lora_fixed-last-train_full | 724507 | 2026-10-03T02:58:38.310640+00:00 | 2026-10-03T02:59:54.852579+00:00 | 76.541939 | 0 |

两份 step2 内部 paused 状态原件保留在 administrative/；训练累计摘要和所有外层 job_records 同时保存。没有 hard-crash issued 重放或漏算恢复费用的证据，recovered_sealed_evaluation_seconds 两臂均为 0。

## 全图 QA、来源、权重与实际命令

已实际查看 [history.png](analysis/history.png) 与下列全部 12 父图：两臂 best/last、每父 3 目标、每目标 K4，XY/XZ 均保留。图可见高弧、外绕和折返；未裁去不利路径、未修改原图。原图没有障碍或参考路径叠加，目视不能独立认证碰撞；所有统计来自同一原封存 checker。具体目视记录和 SHA 在 [VISUAL_QA.json](VISUAL_QA.json)。

 | 父图 | 目视记录 |
|---|---|
| [two_row_reach_283264](analysis/two_row_reach_283264.png) | 三个目标均展示同起点的多分支XY绕行与XZ抬升；best/last中段形状有差异，终点多处叠合。 |
| [two_row_reach_283265](analysis/two_row_reach_283265.png) | LoRA last在target1/2的XZ抬升环较明显；仅靠终点叠合不能说明中段有效，三目标均保留。 |
| [two_row_reach_283266](analysis/two_row_reach_283266.png) | target1的last两臂都有较高抬升弧；target2的中段XY绕行也不同，未隐藏失败标记。 |
| [two_row_reach_283267](analysis/two_row_reach_283267.png) | target2显著抬升路径与低位路径并存；target1终点在best有可见两臂偏移。 |
| [two_row_reach_283268](analysis/two_row_reach_283268.png) | target1/2的best与last均有复杂中段折返，图中所有候选未被平滑或截短。 |
| [two_row_reach_283269](analysis/two_row_reach_283269.png) | target0的两臂外侧XY轮廓接近，但XZ局部不同；target1的last末端存在可见高度差。 |
| [two_row_reach_283270](analysis/two_row_reach_283270.png) | target1在两阶段均有高弧，其他路径低位接近；二维投影相近不能推出净空相同。 |
| [two_row_reach_283271](analysis/two_row_reach_283271.png) | target2存在向起点左侧延伸的XY分支；两阶段目标端点细微差异可见，保留原始比例。 |
| [two_row_reach_283272](analysis/two_row_reach_283272.png) | target0的last两臂末端明显分离；target1/2也完整呈现，未仅展示改进目标。 |
| [two_row_reach_283273](analysis/two_row_reach_283273.png) | target0的LoRA last高弧达到该图较高Z范围；其他目标仍完整呈现，没有裁去长弧。 |
| [two_row_reach_283274](analysis/two_row_reach_283274.png) | target1中段有多次折返；target0/2两臂末端偏移可见，未据图重判标签。 |
| [two_row_reach_283275](analysis/two_row_reach_283275.png) | target0/2有较大的XY外绕和XZ高弧；target1低位路径细节保留，所有失败交叉标记保留。 |

训练 source：`f41be1ff1a35b0eab3d36edf4ca23197359f4731`；零 forward 分析 source：`a3daf0da0c7a30279d38e9a4a18ee93b989911d4`。Qwen 官方 revision `89644892e4d85e24eaac8bacfd4f463576704203`；prefix manifest `4a3152db319e67c158f692a4b88e8575fd5e00b7ebe483ebfe4fafec1552bf15`。

| 原件 | SHA256 |
|---|---|
| [frozen/summary.json](frozen/summary.json) | `2809a62016802ccf9dfe8437d5ed5581d36edf95cf6aba08dbd765b28da3627e` |
| [lora/summary.json](lora/summary.json) | `9c034c2a03bb7df4171623c6ad5a34440fea96238f4a47760d464f314ff6a631` |
| [analysis/analysis.json](analysis/analysis.json) | `92f05e4013a2b909a66ad0df3a8e55be93cec13c92c89456c843865befb45f76` |
| [analysis/history.png](analysis/history.png) | `8a778f5acb743ca429975dee8d7f7654705a6884faa7be0a7e407799ef8776fd` |
| server frozen/best.pt | `b6298650e640fe8f69e5c3117b439340500384f440c203425d00c76145ac880b` |
| server frozen/last.pt | `86995f4da0bd948c4028c849de1dd23ac947b6b04440b99c7cda1b3078fbfa7e` |
| server lora/best.pt | `f7d98ae84373f5941f4c8fdab07e68aafa7c7d8e009cc24e55d921facbdf1add` |
| server lora/last.pt | `b9849166bd0437e41d17c10bb233d311fdeae0b0e2bf510fdbc8d27ba79997ff` |

原始下载 tar 为 44,410,880 B、SHA `a5a1d12e244ac75d45d7d03f3d770ec2370369e51801c3d77e4760869a2e0eb3`。250 服务器原件中 164 个文本/图原件保存于本目录、78 个小 NPZ 保存于被忽略的 `runs/synced_observed_two_row_lora_continuation_v1`；4 个 PT 和4份完整 requests 账本共8件只保留服务器真实路径/hash。两个行政 pause 原件另从先前本地收据纳入。完整索引见 [REMOTE_ARTIFACT_INDEX.json](REMOTE_ARTIFACT_INDEX.json)、[LOCAL_ARTIFACT_INDEX.json](LOCAL_ARTIFACT_INDEX.json)。归档和本报告没有新增模型调用、搜索、raw 数据读取或权重下载。

以下是已实际执行的冻结入口，完整 argv、PID、恢复字段见 job_records；现有已完成输出不能直接重跑覆盖：

```sh
taskset -c 1 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_f41be1f.sh train frozen pause2
taskset -c 1 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_f41be1f.sh train frozen resume
taskset -c 1 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_f41be1f.sh fixed-last-train frozen full
taskset -c 1 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_f41be1f.sh train lora pause2
taskset -c 1 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_f41be1f.sh train lora resume
taskset -c 1 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_f41be1f.sh fixed-last-train lora full
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/incoming/qwen_continuation_analysis_a3daf0d.sh analyze
```

MAIN 追加准备只输出 `.bootstrap` 候选文件，不原地写表。原273行 JSON 对象及CSV已有单元格必须不变；新增四行的训练/诊断费用各自只在 best 行记一次，last 引用相同 run；共享 prefix 成本单独保存于 COST_ATTRIBUTION.json。候选 helper 16 项纯检查通过，完成归档后又与实际两臂摘要逐指标核对。

以上仍是单种子、反复开发的 12 父场景结果；不能宣称开放语言、新任务/OOD 泛化、真实机器人执行或新的核心集合方法成立。
