# 主张—证据对应表

## 2026-10-03 Min-SNR真实TRAIN配对筛选

同初态/四流的500步修复：Tip470→494/1140、Any237→253/285，known unique212→234；95父42改善43退步，unknown193→237，高t75/99重建变差。保留uniform后验exit1及无重训恢复。只有一个训练种子的TRAIN内部证据，无DEV/新核心/泛化主张，MAIN293未改。外层GPU成本0.041738930小时；[完整分析与所有失败](observed_diffusion_minsnr_probe_v2/MIN_SNR_RESULTS.md)。下一+2500续训暂为提案，优先实际核心三臂。

## 2026-10-03：有序损失实际资源门、外部系统接口限制

e69有序损失服务器真实20tests/0skip；B32/K4/R9/H24/N12544全部15次synthetic forward/backward有限，A/B/C中位.014036/.140202/.196731秒，C峰值allocated432907776B，GPU外层5.368177秒。当前资源可做固定三臂3000步验证，训练器准备中；这些是合成数值/成本结果，没有方法质量优势。[原件](observed_ordered_relation_microbenchmark_v1/MICROBENCHMARK_RESULTS.md)。

HAMSTER已封存TRAIN0原4点按原reach检查TipValid@1=0：起点42.427cm、语义终点34.314cm且最近为另一目标，夹爪事件失败，原3段中2段不满足2cm净空。无加起点、截首点、H24、修复或重新生成，原K1/151forward不变。这反映当前稀疏manipulation-waypoint接口与派生reach任务不匹配，单TRAIN例子不能代表系统级机器人成功率或公平方法对照。[单独验收](hamster3d_saved_train0_check_v1/TECHNICAL_CHECK_RESULTS.md)。

Min-SNR普通臂实际完成500步及全部TRAIN诊断，外层postvalidation因recipe误写18000而非16000 inputs失败；内层12500完成且预算正确。修复只做新只读封存与接续weighted，不重训该臂、不覆写failed。MAIN293保持，核心方法成立仍待真实配对证据。

## 2026-10-03：可变布局TRAIN12真实小批结果

原eba采集324请求槽：108接受、162路线失败、54未尝试；后者是两个closed布局的浮点半毫米hash边界初始化失败，不能视为无解。270次实际restore均exact、108正例raw/H24验收和H24/H64重采样重现，15unknown保留。六柱四父108attempt/26接受，无条件已知类型>4；27/36请求目标条件有正例。原首4的238文件与12图字节不变，2registry仅append，全部失败和4.297m长弧保留。两开闭对尚无closed witness，未certify。root实际看ALL12_RGB、401008/401010/401005的target0完整页，agent完成其余新图；投影不代替原3D检查。新增8父outer1220.834587秒，两stage累计1893.042100秒，各nested worker口径不相加。此为TRAIN采集验证，非独立测试或方法优势。[完整结果](observed_layout_variation_pilot12_v1/PILOT12_RESULTS.md)。

## 2026-10-03 06:15 UTC：强基线真实预算与方法边界

真正K条件普通回归3000步已完成，K1/2/4/8有效率99.22/100/100/99.71%，已知Unique0.9922/1.9375/3.3984/4.9766。K8已637/639容量，384重复主要是R<K合法变体，无大幅去重空间。MAIN293保留旧289对象和CSV原单元格；共享单seed训练成本只计一次，模型曝光属性非加性，非旧single-K同曝光声称。[完整结果](budget_conditioned_regression_v1/BUDGET_RESULTS.md)。新布局pilot12、HAMSTER唯一完整请求在运行，技术/采集进展不算方法优势；Min-SNR尚待实际阶段。有序观测对应仍是单一待验证候选，核心新颖性、独立观测多种子优势与评分校准未成立。

## 2026-10-03 05:54 UTC：小批与技术门实测，尚无新方法优势

可变布局TRAIN4实际108槽：65接受/43失败/3接受unknown，108严格恢复、原验收与重采样重现；已知类型下界1–4。全目标图已看，按原协议继续余8父，不把采集管道当生成优势。[小批实测](observed_layout_variation_pilot4_v1/PILOT4_RESULTS.md)。HAMSTER16真实forward下8步输出逐值相同，decode比0.34049，35%内存门通过；完整一请求尚未执行，原失败保留。[技术对照](hamster3d_transport_exact8_v1/TRANSPORT_RESULTS.md)。普通真实K条件基线已45服务器测试通过、pause2后预定3000步已完成，原池分析待进行；Min-SNR只计划固定TRAIN配对诊断。MAIN仍289，核心成立/公平强基线优势/独立设置/多训练种子/评分校准未闭合。

## 2026-10-03 05:12 UTC：实测配对与当前决定

观测扩散真实12k两臂、teacher、全285 TRAIN保存池及3次独立采样全部完成。best Tip独立32.87%/集合29.17%，known Unique0.8426/0.9259；普通同曝光43.75%/0.9444仍强。TRAIN拟合34.39%/40.79%也弱，继续查基线去噪尺度，不把集合交互写成方法贡献。MAIN289保留原277行，全部12父图root实际查看；[原指标/预算/失败与权重hash](observed_two_row_diffusion_v1/DIFFUSION_RESULTS.md)。每臂1训练seed，3采样repeat不冒充3种子。

固定TRAIN空间probe实际完成但仅2碰撞段，`stop_underpowered`，封存不补数。[诊断](two_row_segment_observability_v1/SEGMENT_RESULTS.md)。HAMSTER真实官方加载并发出第一请求，300秒内87token未成完整JSON、其余5未发，0质量指标；[完整失败](hamster3d_train6_probe_v1/PROBE_FAILURE_RESULTS.md)。后续只先做同值传输工程核验，不能当新方法。

可变1/2/4/6障碍布局12 TRAIN父已完成114项真实测试与独立登记，首4正在实际采集，余8未启动；[登记证据](observed_layout_variation_preparation_v1/PREPARATION_RESULTS.md)。原TRAIN128质量已完成，256追加收集进行中；数据扩展不自动改本轮训练人口或启用新DEV。核心创新/公平强基线优势/独立设置与多种子、SelectedValid及校准仍未闭合。下一步真正K条件普通基线和基于实际失败的低噪声拟合修复，与采集/系统对照并行，不以新查询或普通匹配命名新核心。

## 2026-10-03 03:32 UTC 实测更新

Qwen冻结/LoRA普通集合回归配对已完整完成：best TipValid62→58/144（原规则选点），last53→62/144，TRAIN983→1036/1140；同draw/曝光，真实adapter更新。保留两臂，未证实稳定best优势；主要剩余失败是语义正确但路径碰撞。完整结果见[CONTINUATION_RESULTS](observed_two_row_lora_continuation_v1/CONTINUATION_RESULTS.md)，MAIN277保留旧273行。两臂与诊断/共享prefix外层成本1.229347GPUh，新增训练不是新机制。

TRAIN128全部3456槽质量与全新增图复核完成：2248接受/1208失败，938unknown有效参考、1已知类型重复均保留；[TRAIN128_RESULTS](observed_two_row_extension128_quality_v1/TRAIN128_RESULTS.md)。按原注册启动TRAIN256追加采集，新DEV仍封存，不改变当前285训练人口。

下一真实实验为观测独立/集合扩散普通强基线。历史receipt读取故障已保存并最小修复，6e实际37服务器测试通过；独立臂已pause2恢复至固定12k训练。尚无完成质量结果，随后按同数据/抽样/候选曝光执行集合臂与分别固定诊断，不以扩散或普通候选attention作为创新。HAMSTER官方固定权重正在本地校验下载，尚未加载。论文核心方法与独立设置、多种子、SelectedValid/校准和系统对照证据仍不足。

## 2026-10-03：extension TRAIN64 完整质量核验

前64父/192条件/1728请求槽全部保留：1131接受、597失败、465有效unknown、0已知重复，50条件已知R>K4；严格恢复1728/1728。新增32父576接受/288失败、20种颜色；长弧最长6.569m，未过滤。新96路线图及32front全目视，旧32图逐字节一致。仍为窄几何范围的RLBench-derived reach，不是OOD或新任务，不是全部解集或连续全臂安全证书。[完整数据报告与成本](observed_two_row_extension64_quality_v1/TRAIN64_RESULTS.md)。

原5c8 collector的train128 resume于01:16:26UTC独立启动，仅新增indices64..127；新DEV继续封存。该采集不改变当前composite108训练285输入。全321前缀实际等价检查已通过，正式frozen/LoRA续训尚未启动；不将数据量或缓存成功计为方法成绩。

## 2026-10-03：composite108普通数据控制已完成

固定总12000×32/K4，实际TRAIN输入189→285、参考1108→1663；原DEV best Tip59→63/144，但last62→56/144。公共TRAIN last76.06→85.05%，新增TRAIN90.10%。全部原件/逐场景/失败/成本见 [完整报告](OBSERVED_TWO_ROW_COMPOSITE108_BASELINE_RESULTS.md)。仅单种子、重复使用DEV的小幅best收益；不是同逐输入曝光、方法贡献或机器人执行证据。下一步骤先核真实Qwen末两层LoRA的冻结前缀回放与梯度，再预登记同common-head续训对照。此处尚无LoRA正式训练结果。MAIN273。

## 当前执行 — 2026-10-03 00:13 UTC

本段覆盖下方旧状态。source1fccf48的88项服务器测试通过，两臂各36次真实Qwen/K4已完成exit0，72份特征字节及288候选检查决定均与原best一致。完整请求中位constant/no-direct为74.5681/74.5213ms，P95为86.1414/91.2371ms；未显示可靠加速。原质量59/144对52/144保留。MAIN271，原269字段未改，新两行只记在线成本；见 reports/observed_two_row_12000_online_v1/ONLINE_RESULTS.md。

extension288首批train32已于00:02:21UTC结束，两shard exit0，原SSH89814结束。source1a3的完整TRAIN32分析于00:07:08UTC exit0：864已发槽、555accepted、309failed、247有效unknown、0已知重复，864严格状态恢复全部通过；无缺初始观测/blocked/跨父重复。96条件中20个已知类型数超过4，96全图正在独立视觉核验和归档。没有将提案失败隐藏或将unknown视无效。

source1fcc的composite108 readiness/export已独立完成，export于00:08:26UTC exit0；321输入=285TRAIN+36原DEV，1868正参考=1663TRAIN+205DEV。原225行字节保留，旧3缺失输入不替换。新DEV封存。下一步是新冻结pipeline的真实Torch测试、完整TRAIN表示/端点容量检查、仅新96条Qwen编码，然后同总12000步普通模型训练；所有阶段分别启动，当前尚未运行后四阶段，不把export成功称训练完成。pipeline已独立复核，本地22pass，3项Torch待服务器验证。

JOBS快照 2026-10-03T00:11:37.242427+00:00 共534条；registry实际同步。当前没有GPU训练或采集作业运行。待读完32质量后才单独启动train64 resume，使用原5c8f8e4固定collector，不重采前32。核心新方法优势仍未成立，数据扩大是普通基线控制。恢复需先核实际PID/status/immutable source；不重启上述已完成作业。

## 最新执行状态 — 2026-10-02 23:56 UTC

本段覆盖历史 running 记录。no-direct12000 已完成，source1a3eef1、PID628749/628753于23:38:29UTC exit0；118项真实服务器测试全部通过。相同初始化共有张量、384000抽样链、1536000路径状态和48次选择核验通过，但删除549120参数是容量混杂。固定last TRAIN有效率91.93%高于原76.06%，DEV39.58%低于原43.06%；best52/144低原59/144，已分类类型求和31比30多1但已知参考覆盖更低。停止该结构扩展；全部保存池/12父图/失败和权重hash已归档，0额外forward分析。见 reports/observed_two_row_prefix76_no_direct_v1/NO_DIRECT_RESULTS.md。核心方法优势仍未成立。

MAIN269，原267字段保留。JOBS快照为2026-10-02T23:52:09.965049+00:00，520条；registry更新实际状态。下一步仅固定两臂原best，各36次真实Qwen/K4，补齐12000模型在线成本；3文件已独立复核，本次冻结后单独tests/preflight/两臂，尚未在线运行，不搬用1500时延。

extension288 source5c8f8e4的train32 fresh仍活跃（session20261002T231813Z_train32_624110，child624144/624145，CPU2/3）。23:54机械检查24/32闭合，不等于24成功场景，raw尚未全32分析。首批全部闭合后，从1a3eef1单独运行TRAIN32分析，读质量后再独立train64 resume。新DEV保持封存。composite108固定旧64TRAIN+新32TRAIN+旧12DEV导出实现已61本地测试和独立复核通过，尚未实际导出；旧225行原字节保留，失败父不替换。后续质量/缓存/同总12000曝光普通训练仍待实际数据门禁，不自动启动。

恢复先检查上述PID的实际命令及status，禁止重复fresh或重采闭合父。所有服务器操作只使用不可变release。当前活跃会话继续执行；没有会话结束后自动分析/改代码服务。

## 2026-10-02 23:24 UTC 实施更新

TRAIN12条件探针已完成36 K4前向，未建立泛化/新方法优势；详见[原始诊断结果](OBSERVED_TWO_ROW_ROUTE_CONDITIONING_RESULTS.md)。全部69正参考被grid拒绝不能解读为路线类型不存在，原判定未改。下一普通去direct分支对照尚在实现；Qwen冻结、geometry与剩余decoder仍训练，减参是必须披露的混杂。

独立extension288的158服务器测试和prepare已完成，首批32TRAIN/864槽在CPU2/3采集，未提前报告质量。256TRAIN+32新DEV一次登记，不替换失败；新DEV后采并封存，旧12DEV保持reused。全部32父闭合后才做全槽质量核验与后续导出。详见[登记协议](OBSERVED_TWO_ROW_EXTENSION288_PROTOCOL.md)及[实际验证](observed_two_row_extension288_validation_v1/VALIDATION_RESULTS.md)。本次扩数据不构成核心机制，论文A–H仍未满足。

## 最新实际快照 — 2026-10-02 22:41 UTC

本段覆盖下方较早的running/待运行记录。root实核MAIN267（原263所有字段严格不变）、JOBS484@22:40:58.411450UTC无新job运行、registry501；429部署已登记。cosine与native100k均已实际完成，不能按旧记录重复启动。

两臂共同100k/2秒/K4的native传统对照，source42904f0，DEV均Tip96/144=66.67%、Any24/36；edge/spatial已分类不同有效数24/36=.6667、62/36=1.7222，已知类型覆盖.0319444/.0685185。全部36请求首路径相同；两臂均44附着失败+4有限错误目标、0节点上限/0超时。完整请求中位.707706/1.220680秒，共同body57.463680秒；更多类型保留了有效槽率，但请求耗时更高。它是标准传统规划的质量—覆盖—成本取舍，不是新学习核心、不比原20k同节点预算，也不构成固定端到端时间优势。完整100k归档由observation负责；原192edge等价证据保留，新100k不替代它。

cosine同预算12000末步TRAIN Tip94.84%改善拟合，DEV26.39%低于constant43.06%；保留best/last全部结果，停止LR扩展。[完整结果](observed_two_row_prefix76_cosine_v1/COSINE_RESULTS.md)。没有新核心方法胜利，也没有12000真实Qwen在线时延重测。formal116 reserved raw持续封存。

下一步仅只读评估独立256TRAIN+32新DEV父的数据扩展成本/去重/最小实现，保留旧12DEV的reused性质；尚未授权采集或修改旧collector。数据规模控制本身不能称方法贡献。

## 最新证据：cosine实际完整配对

| 主张 | 实际证据 | 判断 |
|---|---|---|
| LR之外的训练差异已受控 | 相同初始模型、384000抽样完整链、最终RNG/sampler；12000步/1536000状态/48选择均相同 | 本次单seed严格配对通过 |
| cosine改善末步TRAIN拟合 | Tip76.06%→94.84%，语义81.75%→98.68%，最近正例ADE2.791→.785cm | 支持本次拟合改善，不等于新机制 |
| cosine改善DEV泛化 | last Tip43.06%→26.39%、Any28/36→22/36；best40.97%→36.11% | 不支持；停止schedule扩展 |
| cosine所有覆盖指标均更差 | best knownUnique.8333→.8611、已知覆盖.0588→.0875，但有效率下降 | 必须保留取舍，不写全面劣势或方法胜利 |
| 改善只来自少数选择样本 | 全189TRAIN/36DEV四池、全12父图，0forward核对 | 未筛例；单seed、窄ID开发数据仍有限 |
| 新集合核心贡献或全机器人有效性 | 仅普通优化与tip2cm/goal3cm评价 | 仍未证明 |

[全部原始证据与完整历史](observed_two_row_prefix76_cosine_v1/COSINE_RESULTS.md)。source71e348，实际exit0。base528.534s/.146815GPUh；没有此12000模型的真实Qwen在线时延。100k传统两臂新预算root已启动TRAIN，不能预写完整结果。下方22:15 running与计数是历史快照。

## 最新证据：2026-10-02 22:15 UTC

实际快照22:15:44.071911UTC：MAIN263（原261所有字段不变），JOBS479、registry495；cosine PID593217/child593221于22:12:19.387085启动，SSH99280仍在，约4500步。此为该时刻running记录，不是完成状态。100k控制仅此一个档位，两臂共同变更、K4和2秒上限不改，尚未冻结或运行。

| 主张 | 实际证据 | 判断与边界 |
|---|---|---|
| native实现保持原edge算法结果 | 全192历史槽raw/H24/events、确定计数和checker决定一致；74Linux测试通过 | 支持这些输入上的等价，不证明任意输入全局浮点等价 |
| 标准空间惩罚能增加有用类型 | 同native DEV Unique.6667→1.4444；Tip66.67%→51.39%，Any同24/36 | 真实质量—覆盖取舍；非学习机制贡献 |
| native空间惩罚提供固定时间优势 | 完整请求中位.6264→.9263s；24失败达20k节点，无timeout | 不支持固定端到端时间优势；下轮100k是不同预算 |
| cosine恢复/调度实现通过真实门禁 | source71e348，91tests/0skip，3新Torchcases实际成功；逐步LR/基准LR/采样链受限 | 仅工程验证；fresh12000正在运行，尚无研究结果 |
| 已有新集合方法优势 | completion/refiner/crossgoal失败保留；不新增失败记忆微小机会审计 | 核心贡献仍未成立 |

[native完整证据](OBSERVED_TWO_ROW_NATIVE_ASTAR_RESULTS.md)与[cosine验证](observed_two_row_cosine_validation_v1/VALIDATION_RESULTS.md)均已原字节归档。下一100k两臂传统控制仍在实现。6000/constant12000完整在线Qwen时延未重测；不挪用旧1500时延。以下日期为历史语境。

## 最新证据：2026-10-02 21:41 UTC

后续唯一神经控制已固定为fresh12000 cosine配对：同初始化/抽样链/189输入/48次选模，无warmup/restart/LR扫描/新seed。状态为实现中、未启动；旧constant原源/结果保留。它是普通强基线修复的证伪实验，尚无LR因果或收益证据。


| 主张 | 当前判断 | 实测证据 | 边界 |
|---|---|---|---|
| 原64/1500足以作为充分训练普通基线 | 否；已补有界强控制 | 同数据12000，前1500全状态exact；DEV best40.97%/last43.06%，TRAIN best85.45%/last76.06% |8倍训练、48对6选模机会；非机制收益 |
|12000已稳定拟合全部TRAIN | 不支持稳定性主张 | best语义97.09%→last81.75%，净空87.96%→91.27%；138末端失败均≤4.288cm | saved anchor移动与失效相关，不证明LR/attention/梯度因果；不改3cm门槛 |
| 更低平均训练loss保证最后质量更好 | 不支持 |5500→12000最近100batch pathloss1.158e-4→9.461e-5，但TRAIN Tip85.45%→76.06% | 不是同批/完整TRAINloss；best与last池原样保留 |
| 空间惩罚实现低成本多样化优势 | 本次总体负面 | DEVUnique.6667→.8056，Tip96→42/144，中位.914→6.143s；56新超时 | 标准算法对照，所有失败照计；两臂native等价/加速尚未实测 |
| Formal116已全部闭合 | 仅机械状态支持 | 两shardexit0，116/116closure文件存在，21:10:27UTC结束 | 不推断保留角色轨迹数量/成功率；lockedraw未读 |

12000的全部12父图与原始42收敛文件已核验，原51文件索引保留；完整在线Qwen时延仍未重测。跨目标.7633cm<2cm的停止决定不因新的事后DEV描述而撤销。以下表格与日期保留历史语境。


## 最新证据：2026-10-02 20:45 UTC

| 主张 | 状态 | 实际证据 | 边界 |
|---|---|---|---|
| 扩至64TRAIN后普通头已充分拟合 | 不支持 | 完整prefix76归档：63实际父/189输入，last1500 TRAIN Tip20.11%；DEV best24.31%、last18.06% | 相同1500×32总曝光；每输入曝光下降，颜色覆盖也变化；不能称纯泛化瓶颈 |
| 32父普通头有容量拟合现有正例 | 有界6000控制支持 | 前1500全状态严格重放；last6000 TRAIN Tip97.04%、语义98.39%、候选→最近正参考ADE0.780cm | 四倍训练预算；不证明64父收敛、不证明机器人执行或新机制 |
| 充分训练后已解决32父DEV问题 | 不支持 | best2500 DEV Tip31.25%/Unique.7778；last6000 Tip34.03%/Unique.4167；目标正确仍碰撞68/52槽 | best有24次而非6次DEV选择机会；完整12图/所有失败保留；6000在线时延未测 |
| 跨目标对应辅助具有预登记实施机会 | 本次门槛未通过，停止 | first16TRAIN：碰撞减clear超包络差.7633cm <2cm；全部必要参考关系条件通过 | 只否定这次screen；同类型接近不是新颖性/因果效果，不用DEV放宽门槛 |
| 同数据传统观测多路径仍是强基线 | 实测支持，重复高 | A*32/64 Tip96/144与100/144，分类重复1.8056/1.9444；各4有限错误目标、44/40未生成槽 | K4同信息字段，但非同骨干/等时；仅tip检查，失败照计预算 |
| 标准空间路径惩罚能降低传统候选重复 | 待验证，尚未运行 | 固定Gaussian2voxels=.05m/penalty4，先TRAIN12输入old/new可行性，再原36DEV | 传统强对照，不是新方法；原proxy/终点/K4/20k节点/2s与标签隔离不变 |

指标命名：收敛报告原“匹配ADE”实为**候选→最近正参考ADE**，并非一对一匹配或saturation loss；公式/原字节保留说明见 [METHOD.md](METHOD.md)。后文日期状态保留历史，以本节为最新主张判断。论文核心优势仍未建立。

2026-10-02 20:05 UTC：新增prefix44普通基线的完整真实训练/在线/固定lastTRAIN证据。32请求父等曝光结果混合：Any提升，逐候选质量与已知Unique未提升；lastTRAIN36.02%显示欠拟合。不能把数据扩展当新机制，不能把16父拟合结论直接迁移至32父。下一次有界6000收敛和TRAIN部分对应审计尚无结果。

2026-10-02 19:30 UTC：两排观测普通基线已完成，best/last DEV均TipValid22.92%，有效已知重复0；固定末步TRAIN87.5%，表明当前主要是泛化差距。真实Qwen在线36请求中位73.15ms；传统观测A*为52.78%有效但平均1.31个已知重复槽、中位779.90ms。主表248行保留全部失败、相同checkpoint成本去重。下一步已确定前32/64TRAIN等曝光普通基线，不把扩数据或现有控制称为核心方法优势。

| 主张 | 状态 | 实际证据 | 边界 |
|---|---|---|---|
| 常规事件辅助在另一批父场景稳定保持收益 | 仅有限支持 | 原best宏ADE8.9879→8.5621cm，2/3seed改善；last8.9621→9.0961cm退化；完整12源固定迁移 | 同六任务DEV，不是OOD/locked；6/12父均值改善，不再调参，非核心方法 |
| 新静态初态避免当前初始化采集障碍 | 有界开发检验支持 | v6四布局全部门禁通过、零初始化规划；原失败组27槽23接受、14已知/9unknown/4失败 | 新RLBench-derived初态，旧失败未覆盖；窄ID，非旧动态复现/最短路径/广泛泛化 |
| 继续相同SFT曝光能解决自由路线生成 | 本次有界修复未支持 | 1500→6000后best NLL.416633；同8TRAIN贪心端点0/8≤3cm，mean23.391→27.769cm | 停止加步/DEV扩展；不是对所有SFT训练方案的否定，额外预算单列 |
| 事件位置辅助改善六任务普通头 | 三种子开发证据 | best ADE9.855→9.083cm、末端17.928→15.689cm；actual采样链/初始化配对通过 | 固定1500 ADE仅2/3改善，杯子退化；常规辅助，非集合创新/语义/碰撞/执行成功 |
| SFT空间失败来自单位或因果接口错位 | 当前小批审计不支持 | 9forward、1113位置×151936词表前缀logits完全相同；单位/mask/shift通过 | 不排除欠拟合/暴露偏差；真前缀末端拟合不能证明视觉定位，下一仅greedyTRAIN8 |
| 真实Qwen已完成直接路线集合SFT | 训练和生成实测完成；当前质量失败 | 3750路线目标槽，8adapter更新；独立4/整集合4全24DEV TipValid与语义均0，格式21/96与5/96 | NLL .4367不是路线质量；有限训练曝光尚不足以否定SFT能力，不把当前失败称强基线 |
| 扩充六任务TRAIN改善相同DEV的普通头 | 单种子开发证据 | 相同1500x32，12→48→96请求TRAIN父（最后95正参考），DEV ADE17.12→12.94→10.00cm、endpoint24.50→19.70→17.43cm | 同时增加variation/颜色；第三点8/12父改善、4变差，reach/slide末端>30cm，无语义/碰撞/执行认证，不是新机制 |
| 语法约束能解决VLM路线空间错误 | 当前TRAIN诊断否证 | 固定8TRAIN K1全部格式通过、终点0/8在3cm内，平均57.37cm；0eeeecb实测 | 标准格式修复；不是DEV结果，也不是训练充分性证明。停止自动DEV扩展 |
| 两排物理通道提供超过K4的已知有效类型 | v4/v5局部可采性支持 | v5四父108请求槽，61有效/49已分类；6/12目标有至少5种，81次严格恢复通过 | 一父setup碰撞保留27未尝试，局部扰动DEV_COLLECTION；无模型优势/广泛泛化，旧v1/v2失败保留 |
| 历史集合回归优于现有两类扩散 | 已复核 | historical commit d0d97eb；multiseed_summary.json；audit_v2/historical_regressor_seed0_metrics.json | 固定三路、真几何/真终点、冻结CLIP；旧TEST/OOD已公开使用 |
| 已构建可变路线类型和多障碍探针 | 已实现并验证 | multigate.py；audit_v2/multigate_v1.manifest.json；1152父场景7131正例 | 两墙平面通道结构、恒定z、固定语言；不是机器人观测任务 |
| 随机K参考子集会造成路径平均 | 受控单种子支持 | v2_round1逐场景结果、配对区间、ROUND1_REVIEW.md | 已知多模态回归问题，不能单独作为创新 |
| 类型不足K时应允许有效重复 | 受控单种子支持 | v2_round2/saturation_summary.json；精确饱和匹配与穷举测试 | 当前开发探针达到可覆盖类型上限；尚无独立任务结论 |
| 补全覆盖记忆稳定优于普通attention | 未获支持 | COMPLETION_THREE_SEED.md；三种子新增有效类型差+0.1641/+0.0156/-0.0469 | seed2反转；父场景bootstrap不能代替训练种子稳定性；不提升为核心方法 |
| 自产草稿补全优于一次联合生成 | 修正后仍未获支持 | v2_completion_selfdraft：attention2+2=3.3047/joint4=3.3125；coverage3.2813/3.3438 | 模型草稿混合确实修复大量失败，但双前向没有优势，保留joint4 |
| 真正使用Qwen3-VL训练/推理 | 已实测 | 官方固定revision；96真实RGB+语言前向；observed_frozen_v1；observed_online_v1 | 不是CLIP替代；真正接入不等于路线质量已达标 |
| 真实LoRA参数更新 | 已实测 | 末两层q/v共114688参数，8张量非零梯度+改变hash；observed_online_v1/lora_seed0 | 短训练无质量收益，不将“参数更新”解释为有效微调；缓存未复用 |
| 观测语义目标精确定位 | 部分积极开发证据 | OBSERVATION_GROUNDING_THREE_SEED：同RGBD头辅助监督使严格3cm均值1.04%→28.82%，终点18.92→14.51cm | 全24指令/8父，23参考；三种子一致但小DEV反复选优，常规基线修正不是核心创新 |
| LoRA提升观测路线质量 | 尚未获支持 | 300步从零头及500步同头warm-start配对；后者两边均选step0 | 实际参数更新与有效微调分别报告；当前不能称充分训练的强LoRA基线 |
| 同初态RLBench多路线采集 | 已完成小批实采 | derived32:275/288成功、32父、restore/RGB0差；4原任务12/12采集成功 | 采集成功不是模型执行；未据轨迹距离定义不同类型，连续全身碰撞未认证 |
| 真障碍下可判别的多路线采集 | 小批通过并正在扩大 | explicit-render四父48尝试20路径17类型，48/48恢复0；实际每步arm/gripper碰撞审计 | RLBench-derived扩展；失败21规划+7碰撞保留；不是原benchmark或生成模型成功 |
| 新布局训练后定位辅助仍有收益 | 已完成配对开发实验 | 新32父三种子 strict0→28.82%，参考终点18.54→12.19cm；新64父seed0 strict0→19.79%、终点15.70→10.50cm | 同一旧8父DEV，固定训练曝光；不证明扩大数据必然提高严格正确率 |
| 任务专用传统定位已是强对照 | 已实测 | TRAIN颜色原型＋RGB-D，旧/新训练集严格19/24与20/24；保留背景误选大离群 | 仅已见精确指令、单终点，无完整路径；参考误差23与目标中心误差24不能混比 |
| 局部编辑门控提高变化后集合质量 | 续训后仍未获支持，放弃此实现 | CONSTRAINT_UPDATE_V2_RESULTS：4000步b1/b2局部较自由补全少0.06537/0.11800有效类型 | 受控真几何；同数据/信息/网络规模、累计候选4+b；停止继续堆模块 |
| 神经观测模型已能稳定绕障 | 未获支持 | obstacle16 plain/aux TipValid2.08%/4.17%，Any均8.33%；12指令全量图 | 仅箱体末端线段检查，非手臂、桌面或执行认证；实际训练失败仍大量存在 |
| 路径内部损失与定位损失冲突 | 当前诊断未支持 | 固定6个TRAIN样本、best250/last1000共享参数梯度cos为正，损失重构误差<3e-9 | 一批局部诊断，不能证明全局无冲突；没有据此增加detach模块 |
| 软坐标均值会损害当前观测定位 | 配对三种子支持常规修复 | OBSERVATION_PEAK_ANCHOR；natural64严格语义17.36±13.19%→83.68±6.62%，参考ADE6.47→4.55cm；同参数、数据、曝光 | 仍是旧8父DEV；峰值读取是传统定位修复，不能包装为集合新机制 |
| 峰值读取稳定改善障碍路线 | 尚未获支持 | obstacle32原ADE选模语义19.44→28.47%但峰值seed1为0，ADE18.38→19.30cm；seed0 TipValid4/48→8/48 | 额外固定步检查必须双方所有seed；不替换主选模；box tip检查不等于机器人执行 |
| 在线RGB-D LoRA改善共享预训练头 | 未获支持 | OBSERVATION_ONLINE_RGBD_PAIR：两臂1000×4，均选共同step0；last均退化 | 真实Qwen、8adapter张量实际更新；原始失败与公共预训练成本保留 |
| 观测A*传统基线已合理运行 | v2已实测 | OBSERVED_ASTAR_V2_PROTOCOL：固定TRAIN修复后44/48 TipValid，UniqueClassified1.333，median1.969s；v1失败保留 | 闭集指令原型、末端箱体检查；低覆盖且非整机执行，尚无固定时间公平对比 |
| 峰值定位收益迁移到新布局 | 三种子开发支持 | OBSERVATION_FRESH_DEV_TRANSFER：原64模型soft28.47±3.18%→peak72.05±3.14%，无新训练 | 新16DEV已用于分析，不是锁定TEST；原型单端点40/48仍强 |
| 新192观测训练有效 | 配对seed0实测 | OBSERVATION_RESERVED192_PAIR：原ADE-best语义66.67%→95.83%，各384000槽 | 仅常规定位修复，peak末步回落87.5%，相对64数据与曝光均三倍 |
| v参数化解决扩散路线失效 | 未获支持 | DIFFUSION_MULTIGATE_V3_RESULTS：12000步ind/set Unique1.21354/1.09375，Valid40.17%/33.72% | 单训练seed；累计曝光为回归4倍，仍在改善不能称收敛；停止此支线 |
| 错误参考或H24重采样导致观测碰撞 | 当前TRAIN诊断不支持 | OBSERVATION_TRAIN_GEOMETRY_DIAGNOSTIC：181/181原始与H24正参考通过固定检查 | seed0模型前段碰撞仍多；不以此证明未观察几何/整臂安全 |
| 模拟器跨进程同父恢复 | 已实测通过 | observation_multitask_resume_probe_v1：world/RGB-D/camera/language精确一致，首记录hash未变，3/3任务成功 | 只有恢复机制证据；不等于六任务已完成或路线类型可判别 |
| 新96障碍普通定位头有积极结果 | 单种子配对支持 | OBSERVATION_OBSTACLE_RESERVED96_PAIR：新8父/24DEV，soft/peak最佳Tip44.79%/52.08%、Unique.75/1.125 | 常规修复；peak末步Tip43.75%退化；同数据A*87.5%/1.25仍强，非新核心 |
| 缓存头质量对应真实在线Qwen推理 | 实测一致 | 同24请求的路径最大差<=1.431e-6m；全链生成中位61.73/77.20ms | 包含读取/processor/Qwen/RGB-D/head，评分、碰撞验收和执行不在计时内 |
| 观测球形边界提供可分配路线分支 | 被当前TRAIN审计否定 | OBSERVED_DEPARTURE_REGION_AUDIT：24指令×3半径全部J1、无失败或预算耗尽 | 放弃此表示，不外推场景只有一条路线；不实现分配器 |
| 早段碰撞附近有局部可见几何支持 | TRAIN诊断支持 | OBSERVED_LOCAL_SUPPORT_AUDIT：33首次精确接触的10cm邻域均有点，前半弧长覆盖30/33 | 可见点支持不是已知自由空间；680参考前缀点仍有遮挡/画外，更新界限未证明充分 |
| 研究初版核心已成立 | 尚未成立 | REMAINING_EXPERIMENTS.md | 主要缺口是有效观测规划和稳定核心优势；剩余工作不只是扩大规模 |
| 一次局部观测更新改善集合 | 当前配对未支持，停止此实现 | OBSERVATION_REFINEMENT_PAIR：best修1坏0，last修0坏3；局部耗时增加39.1%，同8完整路径状态预算 | 两臂同参数/初始化/采样/曝光；逐坐标更新未饱和；仅常规修复，不将微小best增益称核心 |
| 白名单相机策略下跨进程恢复 | 新独立父实测通过 | observation_multitask_validated_resume_probe_v1：3/3成功，全部初态/输入/首slot hash保持，46.397s | 新144父批次已实际运行但未完成；五任务单父相机等价不外推所有父，push保持开启 |

所有开发集bootstrap区间只描述已选择开发数据上的配对差异，不能作为最终锁定测试的确认性推断。未实测字段在MAIN_RESULTS中留空。
# Latest six-task/direct-VLM boundary, 2026-10-02 13:37 UTC

Sealed prefix24: six original tasks,24 parents/72 successful references,12 TRAIN and12 DEV parents; three source attempts per parent. Actual frozen Qwen96 encodings and ordinary free-endpoint head1500-step training completed from6bc2b8. TRAIN/DEV macro path ADE1.783/17.121cm shows a large generalization gap, not representative task success. Semantic, collision, UniqueValid and execution results remain null. Complete provenance: `observed_multitask_prefix24_v1/artifact_index.json`.

Direct VLM SFT boundary: real2B preflight v1 failed memory, v2 passed exact-prefix masking and4 actual LoRA steps at4.839GB. This establishes a working memory-bounded training interface only; full SFT and independent-versus-whole-set generation have not completed. No new core advantage follows from either result.


## 2026-10-02 16:50 UTC — latest method/data evidence

| Claim | Status | Measured evidence | Boundary |
|---|---|---|---|
| Same physical initial state can support more than K4 route relations | Single-parent feasibility supported | v4:8/9valid,5 distinct lateral sequences,9 exact restores | 3 valid unknowns,1 H24 type-instability rejection; one target/layout, no learned-method claim |
| Response-aware pair selection improves on geometry/DPP for single gate closures | Rejected in current controlled setting | TRAIN768:3806closures,3424solvable; all720evaluable parents tie at100% AnyValid | Reference controls privileged; stop module, never promote weak K4prefix2 comparison |
| Real K2 ordinary small-budget baseline exists | Registered and training, outcome pending | Same3000x64/old static selection,384000 path states | Half K4 path budget, different query count; no causal joint-risk claim |
| Extra SFT exposure restores route quality | Unverified; actual recovery implementation tested |13server CPU tests pass, including exact tiny continuous/continued/resumed loop | Original1500 results remain;6000 GPU outcome not yet present |
