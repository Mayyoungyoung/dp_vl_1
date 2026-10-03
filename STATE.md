# State — 2026-10-02

## 实际技术门禁完成 — 2026-10-03 01:10 UTC

此段覆盖旧待probe/采集运行描述。eddfacaddab2d12c67f5a56fd775de173c05f9b2已推送核验、冻结部署；37服务器测试0skip通过。真实Qwen TRAIN6 probe PID669594/669595于01:00:36.984377–01:01:24.293499UTC完成exit0。历史初始缓存6/6、写盘重读prefix回放6/6、两步full/replay特征/损失/梯度/optimizer/RNG及更新后特征全部逐值一致。8LoRA张量及60head张量真实改变、625冻结参数hash全同。实际10full+10tail+4head/16候选状态、4optimizer执行，无DEV、无额外重试。见 reports/observed_qwen_prefix_replay_probe_v1/TECHNICAL_RESULTS.md；49本地文件逐SHA核验，13PT仅远端索引。

probe主体46.065015726秒/.0127958377保守GPUh，外层47.309122秒/.0131414228GPUh，嵌套不能相加；allocated峰值4375957504B，reserved4406116352B。6prefix共2495388字节。仅支持串行原长度回放，不是方法质量、端到端时延或正式训练加速证据。私有.venv-qwen仅新增固定pytest8.4.2及测试依赖，原Torch/transformers版本未改；安装记录保留。不要重跑已完成fresh-only probe。

下一轮普通强基线已决定：同一composite last12000权重、两个新AdamW、各追加3000×32/K4曝光，frozen vs末2层q/v LoRA（rank8/alpha16，头3e-4、LoRA1e-5），同固定draws、12个旧DEV选择，fixed-last285独立诊断。技术前置已通过；321输入prefix corpus与完整训练/恢复/调用账本实现中，尚未冻结或正式运行，仍需实际工程测试和全321缓存等价检查。没有更改训练人口285/参考1663或解封新DEV。

extension train64 session20261003T001613Z_train64_651720已于01:00:01UTC两shard exit0，SSH57754结束。root随后独立启动source1a3eef1完整TRAIN64分析，PID670625/670626，01:02:48.257755UTC，SSH16931，CPU1/无GPU；输出 runs/observed_two_row_extension288_quality_v1/train64。此处尚未记录分析完成结果。前32闭合父不重采，分析读全部64已闭合TRAIN；质量读完再决定train128，不能自动开始。reserved原始数据继续封存。

MAIN保持273，技术probe不增加质量行。JOBS快照 2026-10-03T01:09:30.156504+00:00 共578条，registry已同步。核心方法优势/论文A–H仍未成立；当前活跃会话继续正式配对实施，没有会话外自动研究服务。恢复先读最新status与实际PID，不能根据下方历史段重启已完成作业。

## 实际完成与接续 — 2026-10-03 00:50 UTC

本段覆盖下方旧running状态。composite108普通12000已于00:27:47UTC exit0，固定last285 TRAIN诊断于00:28:48UTC exit0。source71cf0c1，70服务器测试通过。全部46服务器原件、18离线分析产物、12父全图和权重hash已核验；见 reports/OBSERVED_TWO_ROW_COMPOSITE108_BASELINE_RESULTS.md 及 reports/observed_two_row_composite108_v1/INDEX.md。此轮不重启。

原36 DEV best Tip59→63/144、已分类类型总和30→34、Any30/36不变；last62→56/144、Any28→27/36。公共189 TRAIN last575→643/756（76.06→85.05%），新增96为346/384（90.10%）。扩大数据改善训练拟合，但DEV收益不稳定；数据人口/颜色频率/逐输入曝光变化不是方法贡献。保留48次历史、全部失败和unknown。新训练base580.198116s/.1611661434GPUh，完整job593.489648s；诊断CPU19.117344s，嵌套成本不相加，没有新模型真实在线时延。MAIN273，旧271字段保留。

下一实际研究决定为同一composite last头的冻结Qwen/末两层LoRA普通基线配对。先实现TRAIN6技术门禁：官方完整forward与layer26冻结前缀缓存后的原尾层回放，两逻辑微步、两独立optimizer共4次更新；上限10full+10replay+4head/16候选状态，无DEV。此时实现已完成、独立审查中，尚未运行真实probe，更未启动正式LoRA续训。不会使用更新后过时的最终4096维条件缓存；只核验serial原长度，未证明batched padding。技术通过后另行冻结续训协议，不自动启动。

extension train64仍由原5c8f8e4 collector/session20261003T001613Z_train64_651720在CPU2/3执行，SSH57754；只追加注册32..63，前32不重采。全64闭合前只读机械进度，不读新增raw；之后独立TRAIN64分析，不能自动启train128或解封新DEV。所有reserved原始数据保持封存。恢复先核PID/status实际命令，禁止重复启动已发父。

JOBS实际快照 2026-10-03T00:47:54.427626+00:00，566条；registry已同步。当前没有GPU训练任务。核心方法优势及论文A–H尚未成立，当前活跃会话继续实施，不存在会话结束后自动分析改码服务。

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

## 最新执行状态 — 2026-10-02 23:56 UTC

本段覆盖历史 running 记录。no-direct12000 已完成，source1a3eef1、PID628749/628753于23:38:29UTC exit0；118项真实服务器测试全部通过。相同初始化共有张量、384000抽样链、1536000路径状态和48次选择核验通过，但删除549120参数是容量混杂。固定last TRAIN有效率91.93%高于原76.06%，DEV39.58%低于原43.06%；best52/144低原59/144，已分类类型求和31比30多1但已知参考覆盖更低。停止该结构扩展；全部保存池/12父图/失败和权重hash已归档，0额外forward分析。见 reports/observed_two_row_prefix76_no_direct_v1/NO_DIRECT_RESULTS.md。核心方法优势仍未成立。

MAIN269，原267字段保留。JOBS快照为2026-10-02T23:52:09.965049+00:00，520条；registry更新实际状态。下一步仅固定两臂原best，各36次真实Qwen/K4，补齐12000模型在线成本；3文件已独立复核，本次冻结后单独tests/preflight/两臂，尚未在线运行，不搬用1500时延。

extension288 source5c8f8e4的train32 fresh仍活跃（session20261002T231813Z_train32_624110，child624144/624145，CPU2/3）。23:54机械检查24/32闭合，不等于24成功场景，raw尚未全32分析。首批全部闭合后，从1a3eef1单独运行TRAIN32分析，读质量后再独立train64 resume。新DEV保持封存。composite108固定旧64TRAIN+新32TRAIN+旧12DEV导出实现已61本地测试和独立复核通过，尚未实际导出；旧225行原字节保留，失败父不替换。后续质量/缓存/同总12000曝光普通训练仍待实际数据门禁，不自动启动。

恢复先检查上述PID的实际命令及status，禁止重复fresh或重采闭合父。所有服务器操作只使用不可变release。当前活跃会话继续执行；没有会话结束后自动分析/改代码服务。

## 当前执行与下一决策 — 2026-10-02 23:24 UTC

本段覆盖下方历史running记录。MAIN267不变；JOBS497@23:23:57.395618UTC、registry随后按同快照同步。source905214e的TRAIN12诊断已经完成并归档：36 K4前向/0搜索/0更新，直接分支swap使净空45→33/48（14坏2好）；仅通道敏感性，不是DEV因果或方法质量。69条正参考完整grid通过0受到终点voxel与其它保守阻塞共同影响，不能推导模式不存在。[完整结果](reports/OBSERVED_TWO_ROW_ROUTE_CONDITIONING_RESULTS.md)。暂缓learned-field，下一唯一神经控制为去direct分支的普通头：相同constant12000曝光/共有init/抽样，geometry等剩余模块继续训练，参数量减少明确披露；尚未server训练。

extension288 source5c8f8e4f5cd478c793a0e0d9640005deaf700973已实测158tests通过、独立prepare成功并封存来源。root单独启动train32 fresh，session `20261002T231813Z_train32_624110`，SSH89814，CPU2/3、GPU隐藏。两coordinator PID624140/624142，child624144/624145；23:23:57快照前四parent job exit0，父004/005运行中。这是机械进度，不是成功场景/有效路线数。此阶段只32父864请求槽，后续不会自动采集。新DEV须全部256TRAIN机械闭合后另启并继续封存。

恢复前先核实session/status、parent status与PID实际command；完成的stage和已发父绝不重采。确有中断且现有coordinator已退出时，同source恢复命令为 `bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/5c8f8e4f5cd478c793a0e0d9640005deaf700973/scripts/launch_two_row_extension288_v1.sh 5c8f8e4f5cd478c793a0e0d9640005deaf700973 train32 resume`。只有前32全部闭合并读完整质量后，才独立执行train64。GPU1当前无研究训练占用；不重复运行probe。没有后台自主决策服务，当前活跃会话继续实施与分析。

## 最新实施状态 — 2026-10-02 23:13 UTC

本段覆盖下方较早的待运行和资源状态。研究分支 `codex/multiroute-v2` 的905214e8050d594b157dc7cacf6b446e7c183c85已推送、核验并冻结到服务器。TRAIN条件诊断14项真实Torch测试全部通过（2.03秒、0跳过），随后独立audit于23:09:22–23:10:13UTC完成exit0，PID619059/619064。36次K4前向、144路径状态、0新Qwen编码、0搜索、0优化更新；不是MAIN质量实验。正常预测复现、identity、geometry和权重字节门禁通过。逐场景解释与归档正在处理，不重跑已完成诊断。

新extension288实现为独立注册和采集器：256TRAIN＋32后采集封存DEV，7776预登记提案，首个明确阶段只有train32；后续阶段按真实闭合状态单独推进。旧代码/116父数据/失败/角色保持。当前本地完整157测试及最终采集20测试通过，服务器tests/prepare/采集尚未运行，须按真实job receipt更新。8GiB新增内部预算、CPU2/3、GPU隐藏；原12DEV继续标记reused。用户已授权的研究闭环继续，新增数据不是新方法证据。

cosine和native100k结果已完整归档、实核并提交82c17d2。MAIN267保持；新诊断不写作泛化收益。cosine改善TRAIN拟合但未改善DEV，停止LR扩展。native100k的空间惩罚增加路线类型但请求成本更高，仅是传统对照。论文核心优势仍未成立。

## 最新实际快照 — 2026-10-02 22:41 UTC

本段覆盖下方较早的running/待运行记录。root实核MAIN267（原263所有字段严格不变）、JOBS484@22:40:58.411450UTC无新job运行、registry501；429部署已登记。cosine与native100k均已实际完成，不能按旧记录重复启动。

两臂共同100k/2秒/K4的native传统对照，source42904f0，DEV均Tip96/144=66.67%、Any24/36；edge/spatial已分类不同有效数24/36=.6667、62/36=1.7222，已知类型覆盖.0319444/.0685185。全部36请求首路径相同；两臂均44附着失败+4有限错误目标、0节点上限/0超时。完整请求中位.707706/1.220680秒，共同body57.463680秒；更多类型保留了有效槽率，但请求耗时更高。它是标准传统规划的质量—覆盖—成本取舍，不是新学习核心、不比原20k同节点预算，也不构成固定端到端时间优势。完整100k归档由observation负责；原192edge等价证据保留，新100k不替代它。

cosine同预算12000末步TRAIN Tip94.84%改善拟合，DEV26.39%低于constant43.06%；保留best/last全部结果，停止LR扩展。[完整结果](reports/observed_two_row_prefix76_cosine_v1/COSINE_RESULTS.md)。没有新核心方法胜利，也没有12000真实Qwen在线时延重测。formal116 reserved raw持续封存。

下一步仅只读评估独立256TRAIN+32新DEV父的数据扩展成本/去重/最小实现，保留旧12DEV的reused性质；尚未授权采集或修改旧collector。数据规模控制本身不能称方法贡献。

## Current checkpoint — cosine COMPLETE, actual exit 2026-10-02 22:21 UTC

This block supersedes earlier running statements;22:15 JOBS/MAIN numbers below remain a dated historical snapshot.

单次cosine普通控制已实际完成，source71e34850481b79bdbc828d1c1944f0de00329960，PID593217/593221于22:21:21.678075UTC退出0。91服务器测试均通过；初始化、完整384000抽样链及最终RNG/sampler严格同constant12000。两臂均1536000训练路径状态、48次DEV选择，只有LR日程变化。相同64请求TRAIN父中63可观测父/189输入，原缺失父不替换，DEV固定36输入。

固定last12000 TRAIN Tip575/756=76.06%→717/756=94.84%，语义81.75%→98.68%，候选→最近正参考ADE2.791→.785cm；DEV Tip62/144=43.06%→38/144=26.39%、Any28/36→22/36、knownUnique.6389→.4444。cosine原规则best4250 DEV Tip52/144=36.11%、Any27/36、knownUnique.8611；constant best5500为59/144=40.97%、30/36、.8333。保留cosine best多1个跨条件求和已分类有效类型的取舍，不写全面优胜。严格配对支持训练拟合改善，不能支持DEV泛化改善或新集合贡献。

已归档全部189TRAIN/36DEV best/last保存池、原始cost/LR/history/checkpoint hashes和12父全图，0新增forward完成配对与QA。DEV末步Tip有7条件改善/9相同/20下降，端点和线段净空均有缺口；cosine最后3000步DEV Tip26.39–31.25%，不只是最后一次偶发失败。停止schedule扩展，保留constant有效率更强的普通基线。base528.534秒/.146815GPU小时，外层542.291秒；没有重测12000完整Qwen在线时延。

读取 reports/observed_two_row_prefix76_cosine_v1/COSINE_RESULTS.md。下一传统控制是两臂共同100k节点：root已从42904f0启动TRAIN门禁SSH66842，此处尚未记录其结果；不自动启DEV或更多节点档。它改变搜索预算，不是20k同预算或新学习机制。formal116仅机械完成，reserved raw仍封存；新核心方法优势未成立。MAIN/JOBS由root另行更新，不猜测最新计数。

## Current checkpoint — 2026-10-02 22:15 UTC

实际快照22:15:44.071911UTC：MAIN263（原261所有字段不变），JOBS479、registry495；cosine PID593217/child593221于22:12:19.387085启动，SSH99280仍在，约4500步。此为该时刻running记录，不是完成状态。100k控制仅此一个档位，两臂共同变更、K4和2秒上限不改，尚未冻结或运行。

This block supersedes older running/pending statements below; retain those as history. Native two-arm A* source b184060 is COMPLETE, archived and visually reviewed. All192 historical edge slots match the native implementation in raw/H24/events, deterministic counts and checker decisions. DEV edge/spatial Tip96/144=66.67% versus74/144=51.39%; knownUnique24/36=.6667 versus52/36=1.4444; Any24/36 for both. Full-request median .6264s versus.9263s. Spatial has44attachment failures+24node-cap failures+2finite wrong-goal routes; no timeouts. This is a standard quality–coverage–cost tradeoff, not a learned contribution or fixed-time advantage. Read reports/OBSERVED_TWO_ROW_NATIVE_ASTAR_RESULTS.md.

Cosine source71e34850481b79bdbc828d1c1944f0de00329960 passed91 actual server tests in8.85s,0skip including3newTorch cases. Validation PID592391/592393 ended22:10:54.265947UTC;18evidence items and335source hashes are archived in reports/observed_two_row_cosine_validation_v1. Root separately launched fresh12000 onCPU0/GPU1/35%,SSH99280; RUNNING per the latest launch notification. Do not relaunch or read dependent completion artifacts before actual exit. Same initial tensors/full sample-chain/189TRAIN/K4/H24/1536000states/48DEV choices; only one cosine cycle,no warmup/restart/floor/sweep. No result or LR-causality claim yet.

Next conventional control is one jointly raised100000-node budget for BOTH native arms, being implemented independently. It is a new node-budget setting, not the original20000-node comparison and not proof of fixed end-to-end time advantage. No new set module or failed-memory audit is scheduled. Formal116 remains mechanically complete and reserved raw sealed.

## 2026-10-02 22:04 UTC：同预算cosine普通控制准备

独立五文件已实现、复审无阻断；本地13pass/3因无Torch而skip。下一步在冻结源码上通过实际Torch恢复等价/历史回归门禁，之后单独启动fresh12000。相同真实Qwen缓存、初始化、189 TRAIN输入完整抽样链、K4/H24、损失、1536000路径状态和48次DEV选择；仅替换为无warmup/restart/floor的单周期cosine。原constant12000记录保持不变。不因新DEV选择更多步/曲线/seed。

native b184060的Linux74测试、TRAIN24请求96槽和DEV72请求288槽均实际完成退出0。DEV edge Tip96/144、Unique24/36；spatial Tip74/144、Unique52/36；Any均24/36。spatial24节点上限失败全部20000节点、搜索中位18.434ms，无超时。历史Python/native edge全部192槽raw/H24/events/确定性搜索计数及checker决定一致。全归档/逐图QA正在收尾，MAIN尚未加入这两新行。此为标准传统对照的质量覆盖成本取舍，不是新集合贡献。

## 2026-10-02 21:54 UTC：下一配对控制源码冻结准备

证据e2f67751e05dec9986a15234b8492f773077d789已推送并核验。两臂共同native A*内核、三阶段runner/launcher及门禁测试已实现并独立复核；本地实际编译74 tests通过，Linux服务器编译/测试与真实请求尚未运行。生产沿用K4/20k节点/2秒/64节点计时，两臂共同使用同一库，原planner/空间场/评价器源码未改。首次tests stage必须有真实24项native差分及同编译器/flags/源码生产构建证明，root读后才能单独TRAIN12，再单独DEV36。

实现审查已在真实请求前修复两处测试入口问题：其它测试不能替代24项native差分；单元测试的模拟build receipt不能被当作真实编译。一次本地测试临时目录错误设在源码内被既有保护拒绝，保留失败日志，移至OS temp重测，未放宽保护。新的cosine12000普通基线仍在独立实现，未启动。

## Current checkpoint — 21:41 UTC / 2026-10-03 05:41 China

- Current snapshot: MAIN261 rows (previous258 fields unchanged), JOBS473 at21:40:55UTC with no live PID, registry489. Next neural ordinary control is one fresh12000 cosine-LR pair, same initialization/sample chain/189 inputs/48 DEV selections; implementation only, no launch. Preserve constant-LR source/results. No warmup, restart, LR sweep or new seed; endpoint drift motivates the test but does not prove LR causality.

This block supersedes older running/pending statements below; retain them as historical records.

- Prefix76/12000 ordinary control from1417cdc is COMPLETE:98 actual server tests, exact original64 first1500 state proof, separately launchedfinish, both exit0. Best5500 DEV Tip59/144=40.97%,Any30/36,knownUnique.8333;fixedlast12000 Tip62/144=43.06%,Any28/36,knownUnique.6389. BestDEV has2classified duplicate slots;last has0. Full history, all saved pools and12DEV plots are archived and reviewed.
- TRAIN improves from original1500 Tip20.11% tobest85.45%/last76.06%, but last semantic734→618/756 falls relative tobest whileclear665→690 improves. All138last endpoint misses are3.006–4.288cm;133 lie within3–4cm. Shared-anchor movement correlates with degraded conditions;constant LR instability is only a hypothesis, not proven cause or wrong-object identification. Do not automatically extend or select another checkpoint.
- Actual12000 budget1536000pathstates/384000draws;48DEV selection opportunities versus24for32/6000 and6for64/1500. Per-input exposure ratio to32/6000 is62/63, totaltraining2x. Outer stage cost559.381s,base nested536.405s;no12000 real-Qwen online latency was measured. See reports/observed_two_row_prefix76_convergence_v1/CONVERGENCE_RESULTS.md.
- Standard spatial-penalty A* is COMPLETE and unfavorable overall:DEV knownUnique.6667→.8056,Tip96/144→42/144,median request.914→6.143s;56new search-timeout slots,all retained. Field preparation explains only6.18%of extraDEV time. It is a traditional baseline tradeoff,not a method contribution.
- Next implementation only: jointly compile both originaledge andspatial arms without changing neighbor/cost/tie/geometry/path semantics or20k/2s limits;first prove equivalence then measure. No native experiment or gain yet. The prior cross-goal screen remains stopped at.7633cm<2cm.
- Formal116 collection mechanically COMPLETE:both source2626487shards exit0,session ended21:10:27UTC;116/116closure filenames exist(64TRAIN/12DEV_MODEL/12DEV_SCORE/12CAL/16LOCKED). This does not mean116successful scenes or3132valid trajectories. Only opaque hashes/statuses ofreserved roles were read;lockedraw remains sealed. Do not restart completed collectors or training. Corepaper criteria remain unmet.


## Implementation checkpoint — 21:12 UTC / 2026-10-03 05:12 China

- Full preceding evidence is pushed and remote-verified at84cd563b99fb37e308b05eb310ff0f563bb8e123: MAIN258, historical251 rows unchanged. Current new source contains an independent prefix76/12000 ordinary driver with exact first1500 reproduction gate, and a standard spatial-penalty A* adapter. Both are implemented, not yet server-validated or launched. Keep original6000/base/planner/collector files unchanged.
- Local convergence new+old checks61passed/2Torch skips; spatial new+old23passed, independent spatial review has no blocker. Actual server Torch validation is still required. A copied budget-reporting defect was caught before any run and replaced by derived stage/total budget identities. Spatial shared-fit identity excludes only the existing fitter's measured training_seconds, retaining all parameters/data hashes.
- At21:06:51UTC the formal116 mechanical-only audit saw112 closure filenames:64TRAIN/12DEV_MODEL/12DEV_SCORE/12CALIBRATION/12of16TEST_LOCKED. Session had no finish/exit marker yet. No closure contents, locked arrays or outcomes were opened. Do not treat marker counts as successful references.

## Current checkpoint — 20:45 UTC / 2026-10-03 04:45 China

This block supersedes all older running/pending statements below. The older dated blocks are historical records, not restart instructions.

- Prefix76 (64 requested/63 observed TRAIN parents,189 inputs) has completed preparation, frozen-Qwen cache, ordinary1500x32 training, fixed-last TRAIN diagnosis, actual online36-request evaluation and full archival. Best500 DEV Tip24.31%,Any33.33%,knownUnique.6667;last1500 Tip18.06%,Any61.11%,knownUnique.1111. Last TRAIN Tip20.11% despite semantic92.99% remains severe underfitting at this exposure; it cannot establish a fully fitted64-parent baseline. See reports/OBSERVED_TWO_ROW_PREFIX76_BASELINE_RESULTS.md.
- The separate prefix44/6000 ordinary control completed from ac6882c, following73 actual server tests and exact first1500 model/optimizer/scheduler/scaler/RNG/sampler/history proof. Last TRAIN Tip97.04%,semantic98.39%,candidate-to-nearest-positive-reference ADE0.780cm;last DEV Tip34.03%,knownUnique.4167. Best2500 DEV Tip31.25%,knownUnique.7778. This proves fit for31 observed TRAIN parents, not for64; four times training exposure and24 versus6 DEV selection opportunities are explicit. All12 DEV plots and original pools are retained. No real-Qwen online timing was repeated for6000; never copy1500 latency into a measured6000 result.
- The registered first16TRAIN cross-goal association screen FAILED its decisive gate: collision-minus-clear excess.7633cm <2cm. Stop this particular auxiliary trial without relaxing the threshold or searching DEV for another gate. Stable same-type reference crossings alone do not establish a method opportunity or novelty.
- Subsequent root decision: implement a separate fresh prefix76/12000 ordinary control, leaving prefix44/6000 byte-identical. First1500 must strictly reproduce original64 last state before an independently launched finish; all189 actual TRAIN inputs get fixedlast diagnosis.12000x32 gives384000 draws/1536000pathstates, about2031.75 draws/input versus2064.52 for32/6000; approximately equal per-input exposure, not equal total budget.48versus24 DEV selection opportunities must be explicit. Authorized implementation only; no12000 run has launched.
- Same-data original observed A*32/64 controls completed:Tip96/144=66.67% and100/144=69.44%; each retains4 finite wrong-goal slots plus44/40 ungenerated slots. Classified duplicate means1.8056/1.9444 remain. These are different-backbone/time-budget strong controls, not a learned-method advantage.
- Next implementation only: a standard spatial-path-penalty A* control, separate new module while old sources remain unchanged. KeepK4, original observation proxy/endpoints,20k-node/2s search limits, penalty4; Gaussian scale2voxels=.05m. First fixed first4TRAIN parents/12inputs old/new feasibility, then fixed36DEV only after review. No acceptance labels enter generation. This control is not yet run and is not a novelty claim.
- Latest mechanical JOBS snapshot20:45UTC has451records; registry463. Formal116 parents98/99 (CALIBRATION) are running onCPU2/3; no locked raw/metrics opened. No GPU job remains. Inspect fresh actual PID/status before any launch; do not replay completed controls. Core paper criteria remain unmet.

## Current checkpoint — 20:25 UTC / 2026-10-03 04:25 China

- Actual9e0094a fixed64TRAIN preparation and ordinary1500x32/K4 training COMPLETE; original35% GPU1/CPU0 limit. All64 registered TRAIN parents closed before export. Actual63 parents/189 inputs/1108 positive references, one missing parent283220 unchanged. Capacity/H24 checks1108/1108; same36DEV identity/feature arrays and192000 exposure/init receipts retained. Source collector2626487 continues other registered roles onCPU2/3; no locked raw used.
- Prefix76 best500 DEV Tip35/144=24.31%,Any12/36=33.33%,knownUnique.6667;last1500 Tip26/144=18.06%,Any22/36=61.11%,knownUnique.1111. These remain mixed baseline results, not method evidence. Saved-pool diagnosis, fixedlastTRAIN and real online36 requests also completed exit0; full archival/analysis is underway, MAIN currently251rows still awaits these new rows.
- New bounded prefix44/6000 ordinary driver, first16TRAIN cross-goal opportunity audit and original A*32/64 scaling adapters are implemented and independently reviewed. Local combined46pass/1Torch skip before the final metadata-guard repair; server Torch validation is required before execution. No convergence, cross-goal audit or new A* generation has launched yet. Preserve first1500 exact-state gate and separate finish launch; do not interpret extra exposure as a method gain.

## Current checkpoint — 20:05 UTC / 2026-10-03 04:05 China

- Actual9e0094a fixed32 requested TRAIN /31 observed parents ordinary pipeline is COMPLETE:140 server tests, TRAIN556/556 endpoint-capacity/H24 references pass, 129 real Qwen encodings, same36DEV input/feature arrays and initial weights,1500x32/K4=192000 training states. Missing parent283220 retains3 unavailable inputs/27 unattempted slots. No filtered long/unknown positives.
- Prefix44 best1000 DEV Tip31/144=21.53%,Any24/36=66.67%,knownUnique.5;last1500 sameTip/Unique,Any17/36. Sixteen-parent best was22.92%/44.44%/.5833: higherAny alone is not overall quality improvement. Online all36actualQwen requests completed; original decisions reproduced, timing/cost in original receipts.
- Actual fixedlast TRAIN93:Tip134/372=36.02%,Any70/93,semantic304/372=81.72%,loss.00298756. Sixteen-parent last fit87.5% does NOT justify attributing all32-parent errors to generalization. Next ordinary convergence control is prospectively bounded fresh6000, gated on exact original1500 model/optimizer/RNG/sampler replay, then a separate finish. Implementation only; not launched.
- Fixed64TRAIN+same12DEV/1500-step scaling remains scheduled after all64TRAIN closures; do not change this exposure or data based on32results. Mechanical19:58:54 had51TRAIN+12DEV closed; JOBS19:59 has403records and only formal116 live onCPU2/3. All prefix44 GPU/CPU analysis jobs have completed. Never inspect locked raw or replay completed stages.
- MAIN_RESULTS251rows now includes best/last/online with costs deduplicated. First16TRAIN-only cross-goal positive-correspondence opportunity audit is under implementation; not a core method or verified causal explanation. Paper criteria remain unmet.

## Current checkpoint — 19:30 UTC / 2026-10-03 03:30 China

- Prefix28 ordinary/cache/online/CPU diagnosis are COMPLETE, not running. Actualce548 ordinary1500x32/K4: best500 DEV Tip22.92%, Any44.44%, Unique.5833; last1500 Tip22.92%, Any55.56%, Unique.3611. Both zero valid classified duplicates. Source743 online36real Qwen requests reproduce decisions, median73.146ms/p9580.546/first656.477, 4.284GB peak allocation. All12parent plots reviewed and hashes retained.
- Crucial actual743d9b2 fixed-last TRAIN48 diagnostic: Tip87.5%, Any100%, semantic96.35%, Unique1.4792, validunknown2.0208, duplicate0. Original best500 TRAIN24.48% must not be mistaken for last fit. Matched route ADE11.437→2.728cm and exact saturation loss.0050795→.00026637. Generalization, not inability to fit the training positives, is the current main gap.
- Actual743d9b2 original observed A*v2 control COMPLETE:10server tests;36inputs/144slots,76TipValid/68failed, Any52.78%, Unique.5278, classified duplicate1.3056. Eight unsupported instruction conditions and nine no-goal-attachment conditions retained. CPU median779.898ms. Source0d730 first guard failure5pass/1fail preserved; only verified LF/CRLF exact byte forms were allowed after proving same old planner bytes. No planner/threshold change.
- MAIN_RESULTS248rows retain four new DEV result rows; same training/checkpoint/cache costs explicitly deduplicated. Latest JOBS19:26:46 contains372records; only formal116 collection remains live, CPU2/3. GPU and CPU0/1 released. No locked raw/metrics opened.
- Next actual implementation fixes first32TRAIN+same12DEV (prefix44), then64TRAIN+same12DEV (prefix76), same seed0/1500x32/K4/ordinary recipe. Exact registered identities, failure retention, TRAIN capacity and unchanged DEV arrays must pass before each run. No core module or novelty claim yet; do not mistake scaling for mechanism evidence.

## Current checkpoint — 19:12 UTC / 2026-10-03 03:12 China

- Formal116 continues from2626487 on CPU2/3. First16 TRAIN and12 DEV are closed; no locked raw/metrics opened. TRAIN-only full432-slot analysis completed:285accepted/147failed,181known/104validunknown,14/48conditions have known R>4, all432strict restores pass. All48nine-slot plots reviewed; accepted lengths .574–5.895m remain unfiltered.
- Sourcece548 export completed19:10UTC:28parents/84actual inputs/490positive references. TRAIN-only endpoint capacity passes285/285 and modelH24 tip checks285/285; no reference deletion or threshold change. Actual83 related server tests passed, including full-state training recovery; source743 real-request timing14tests passed.
- Actual frozen-Qwen cache then ordinary K4 seed0/1500x32 pipeline is RUNNING fromce548f43ab22e804a8b70dea2f8bf297e20c8b84, SSH7878, runs/observed_two_row_prefix28_v1. CPU0, GPU1/35%. Read stages before any recovery; never replay the fresh-only whole launcher. This is the ordinary baseline, not a new mechanism.
- Same-data original observed A*v2 adapter has6localtests and independent source/label/budget review; server tests and actual evaluation are next. No A* result yet. After baseline completes, run sealed-pool diagnosis and actual Qwen online36-request timing; decide the next intervention from measured failure/coverage evidence. Core-paper criteria remain unmet.

## Current checkpoint — 18:33 UTC / 2026-10-03 02:33 China

- Formal116 is ACTUALLY RUNNING from immutable262648796fd47b25a2051cc227b6838515b651c6, session runs/observed_two_row_formal116_v1/sessions/20261002T183023Z_485548, outer PID485548/SSH85859. Server111tests passed7.88s and registration_prepare exit0. First0/1 workers485599/485598 started18:30:34UTC, affinity2/3, one thread each, GPUhidden.
- Corpus data/observed_two_row_formal116_v1 has once-frozen116parent plan; order16TRAIN →12DEV →48TRAIN →score/cal/locked,3132requested slots. Failure/unknown/no-input remain, no replacement/replay. Only mechanical receipts read across locked roles. Read actual session/parent statuses; initial running is not collection success.
- Next: first16 TRAIN all-slot quality/visual analysis, closed28-prefix export and TRAIN-only endpoint support/H24 audit, then ordinary actual-Qwen K4 1500x32 training. Driver under implementation; no new GPU training yet. Core mechanism remains unestablished.

## Current checkpoint — 18:10 UTC / 2026-10-03 02:10 China

- All previously launched jobs are complete; JOBS snapshot18:06:51UTC has310records and no live running recorded jobs. Old6c formal collector completed144/144closed at17:38:07UTC,exit0; only mechanical closure inspected for locked roles, never contents.
- Canonical-start v6 actual60a01ea:81tests passed;4/4 new static initial states pass full gates, initialization/restore planning entries0. Only283102 received27slots:23accepted,14knownlateral/9unknown,4failures. Target known counts5/7/2;27strict route restores pass.187.412s collector,187.881s process. Same DEV_COLLECTION geometry groups; new static start, not old dynamic replay or causal cross-version proof.
- Fixed legacy12 transfer actualfe564:13server tests;12parents/36references/48actual Qwen inputs;all12fixed checkpoints completed with original bytes unchanged. Best three-seed ADE8.9879→8.5621cm (2/3 improve);fixedlast8.9621→9.0961cm (onlyseed0 improves). Best endpoint11.5124→10.6636cm;last11.3679→10.8122cm. Six/12parent mean ADE improve. Keep limited conventional auxiliary evidence, no new seed/tuning or core-method claim.
- MAIN_RESULTS now244rows;12new CPU transfers incur zero training, shared actual encoding11.145s/0.003096GPUh counted once by shared_encoding_id. Saved candidate pools stay separate. Semantics/collision/execution/robot success null.
- Actual next implementation:116new physical parents (64TRAIN/12DEV_MODEL/12DEV_SCORE/12CAL/16TEST_LOCKED), same canonical q and narrow v5 geometric ranges, all27slots/no replacement. First16TRAIN+all12DEV precede remainingTRAIN. Two CPU1 workers permitted after source/registration/tests frozen. Collector is NOT YET RUNNING.
- Prepare closed-prefix export and ordinary peak/surface-anchor K4 saturation1500x32 baseline. First inspect TRAIN-only endpoint support/long-route quality; no unseen labels in forward. New two-row type evaluator preserves prior validity thresholds and actual reference classifier. No core mechanism starts before ordinary-model opportunity evidence.

## Current checkpoint — 17:35 UTC / 2026-10-03 01:35 China

This block supersedes earlier running statements; actual job receipts remain authoritative.

- Bounded Qwen SFT1500→6000 completed exit0 from5bb9087 at17:22:30UTC. Added4500 requests/11250 route slots,1777.327 seconds; original nine source artifacts unchanged. New best5750 NLL.416633 is not generation success.
- Exact registered continued TRAIN8 greedy probe completed exit0 from57e5231 at17:24:58UTC after14server tests. All8 calls formatted correctly, but0/8 endpoints within3cm. Mean23.391→27.769cm,3 improve/5 worsen. Stop this continuation/decoding branch; no DEV/K4/more-step expansion. GPU released.
- Four-layout v5 physical pilot from9324efc completed4 parents:108 requested slots,81 attempted,61 accepted,49 classified/12unknown,20 route failures,27 unattempted due to one setup collision. Six of12 requested target conditions have≥5known types. All81 strict restores pass. Four local perturbations are DEV_COLLECTION, not formal training/generalization evidence. Formal report/analysis and all-slot figures are archived.
- CPU source audit found framework reset validation performs additional planning calls. Historical83/683 counts cover collector-explicit calls only; total walltime includes framework costs, unknown internal counts are not reconstructed. One new static canonical-start gate on all4layouts and27slots only on formerly failed283102 is under implementation; not launched.
- Read-only old-batch eligibility: all12 old DEV closed, no exact/1mm/RGB-file duplicates against108 new TRAIN/DEV mechanical records, and no usage in62 indexed model configs/10input manifests/27per-scene reports. This supports a prospective frozen12-checkpoint transfer, not statistical independence/OOD/locked TEST. Export/gate preparation only; no transfer predictions yet.
- JOBS snapshot17:31:04UTC contains303 records. Read it and current PIDs before action. No locked images/routes/model metrics were opened. Core-paper A–H remains incomplete.

## Current checkpoint — 16:57 UTC / 2026-10-03 00:57 China

- GenuineK2 ordinary3000x64 completed exit0 from5bb9087,384000path slots. Both original static-best500 and last3000 DEV Valid100%/Unique1.9375. Fixed-best TRAIN closure AnyValid99.65278%;16 failed solvable changes across8parents all involve an originally invalid candidate, zero common-failure events from two originally valid distinct routes. This reinforces abandoning the joint-risk mechanism. Complete artifact/analysis report is being synchronized.
- Bounded Qwen continuation ACTUALLY RUNNING from5bb9087 since16:52:48UTC, record440253/child440254, runs/vlm_route_sft_continuation_v1/seed0. Actual13CPU recovery tests passed; latest inspectedstep2150. Original1500 source immutable; total limit6000, no post-limit continuation. Read actual checkpoint/status before resume.
- Four-layout/three-target physical pilot ACTUALLY RUNNING from9324efc, runs/data/observed_two_row_layout4_v5, SSH12430. Actual62server tests passed;108 route slots across4 once-sampled layouts, no replacement/retry. CPU1, GPU hidden. All outputs remain DEV_COLLECTION; no model-training authorization or claim follows from partial parents.
- Old6c formal collector mechanical check16:53:11UTC:125/144 closed markers (TRAIN85/96, remaining roles10/12 each), still running. No locked contents inspected. JOBS snapshot293records, registry updated; new84ee144-parent collection remains complete. Current process receipts supersede snapshots.

## Current checkpoint — 16:42 UTC / 2026-10-03 00:42 China

- Actual7b496e TRAIN768 closure audit completed:3806 changes,3424 solvable,720 evaluable parents. Geometric farthest/DPP K2 and fixed response oracle all reach100% conditional AnyValid; opportunity gap0. Stop the joint-risk module in this setting. K4 prefix2 is not a trainedK2 baseline; an actual K2 ordinary budget control is being registered separately.
- Actual7b496e physical v4 completed9 proposals/9 strict restores:8 valid,5 distinct lateral sequences,3 valid unknown,1 rejected H24 type instability. No planning or execution collision failure. Total62.996s; same single central-target parent only, not method evidence. Fixed-height four-layout/three-target pilot is being prepared; no post-hoc label rescue or height sweep.
- Qwen explicit1500-to6000 continuation source is frozen for server CPU recovery tests, not yet GPU launched. Original run unchanged; added11250 TRAIN route slots and original3750 are accounted separately. Real tiny-loop restore test must pass before launch. GPU queue: short genuineK2 then this bounded continuation. No extension beyond6000 authorized by the experiment card.
- Old formal collector remains the only previously running job at the latest mechanical check; inspect fresh process/closure receipts before action. Locked images/routes/metrics stay unopened. Core-paper A–H remains incomplete.

## Current checkpoint — 16:25 UTC / 2026-10-03 00:25 China

- Fixed2c47c2a greedy TRAIN8 and static six-configuration diagnostics both completed exit0. GPU is free. Greedy reduces mean endpoint57.372→23.391cm, all8 improve but0/8 pass3cm; no decoding sweep. An explicit same-objective continuation1500→6000 is under implementation, not running, and must preserve complete state in a fresh tree with original/added exposure separately counted.
- Static6 applied all saved configurations after two preserved API failures. All12 pre/post native-state checks pass;0 new IK/path/simulation-start calls. External hits are Panda_gripper against the registered posts in every configuration. Original dynamic instant is not reconstructed. One new independent physical pilot is being prepared:post height.16→.14m, central target9 proposals, unchanged strict route checks and all failures kept. It has not launched.
- Sole new set hypothesis under audit: joint failure under single-opening closures. Only registered TRAIN768 will be inspected. K4 prefix2 is explicitly zero-adaptation, reference-pool optimum/geometry-DPP are privileged diagnostic controls; a genuinely trainedK2 ordinary model is mandatory before a paired learned-risk experiment. No new mechanism advantage is claimed.
- Latest old-collector mechanical receipt16:17UTC:112/144 closed markers, coordinator285850 still running; closure is not success. New84ee144-parent collector remains complete. JOBS16:16 contains278 records, registry281; all locked sample contents/metrics stay unopened.

## Current checkpoint — 16:02 UTC / 2026-10-03 00:02 China

This block supersedes older running statements. Actual status receipts and source hashes remain authoritative.

- All six ordinary/event-supported head training jobs completed. Three-seed original ADE-selected best: ordinary ADE9.8548±0.1615cm versus auxiliary9.0827±0.3151; endpoint17.9283±0.4284 versus15.6893±0.8077cm. Fixed1500 ADE improves only2/3 seeds, and cup fixed-step ADE worsens in all three. Keep auxiliary as a stronger conventional baseline, not the paper mechanism. Source3f9cea5 replication jobs all exit0; actual paired initialization/RNG and48000-draw chains equal. Full evidence: OBSERVATION_MULTITASK_LANDMARK_THREE_SEED.md. GPU released.
- Source1915de7 teacher audit completed exactly9 forwards. Units, label shift/mask and causal prefix tests passed; coordinates account for97.04% NLL. Conditional endpoints cannot establish visual grounding because true earlier path points expose target direction. Only one next control selected: same-checkpoint grammar-greedy TRAIN8, not yet executed; freeze/test source first.
- Source1915de7 static collision attempt failed before any of six configurations was applied, because pinned PyRep mishandles the legal root-parent sentinel. All six remain unattempted. Minimal validated hierarchy fix is ready; only a new immutable release/fresh output may retry. No layout change or physical conclusion from the failure.
- New84ee7a6 formal collector is complete exit0 at15:10:13UTC:144 parent records closed; closure is not success. Old6c44469 collector status must be read before any action. Locked contents/model metrics remain unopened.
- Seed0 first-close spatial analysis from saved predictions completed: macro14.152→13.174cm, but lift/cup worsen; event sequence100% does not certify contact position. All12-parent paired saved-prediction visualization is being prepared, with identical axes and no outcome-selected panels.
- Paper-core A–H remains incomplete: actual Qwen and multi-task data are present, but general route-set mechanism, representative success/coverage and independent confirmation are not yet established.

## Current checkpoint — 15:10 UTC

15:22UTC measured addendum: grammar TRAIN8 completed exit0,8/8 strict format but0/8 target within3cm,mean endpoint57.372cm; GPU released and no DEV/K4 expansion. Twenty-four endpoint IK queries completed,48 exact restores; both orientations3/6 collision-aware versus6/6 ignore, additional configurations collide. No new route references. Ordinary auxiliary37tests and actual prefix108 initial-forward/48000-draw sampler audit passed fromaef3983; launcher55c2c4a is reviewed for the single1500-step run. Read its actual status before claiming completion. JOBS snapshot15:20:40 contains242 records (registry245), showing new84ee formal collector no longer running; only mechanical closure/exit may be inspected for locked roles.9-forward TRAIN SFT token diagnostic and six-configuration static collision comparison are under implementation, not launched. Full SFT plots v2 all48panels/source hashes unchanged; all8pages visually checked.

This block supersedes the historical running statements below. Inspect actual job receipts before any restart.

- Full same-checkpoint Qwen SFT generation and independent analysis completed from8356b09. Both independent4 and whole4 have zero TipValid/semantic success across all24 DEV instructions. Strict H24-format slots21/96 and5/96; request medians41.123/41.211s. All120 requests, failures, raw text, component timing and binary hashes are retained. Valid JSON with wrong horizons also has no endpoint within3cm; no parsing repair changes the main evaluation. Reports/VLM_ROUTE_SFT_AUTOREGRESSIVE_RESULTS_V1.md is authoritative.
- Prefix108 ordinary head completed from6464b3:96 requested TRAIN parents/95 positive,12 unchanged DEV;432 real Qwen features,1500x32/K4, best750 DEV macro ADE10.000cm/endpoint17.434cm versus prefix60 12.944/19.700. Best and last retained;cup/lid ADE worsened and reach/slide endpoint errors exceed30cm. Same DEV raw/cache hashes and reconstructed initialization/RNG verified. One training seed; expansion changes color/variation coverage as well as parent count. This is stronger baseline/data evidence, not core novelty.
- Depth TRAIN8 actual processor audit completed from98f818e after retaining the6464 raw-configuration-hash failure. It finds byte-coded depth interpolation distortion; no causal attribution of SFT endpoint failure is established.
- Fixed release0eeeecbe3d02adac413083a9edc0df674aa53be2 is pushed and deployed. CPU20 grammar tests and actual tokenizer preflight passed. Only eight TRAIN target0 K1 calls are now running at runs/vlm_route_grammar_v1/train8_k1, SSH95364 / child391988. No automatic full DEV expansion. GPU1/35%,CPU1.
- Two-row strict reconstruction v3 failed before any proposal:27 unattempted slots preserved, no tolerance relaxation. New endpoint IK24 diagnostic from0eeeecb uses a fresh common in-process snapshot, same128 search trials per query, two orientations and two collision filters. SSH76851 CPU1; actual status determines outcome. No generated route or robot benchmark claim.
- Two formal collectors continue in their existing releases. At15:05UTC mechanical closure was139/144 new and89/144 old; this is not a success count. Locked contents/model metrics remain unopened. New event-supported attention target is a conventional baseline repair under preparation, not yet trained. Paper-core A–H remains incomplete.

## Current checkpoint — 14:17 UTC

This block supersedes the older in-progress statements below. Actual job records remain authoritative.

- Completed real Qwen K1/K4 LoRA SFT: source ba984062b6c672bb7e4c2572b2112994bbbd0c95, runs/vlm_route_sft_v1/seed0, 1500 steps, 3750 supervised route slots, 1273081 answer tokens. Best DEV token NLL .4367267, 556.787 seconds, 4.834GB peak allocation. All eight adapters updated. Teacher-forced NLL is not route-generation quality. The earlier missing-pytest runtime failure is retained; actual tests passed in .venv before .venv-qwen training.
- Actual autoregressive comparison is RUNNING from 8356b09b8436d00a4f98a76ffdd8c7d84033e6ca, runs/vlm_route_sft_autoregressive_v1/best_seed0, child PID362818 / SSH session70700. Same best checkpoint, all24 DEV instructions including the no-reference case, independent4 versus whole4, one sampling repeat, no retries/repairs/pooling. Seven fixed-source CPU tests passed. Independent CPU geometry analysis follows generation in the frozen launcher. Only GPU1/35%, CPU1.
- Prefix60 sealed snapshot and cached ordinary-head pipeline completed from2f7b3e9:48TRAIN+same12DEV parents,180/180 collected positive references,240 observed inputs. Same1500x32 training exposure as prefix24; best/last1500 macro DEV ADE12.944cm and endpoint19.700cm versus prefix24 best17.121/24.498cm. This is data-scaling development evidence, not a new method or independent test. All48 DEV records,60 raw DEV input/reference file hashes and48 Qwen cache hashes match;024ba7 CPU source/seed initialization reconstruction passed, but no historical initial checkpoint exists. Expansion also adds seen task variations/colors; DEV uses held-out variations, so this is not a pure parent-count effect.
- Two-row observed pilot v1 completed18/27 accepted references, with at most3 distinct lateral types per target. Explicit row-plane guides v2 completed4/27, but initial arm joints differed by2.88969rad across versions despite near-equal tip poses; the comparison is confounded. Do not attribute the drop to guides. A strict reconstruction/readback gate is being prepared before a new attempt; no tolerance relaxation or new simulator run is authorized implicitly by this prose.
- Two formal six-task collectors continue from6c44469 and84ee7a6. Read current statuses before any resume. Locked-content/model metrics remain unopened. Paper-core mechanism and complete robot-task success evidence remain unestablished.

- Branch: `codex/multiroute-v2`.
- Historical snapshot: `d0d97eb22ff1190f4f89d4d2fd7100db116c8a57`.
- Historical baseline: learned-query set regression TEST UniqueValid@3=3.000; historical OOD=2.878 mean across seeds 0/1/2. Controlled true geometry + frozen CLIP only.
- V2 controlled development seed0/K4: subset matching UniqueValid=1.7734, Valid=52.54%; all-positive matching=3.2891/89.65%; saturation-aware assignment=3.3984/100.00% (best step1500 of3000). These are stronger baselines, not a novelty or robotics claim.
- Completion seed0/1/2 finished: coverage-minus-attention mean additional valid types +0.16406 / +0.01563 / -0.04688. The initial advantage is not stable; retain max pooling as an ablation, not the paper contribution. Full evidence: reports/COMPLETION_THREE_SEED.md.
- Self-draft mixture completed: attention own2+2 UniqueValid2.5078→3.3047, coverage2.9531→3.2813. Both remain below same-checkpoint joint4 (3.3125/3.3438) and use two forwards. Keep joint4; known training repair is not a novel contribution.
- Genuine frozen Qwen revision89644892: 96 RGB-language features; RLBench-derived reach32 has32 parents,275 accepted paths/288 attempts, strict restored state/RGB. Four original task collection checks12/12; this is collection success, not learned execution.
- Observation evaluation v2 fixes omitted no-reference instructions: all24 DEV instructions for semantics,23 for reference ADE. All five old cached/online checkpoints re-evaluated; thresholds unchanged and original reports retained.
- RGB-D ordinary baseline and endpoint-attention auxiliary each completed three real seeds at identical1000 steps/128000 slots. Strict3cm semantic accuracy mean1.04%→28.82%; AnySemantic1.39%→31.94%; endpoint18.92→14.51cm; all three seeds improve. Conventional grounding repair,8 DEV parents only; full-path validity and core mechanism still unproved. See OBSERVATION_GROUNDING_THREE_SEED.md.
- Online frozen/LoRA pilot and matched warm-start finished. Eight LoRA tensors genuinely updated; warm-start both select common step0, no DEV improvement. Do not equate adapter updates with effective fine tuning.
- Obstacle four-parent rendered-state pilot completed:48 proposals,20 successful paths,17 classified paths,48/48 exact restore. All four depth/physical surface audits pass within2.4mm after explicit render warmup. Preserve old failures. Formal128 obstacle and256 natural-layout parents are being collected using immutable source; see JOBS and agent job records for current state, not this prose alone.
- New32 TRAIN layouts + reused old8 DEV: six plain/aux training runs completed, strict semantics0→28.82% and reference endpoint18.54→12.19cm mean. Same128000 slots; new training pool, not a new test set. New64 seed0 pair also completed: strict0→19.79%, reference endpoint15.70→10.50cm; fixed exposure means fewer passes over the larger pool, not a monotonic accuracy claim.
- Obstacle16 TRAIN+4 DEV snapshot:90 training paths,47/48 TRAIN instructions with references; null-reference retained. Plain/aux seed0 completed, DEV TipValid2.08%→4.17%, AnyTipValid both8.33%; endpoint precision and collisions still poor. Box-only tip check does not certify arm, table, or execution.
- Localized route update v1 and matched4000-step continuation both completed. At4000 local-minus-full_free UnionUnique=-0.06537/-0.11800 for b1/b2, with lower new-route validity. The proposed gate is discarded; retain full_free, stop this branch of extra training, preserve all failures. See CONSTRAINT_UPDATE_V2_RESULTS.md. Full optimizer/RNG/sampler restoration and source immutability were tested.
- A TRAIN-only exact-instruction RGB-D color prototype reaches19/24,20/24,22/24 with old24/new32/new64 TRAIN parents, but has severe background outliers. Obstacle32 prototype reaches12/12 DEV endpoints. These task-specific endpoint-only controls do not generate paths or establish open-vocabulary planning; same-data comparisons must respect each training pool.
- Obstacle fixed-TRAIN gradient diagnostic at best250 and last1000 found positive interior-versus-endpoint+grounding cosine, so no claimed gradient conflict and no stop-gradient repair added.
- Online RGB-D frozen/LoRA1000×4 pair completed from9c19288: both original ADE-selected best checkpoints are the common step0 (strict19.79%, ADE5.74cm). Last semantics16.67%/12.50%; real adapter updates do not establish useful fine tuning. Scalar-hash preflight failure retained. See OBSERVATION_ONLINE_RGBD_PAIR.md.
- Peak observed-point anchor with straight-through soft gradients completed THREE paired seeds in natural64 and obstacle32, same initialization/data/128000 slots per arm. Original ADE-selected natural semantics17.36±13.19%→83.68±6.62%, ADE6.47→4.55cm; all seeds positive. Obstacle semantics19.44±4.81%→28.47±34.38%, ADE18.38→19.30cm: unstable with worse reference geometry. This is a conventional grounding baseline repair, not core novelty. All best/last artifacts preserved.
- Obstacle32 seed0 original best checks: soft TipValid4/48→peak8/48, AnyTip2/12→3/12, TipClear17/48→40/48. Box-only tip checks exclude full arm, table, unknown geometry and execution. Uniform fixed-step1000 evaluation of BOTH methods/ALL seeds is a separately labeled exploratory sensitivity check; never substitute favorable last outputs for original best results.
- Traditional observed A* v1 failed all48 DEV slots; preserve it. TRAIN-only attachment diagnosis led to v2 virtual endpoints without relaxing geometry. Fixed a32 v2 completed all12 DEV instructions:44/48 TipValid, UniqueClassified1.333, known-reference coverage26.67%, median1.969s. Four failed slots remain NaN. This is now a functioning closed-instruction observed planner control, not full-robot certification.
- Prospective observation roles registered2026-10-02T10:10:31Z in configs/observation_partition_reservation_v1.json. Raw collectors unchanged; no locked-parent content/model metrics opened. Future exports must apply exact parent roles and preserve failures. OOD remains uncollected.
- Natural256 collection completedexit0:256 parents,2304 attempts,2281 successful references,23 failures,1 near-duplicate; all2304 strict restore checks passed. Reserved export192TRAIN+16freshDEV contains624 observations, no failed setup and all inputs have positive references. Input SHAaa17ecef147f73ada2e902ac30f7de39be0acc47a25ced5959607619bf945495; true Qwen encoding is separately recorded, never infer cache completion from this line.
- Both3000-step epsilon and matched v diffusion pairs completedexit0 with identical actual stream hashes. K4 epsilon independent Valid26.37%/Unique.8281; set18.16%/.6302; v independent24.02%/.7839, set24.61%/.7969. Stable v parameterization does not resolve collision failures. One bounded continuation to12000 is being prepared with explicit extra exposure; no PG sweep. Three sampling repeats are not training seeds.
- All six unchanged old64 best models completed transfer to fresh16DEV: soft/peak semantic28.47±3.18%/72.05±3.14%, ADE6.60/6.21cm. A fixed old64 color prototype reaches40/48 endpoints but has severe outliers; no refit. New192 paired fresh training completed3000×32: original ADE-best soft66.67%/peak95.83% semantic, ADE2.872/2.649cm. Peak last3000 falls to87.5%; preserve both. New192 has three times old64 data AND training exposure, not an equal-total-budget learning curve.
- New task feasibility job `observation_task_expansion_v2/four_new_tasks` uses3b597f4, strict rendered restore, CPU1 and GPU hidden. Tasks: pick_up_cup, slide_block_to_target, insert_onto_square_peg, stack_blocks. DEV_COLLECTION only, no route-type or trained model claim. Read actual status before any retry; no automatic resume of partially collected parents.
- Resources: see RESOURCE_ENVELOPE.md. Keep sequential GPU1 queue.
- Git remote authorized by user: `git@github.com:Mayyoungyoung/dp_vl_1.git`; research branch `codex/multiroute-v2`. Immutable release trees and every training source commit are recorded with jobs. Verify actual remote head rather than trusting a stale prose commit.
- Follow current logs in RESEARCH_LOG.md and job records in JOBS.json. Final-paper core evidence incomplete.

## Latest audited decisions — 2026-10-02 12:00 UTC

This block supersedes earlier in-progress collection/continuation statements above.

- Both natural256 and obstacle128 collections are complete. Obstacle128:1536 attempts,746 successful references,493 classified references,1536 exact restores,0 setup failures;9102.59 seconds. These are collection counts, not model outcomes.
- Reserved obstacle export contains96 TRAIN/8 fresh DEV parents,312 observations,285/23 positive-reference instructions;3 TRAIN and1 DEV zero-reference inputs remain. True Qwen cache312/312 hashes verified,22.952 seconds including load. Locked roles were not opened for development.
- v diffusion bounded continuation completed12000 steps per arm,3072000 target slots each: independent Unique1.21354/Valid40.17%, set1.09375/33.72%. Both improved but remain far below regression despite4x exposure; stop this branch, without claiming convergence. Original3k checkpoints and all stream digests unchanged.
- TRAIN obstacle references pass the same2cm tip check181/181 before and after H24 resampling. Most peak-last first-intersecting segment starts lie in the first25% of path length; this is a lower bound, not exact contact progress. Investigate local geometry, not relabel references.
- New96 obstacle soft/peak pair launched from6c44469,3000x32,K4,fresh seed0; prospectively both select DEV by UniqueClassifiedTipValidAtK +0.05*TipValidAtK.14 fixed-source tests passed. All24 DEV instructions retained. Prior ADE selections remain historical evidence.
- New96 observed A*v2 completed exit0 from6c44469 on all24 fresh DEV instructions. Analyze its saved report with the paired neural results; do not infer superiority from completion alone.
- Six-task collection exact cross-process resume passed: state,inventory,RGB,depth,pose,open,camera,language all identical; first slot hash unchanged;3/3 original-task successes. Formal continuation `runs/observation_multitask_six_formal_v1` is running CPU1 from6c44469 over the registered144 parents,432 requested attempts. Actual closure counts remain authoritative. Source and launcher fixed; do not duplicate.
- Next candidate is only a TRAIN audit of whether observation-derived early exit regions distinguish known routes. No allocation module is implemented; connected boundaries withJ=1 would falsify this representation. Conventional local geometry reading remains a possible baseline repair, not established novelty.

### Subsequent measured update — 12:18 UTC

- New96 pair completed: soft best/last3000 TipValid44.79%,Unique.75; peak best1250 52.08%,1.125; peak last3000 43.75%,.875 despite better semantic endpoints. Actual same96 A*v2 is87.5%,1.25. One training seed, no new core claim. All24 fresh DEV and1 zero-reference instruction retained.
- True online Qwen requests from90535b3 completed24 per model: median61.73/77.20ms including reads/processor/Qwen/RGB-D/head/output; excludes score/checker/execution. Online/cached path differences<=1.431e-6m. Model loading and first request recorded separately.
- TRAIN region audit completed9.35s: all72 instruction/radius pairs J1, no budget exhaustion/failure. Reject the connected-boundary allocation representation; do not create artificial fixed experts.
- Implementing only matched local versus global pooling for one draft update, same4995 extra parameters and fixed TRAIN scale audit. Four complete drafts plus four updated states will be explicitly recorded; not equal raw-path budget to one-passK4. No new GPU training launched yet.
- Six-task formal6c continues; independent camera-off performance audits found one non-equivalent push_button trajectory despite identical initial observation and original task success. Retain formal cameras-on source; diagnose with a separately budgeted on/on repeat, do not silently adopt.

- Matched local/global one-update experiment now actually running from5501c5f, `runs/observed_refinement_reserved96_v1`, global then local, fresh3000x32seed0.25 fixed-source tests passed. Gaussian sigma.10m, first.50 draft arc, coordinatewise bound.10m locked from TRAIN audit before launch;4draft+4final states per request and768000 training states per arm recorded. GPU1/CPU1, no extra geometry labels in forward. Compare like-for-like with global update; original peak/A*4 remain lower-state-count references.
- Push_button on/on repeat also differs fromstep2 despite exact initial inputs; the on/off difference cannot be causally attributed to rendering from this probe.13 extra attempts retained. A separately registered new-seed corpus is being prepared with camera-off only for five tasks that passed exact audit and push_button kept on; not launched yet. Mechanical layout duplicate gate is required before cross-role model use.

### Subsequent measured update — 12:48 UTC

- Both5501 refinement arms completed exit0. Global/local best TipValid50.00/51.04%,Unique1.0833/1.1667; last3000 TipValid52.08/43.75%,Unique1.0/.875. Local costs201.80s versus145.07s and does not establish a stable quality/coverage advantage. All drafts/finals/checkpoints retained; independent draft-to-final audit is being frozen, not yet completed.
- New six-task source84ee7a6 is deployed with separately preregistered seed282000 and mechanical cross-role leakage gate. Only a first-TRAIN-parent cross-process proof is authorized so far; full new collection has not started. Original6c formal collection continues unchanged.
- Future resume guard now rejects changed eval_every as well as selection criterion. This cannot alter the completed fresh5501 pair. Full source pytest will run from the next frozen release; earlier model-loop resume tests were not full GPU CLI recovery tests.

- d305 independent refinement audit completed:11 tests and all3 analysis jobs exit0. Local best repairs1/breaks0 candidates, last repairs0/breaks3. No bound saturation; abandon this local update, retain ordinary peak/A* controls. Main table now209 measured rows with both best/last and explicit8-state budgets; static draft/final figure visually checked.
- New84ee7a6 first-parent proof passed exact process restoration,3/3 original-task successes and all camera flags restored,46.397s. Formal `runs/observation_multitask_validated_five_formal_v1` is ACTUALLY running, PID316991, session86465; old6c PID285850/session42864 also running. Both CPU1, no model GPU jobs active at this checkpoint. Inspect JOBS and actual children before resuming.
- Next actual work: closure/gate snapshot and ordinary free-endpoint multi-task head, preserving null task-validity labels; independent TRAIN diagnosis of predicted valid duplicates for the next set mechanism. No new mechanism or paper-core success is established.

### Subsequent measured update — 13:27 UTC

- Prefix24 snapshot is sealed from5fb74b:24 requested parents,72/72 successful references,96 language inputs; TRAIN12 parents/36 references. All36 TRAIN event sequences survive H24; cup6/6 and lid6/6 endpoints exceed the hard-point anchor's coordinatewise5cm capacity. Free endpoint is an ordinary representation repair, not a new mechanism. Snapshot provenance: reports/observation_multitask_prefix24_v1.
- Actual Qwen direct-set SFT preflight from5fb74b failed backward with OOM at the unchanged35% GPU1 cap. Exact processor prefix masking passed, but no completed SFT is claimed. Preserve reports/vlm_route_sft_preflight_v1; implement equivalent answer-only chunked vocabulary loss and rerun in a fresh job.
- TRAIN96 stored-prediction audit:560/560 positive H24 references tip-valid;393 valid classified duplicate predictions, but only28/288 instructions have both a duplicate and a known missing reference type. Label-assisted replacement opportunity30 total is not an achievable model gain or a bound on unknown modes. Do not penalize acceptable duplicates where known modes are fewer thanK.

- Actual6bc2b8 Qwen SFT preflight v2 now completed:7 tests,4 optimizer steps,8 updated adapter tensors,4.839GB peak allocation; original OOM retained. Formal resumable variable-K SFT is being implemented, not yet launched.
- Actual6bc2b8 six-task ordinary baseline completed1500x32,192000 path states. Cache96/96 in8.675s including load; head70.058s/.01946GPUh,950.86MiB. Best500 TRAIN macro ADE1.783cm versus DEV17.121cm, DEV endpoint24.498cm; final1500 ADE17.216cm. Event sequence100% is reference order only, no path/robot success. Both checkpoints/predictions preserved in reports/observed_multitask_prefix24_v1 artifact index. Add nearest-reference retrieval control and prospectively enlarge TRAIN, rather than repeat this tiny-data training for an apparent seed gain.
