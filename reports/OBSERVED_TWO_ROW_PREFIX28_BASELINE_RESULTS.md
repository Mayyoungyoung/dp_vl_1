# 两排观测任务：首28父场景普通基线真实结果

2026-10-02 19:15 UTC完成。新数据上的普通冻结Qwen＋RGB-D集合回归已经真实训练、独立复算，并在全部36条DEV指令上完成无缓存在线推理。结果仍弱：固定best与last均只有33/144条候选通过tip任务检查。当前主要缺口是目标与几何质量，不是大量有效重复；没有建立核心机制优势。

## 数据、模型与协议

正式注册前缀固定16 TRAIN父（283200–283215）与12 DEV_MODEL父（283264–283275），每父同一RGB-D/当前状态对应3个目标指令。84个实际输入全部存在，756次采集提案得到490条已接受正参考（TRAIN285、DEV205），失败266次不删。无替换父场景、无跨父拆分、无锁定角色原始内容读取。场景是两排四柱的窄范围同分布RLBench-derived变化，不能称原RLBench benchmark或宽泛任务泛化。

训练前仅TRAIN做容量检查：285/285参考的终点都在某个可见stride2点的逐坐标0.05m残差范围内，285/285重采样H24通过原tip检查。这个检查只说明表示可容纳已知正例，不说明模型已经学会目标定位或绕障。原始长路线保留，未按DEV结果删参考或改阈值。出口和门禁证据见[准备记录](observed_two_row_prefix28_preparation_v1/export_manifest.json)及[TRAIN审计](observed_two_row_prefix28_preparation_v1/train_quality.json)。

真实Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`，冻结BF16骨干，RGB＋指令mean/last共4096维；当前RGB-D＋相机＋gripper state进入普通点编码器与K4集合头。H24、128宽/2层、point64、stride2、peak表面anchor、逐坐标残差界0.05m；saturation正参考匹配，端点注意力辅助权重0.02。没有模式标签、真目标或箱体进入前向，没有新方法模块。seed0，1500步×batch32×K4＝192000训练路径槽，lr3e-4。

每250步在DEV以 `UniqueClassifiedTipValidAtK + .05 * TipValidAtK` 选择best；真实best是step500，last固定step1500。报告完整1500步成本，不按best500折扣。起点5mm、目标身份且3cm、恒定事件、箱体tip线段2cm阈值保持；没有学习评分器，SelectedValid保持null。TipValid不认证机械臂体积、IK、控制执行或机器人成功。

## 主结果：全部候选均保留

| 条件 | 候选槽 | TipValid@4 | AnyTipValid@4 | UniqueClassifiedTipValid@4 | 已知正参考类型覆盖 | 目标正确率 | TipClear@4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV，best500 | 144 | 22.92%（33） | 44.44%（16/36） | 0.5833 | 1.6204% | 47.22% | 42.36% |
| DEV，last1500 | 144 | 22.92%（33） | 55.56%（20/36） | 0.3611 | 3.7500% | 64.58% | 33.33% |
| TRAIN，best500 | 192 | 24.48%（47） | 47.92%（23/48） | 0.6042 | 0.4255% | 54.17% | 44.27% |

best/last DEV已知类型覆盖分母都是36条件；TRAIN只有47/48条件有已知类型，另1条件只有未知类型正参考。Coverage只对已知正参考类型计算，不把新生成的有效类型判负，不把参考数当解的总数。所有预测xyz/事件均有限，所有起点和恒定reach事件正确；这两个简单条件没有掩盖目标或碰撞失败。

DEV candidate-matched ADE分别11.893/12.041cm，参考末端距离11.194/6.923cm（best/last）；TRAIN best为10.606/9.618cm。这些是参考重建误差，不能替代实际目标语义与碰撞检查。结果和逐候选证据在[训练summary](observed_two_row_prefix28_v1/training/peak_seed0/summary.json)、[best逐场景](observed_two_row_prefix28_v1/training/peak_seed0/dev_model/per_scene.json)、[last逐场景](observed_two_row_prefix28_v1/training/peak_seed0/last_dev_model/per_scene.json)、[独立复算](observed_two_row_prefix28_v1/diagnosis/analysis/analysis.json)。

## 失败来源与下一步判断

best DEV中76/144候选目标错误，83/144碰撞，两项重叠48；另35条目标正确却碰撞，28条几何clear却目标错误。last目标错误降至51条，但碰撞增至96条，目标正确仍碰撞增至60条，所以有效数没有增加。TRAIN best也有88/192目标错误和107/192碰撞。这里的TRAIN评价对应best500，不是last1500；不能用它推断last训练拟合度或过拟合程度，固定last的TRAIN/DEV诊断需单独报告。训练loss从250步0.10376降至1500步0.07507，几何/覆盖指标不单调；不能只凭下降loss继续延长训练。

best DEV的33条有效候选中，21条已分类、12条类型unknown；last为13条已分类、20条unknown。两者**有效已分类重复槽均为0**，每个DEV条件的“用已知缺失类型替换有效重复”的计数上限也都是0。best TRAIN有4个有效重复槽，但只有3/48条件同时缺已知类型，最多3个槽可按该诊断界替换；相对47有效/192总槽，这不是主要瓶颈。不要依据这个结果启动去重或DPP式模块。

best有效已分类路线集中于`over→middle`（13条）及`over→positive_y`（8条），只有2条落在该条件已有正参考类型中；TRAIN已分类有效33条也都先`over`，仅1条匹配已知参考类型。低ReferenceCoverage与未采到的新有效类型并存，不能把这些路径一律判错。跨目标已知共同类型一致性在best DEV/TRAIN均无可评价对；last DEV仅2个可评价有向对，且都未保留。样本不足以支持跨目标迁移机制。

当前证据支持先区分目标定位错误和已定位路径的碰撞来源，并等待相同数据的传统观测规划对照；本报告不推断某个新模块一定有效，不把正常基线修复命名为创新。A*尚无可纳入本报告的结果，未混入其指标或成本。

## 真实在线K4与缓存一致性

固定同一best500，在全部36条件各真实重新读取RGB-D、Qwen processor/编码、点几何与集合头；K4预测先写NPZ并哈希封存，再读取本条标签检查。36次实际请求/144槽，零生成异常、零缺输入、零重试、零修复、零隐藏候选；没有额外热身。last没有再跑在线模型。

连续单请求（含输入I/O及hash、Qwen、几何、head、输出封存、标签I/O及两排检查）中位数**73.1456ms**，p95 **80.5455ms**，首请求**656.4770ms**。首请求保留在全部统计中；中位数不是冷启动保证。模型加载7.0311s、前置完整性/元数据5.9539s；36请求串行段3.2404s，整个pipeline16.5245s。操作系统文件页可能已由前置hash预热。峰值CUDA allocation 4,283,784,192字节（约3.99GiB），GPU1/35%、CPU1。

| 单请求阶段 | 36请求中位数 |
|---|---:|
| 输入读取与hash | 3.559ms |
| RGB-D反投影/传输 | 2.243ms |
| Qwen processor/传输 | 3.889ms |
| 实际Qwen forward/pooling | 50.980ms |
| 学习几何编码器 | 1.490ms |
| 路线头/输出传输/计时开销 | 1.616ms |
| 预测NPZ与SHA封存 | 2.660ms |
| 标签读取/hash及两排检查整体 | 4.103ms |

各阶段中位数不能直接相加当中位总时延。原训练summary的3.502ms缓存头和7.902ms RGB-D＋头均不含Qwen和验收，不能作为端到端数字。

全部36个在线mean/last特征与缓存数值完全相同，token数一致；paths相对原batch缓存评价最大绝对差仅1.1921e-7m，全部逐候选语义/起点/碰撞/事件/类型决定相同。在线指标因此与best缓存指标一致。这个一致性分析只比较保存数组，不追加前向。见[在线summary](observed_two_row_prefix28_v1/online/best/summary.json)、[逐请求连续计时](observed_two_row_prefix28_v1/online/best/requests.jsonl)、[缓存对照](observed_two_row_prefix28_v1/online/best/cache_comparison.json)。

## 成本、失败与复现

| 阶段 | 实际工作 | 内部计时 | record_job进程墙钟 |
|---|---|---:|---:|
| 冻结Qwen cache | 84次真实图文编码 | 9.5543s（含加载3.9675s） | 15.4956s |
| 普通head训练＋开发/最终评价 | 1500步，192000训练槽 | 64.7004s，0.0179723 GPUh | 69.8183s |
| fixed-best在线 | 36请求/144槽 | 16.5245s，0.0045901 GPUh | 17.7168s |
| 独立CPU分析＋12父绘图 | 保存池复算，无生成 | 见analysis JSON | 23.6107s |

三个GPU阶段内部合计90.7791s（0.0252164h），记录进程墙钟合计103.0307s（0.0286196h），二者是不同口径，不能相加。采集与前期数据验证成本独立保留，不包含在这三段GPU成本中；cache不再次计入训练步骤。训练另有336评价条件/1344路径状态，24次head/几何延迟诊断/96状态，同图语言控制只重索引已有预测，额外前向0。head约123万参数，训练峰值951.37MiB；不把冻结Qwen权重排除于在线显存。

本轮cache、训练、独立分析和在线作业全部exit0。准备阶段保留一次错误测试文件名导致的exit4（0.1761s，未执行测试/训练），纠正文件列表后83项真实测试通过（pytest16.05s，进程16.9145s）。在线源码另14项服务器测试通过0.14s。出口进程2.7188s，TRAIN容量审计进程3.7659s（审计体1.4608s）。失败和更正记录在[准备目录](observed_two_row_prefix28_preparation_v1/)；没有省略失败或据此重抽数据。

实际训练/分析release：`ce548f43ab22e804a8b70dea2f8bf297e20c8b84`；在线release：`743175e84eaba4031ce527cdbe1ea3bffcc8fcb9`。准确argv、cwd、PID、起止UTC和退出码分别在[训练status](observed_two_row_prefix28_v1/training/peak_seed0.status.json)、[cache status](observed_two_row_prefix28_v1/training/qwen_cache.status.json)、[分析status](observed_two_row_prefix28_v1/diagnosis/analysis.status.json)、[在线status](observed_two_row_prefix28_v1/online/best.status.json)。重现应使用相应不可变release及新输出；不要重启已完成launcher覆盖原结果。

权重仍在服务器 `/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_prefix28_v1/peak_seed0/`，未进入普通Git：

- `best.pt` SHA256 `cf05c812a1cb3e9f9039e14f97d72ad6560deae207d8bfaa4eeeeb74e4fbf936`。
- `last.pt` SHA256 `7020190139bc712a98cb463d3b9f6f45255cb1e5354f4028fbde76aa6c44ef85`。
- best缓存预测 SHA256 `6103ceebb8f01f911c08fcd9464307ca4cfd2d7f4066e3f479467608b0d41597`；在线预测 `738b418a5393644cad89dc469b5395cb08e2b5271f39251facfa90717e97ed1d`。

同步检查131个服务器文件，复制129个并逐字节核SHA；2个checkpoint只记索引。43个NPZ保留本地 `runs/synced_two_row_prefix28_v1/`（git-ignore），Qwen cache数组未复制。完整对应路径/字节/SHA在[同步索引](observed_two_row_prefix28_v1/SYNC_SHA256_INDEX.json)，全12原图保持服务器哈希。全图可视QA见[三个完整contact sheets及说明](observed_two_row_prefix28_v1/visual_qa/QA.md)。

这仍是单训练种子、窄分布、DEV选择结果。当前有效率及已知正参考覆盖不足；未达到“核心方法有效、主要只剩扩实验”的交付标准。
