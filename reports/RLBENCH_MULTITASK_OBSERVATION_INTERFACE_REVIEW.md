# 六任务观测路径训练接口审查

结论：新数据足以建立多任务“示范路径与夹爪事件预测”基线，但当前 `semantic_targets=None`，不能据终点接近示范或夹爪状态相同宣称生成任务成功、语义目标正确、有效路线或多模式覆盖。现有基础 loader 可读取此监督；旧三目标 snapshot 和当前 RGB-D 终点约束需要明确适配。以下为最小方案，尚未修改训练代码或启动训练。

## 已核验的接口

采集器每父只有 3 个路线提案，原始任务给出若干语言改写，导出 id 为 `<parent>_lang<index>`，不是三种目标。各改写共享初态和同一组正参考。`observations.jsonl` 仍严格五字段 `id,parent_id,split,image,instruction`；监督单独保存 `task,observation,routes,route_types,semantic_targets`。输入 observation NPZ 只提供 RGB-D 对应的 depth、camera、当前 pose7/open1。模型不能读取 reference world、采集 guide、任务内部 state、路径、渲染策略或成功标签。

`load_observed_dataset()` 已允许 `semantic_targets=None`，保留无正参考样本及其原因；训练只选择至少一条正参考的 TRAIN 样本。`semantic_endpoint_accuracy()` 对 None 返回 None，当前通用评价正确保持语义率为 null。`observation_metrics()` 已给出候选到参考 / 参考到候选 ADE、参考终点误差、夹爪逐点状态及压缩事件序列准确率。其 Valid、Unique、ReferenceCoverage、Selected 与语义率应继续为 null，不能套用障碍 reach 的 TipValid 或 3 cm 目标规则。

## 必须避免的三项直接复用

1. `snapshot_observation_learning_curve.py` 硬编码每父三条 `_target` 指令、9 个尝试，新数据是变长 `_lang`、3 个尝试；直接复用会错误判断闭合或漏算失败。应新增很小的六任务闭合父 exporter，读取注册清单、`closed.json`、每槽 record、reference pointer 和语言清单，逐一 join。
2. 当前 `ObservedGeometryRouteHead` 将终点限制为初始观测 surface anchor 每坐标 ±5 cm。lift/lid/cup 等任务最终末端可能在初始图像中没有表面的空中；“最终端点就是图像目标表面”的归纳假设不适合全六任务。应先在 TRAIN 参考上审计最终端点到实际观测点云的距离，再使用不受这项硬限制的普通头。不得为让模型看似正确而把参考最后一点换成更容易定位的中间接触点。
3. `positive_endpoint_attention_loss` 把最终路径点当空间关注监督。此训练标签不进入 forward，信息边界本身正确，但对空中终点不一定存在对应初始可见表面。首个多任务基线关闭这项辅助损失，避免把 reach 的有效辅助项当成任务通用真理。

## 最小不可变快照

只用新 seed 282000 corpus，旧 6c 保留为历史来源，避免未核验的旧物理布局跨批混入。初次检查可固定每任务 TRAIN 索引 0、1 和 DEV_MODEL 索引 16、17，共 24 个请求父；该小快照只作链路与多任务基线开发，不能作为最终方法结论。后续自然扩到每任务全部 16 TRAIN 与 2 DEV_MODEL，约 96/12 请求父。选父按注册前缀，必须等待这些请求父全部闭合，不能用“最先成功的若干父”补齐。

快照在打开原始数据前先生成并核验当前 `layout_usage_gate`、跨批 hash 审计及所选父的机械 hash。任何任务出现跨角色物理/图像重复候选，拒绝为该任务生成可训练跨角色快照，写 `blocked_cross_split_layout_duplicate`，等待新的 group-aware 划分版本；不事后从原划分删除某几个不利样本。未出现重复只意味着未检测到，不能保证独立。仅处理 TRAIN/DEV_MODEL 的内容；DEV_SCORE、CALIBRATION、TEST_LOCKED 原始文件不被枚举读取。角色和 source pointer 不改写。

每父保留 requested=3、实际尝试、setup 失败、所有失败槽、成功与重复参考数、policy、各 worker 成本与中断上界。setup 失败父没有图像时保留于请求分母，不能生成伪观察；有图像但零正参考的 id 保留 manifest，排除正参考训练但不能标记不存在解。选中记录及当前输入/原始路径按 hash 固化；闭合文件读前读后复核，写独立 staging 后原子封口。缓存只读取严格 observation manifest，用真实固定 Qwen revision，不能顺便把 supervision 序列放进 tokenizer。

建议快照契约为严格五字段 `observations.jsonl`、保持现有 label 结构的 `supervision.jsonl`、独立 `parent_inventory.json` 与 `snapshot_manifest.json`。后两者保存每父 task、角色、来源、闭合失败、机械 gate/source/hash，不进入网络；task 仅用于采样和分组指标。loader 当前把 task 暂存 sample 但未返回 task 数组，后续需最小补出 metadata，不能将 task id 当额外条件。

## 最小训练与评价

首先以现有真实冻结 Qwen＋当前状态普通集合头建立无终点硬限制的基线。RGB-D 普通基线可复用相同 geometry encoder/context，但末点改为无 ±5 cm 硬截断的直接三维残差；这只是适应多任务表示的基线修改，不是创新机制。两者预测完整 H24 xyz/open，首点固定当前状态，标签仍按夹爪事件分段重采样，保留每段首尾；不能把抓取/抬起/释放阶段拉平。pose7 中的姿态尚未预测，所以这些输出不是可直接执行的完整机器人动作。

训练沿用 known-positive saturation matching，K=4 首轮（K=1/2 后续对照），同 seed、1000–1500 实际步、宽度 128、相同曝光量与优化预算，checkpoint 恢复完整 RNG/optimizer/scheduler/sampler。每步先等概率选择任务、再选择有正参考的父、再选择其语言改写；不能因某任务自然给出更多同义句而增加训练权重。评价先对同父语言平均，再父平均、最后六任务宏平均，同时给每任务结果与所有请求/可评估分母。checkpoint 只用预声明 DEV_MODEL 参考 ADE；另保留固定最终步，不能换成未定义的任务 success 指标选优。

| 可立即报告 | 含义与限制 |
|---|---|
| Candidate / Reference matched ADE | 跟已收集正参考的几何接近程度；不是正确性或真实解集覆盖 |
| 参考 endpoint error、best-of-K endpoint error | 最终末端点到参考的距离；不是语义目标命中或任务成功 |
| Event state / compressed sequence accuracy | 夹爪开闭及转换次序是否与参考一致；不是抓对物体或交互成功 |
| 每事件转换位置误差（建议） | 只在转换序列一致时按序比较参考坐标，序列不一致单列；不声称接触被验证 |
| 有限输出率、起点误差、长度、格式与候选预算 | 描述生成质量和预算；没有完整场景碰撞/执行证明 |
| 生成 task success、语义目标率、Valid/Unique/Coverage/Selected | 当前均 null，明确缺少评价器，不能用训练参考的原任务成功标签替代 |

快照后先做 TRAIN-only 表示审计：raw→H24 的事件次序与阶段端点、原始/重采样长度和偏差、末点到初始 RGB-D 点云距离；保存不满足 H24 表示的例子和原因，不能静默修标签。同图 `_lang` 是同任务同目标改写，因此当前 `paired_language_indices()` 不应伪造“不同目标”配对；没有可用 counterfactual 时控制样本数为零/不适用。已有三目标 reach 设置仍承担同图不同目标证据，这份六任务数据不重复声称该性质。

若以后需要 generated task success，需另设明确执行/任务专用验收接口，并处理当前缺少姿态、接触和物体运动状态的问题；仅 replay 示例成功不足以验证模型输出。此阶段目标是用真实多任务观测验证路径与事件学习链路和泛化，不把代表性任务数直接当成方法贡献或机器人闭环成功。

## 实现接续状态

新的 `snapshot_multitask_observations.py` 已按上述固定 24 请求父契约实现，加入 Py3.8 兼容的路径越界/符号链接检查、worker 最终会话计时闭合、前后两次机械 gate、初态及正参考完整 hash 核验；模型输入文件和 parent inventory/attempts 的 SHA 保存为 `snapshot_files_sha256`。跨角色重复会拒绝整份预注册快照并保存 blocked 诊断，不选择成功父替代。

`audit_multitask_train_representation.py` 已准备，固定只读 12 个 TRAIN 请求父，调用现有真实 backprojection 与事件重采样函数，报告所有正参考的 H24 事件/阶段端点/长度及点到折线误差、末点到初始 stride-2 点云的 L2/L∞ 距离。5 cm 每轴支撑判据仅针对 hard selected-point anchor；soft weighted anchor 可以处于点间，不能据此声称其表示不可行。没有修改或筛选标签，也不读取 DEV 原始输入来定尺度。本地相关测试 8 通过、1 因 Windows 符号链接权限跳过，待固定源后在 Linux 补跑；尚未对真实快照执行此审计。
