# 正式两排 prefix28：普通强基线预登记

本轮只建设 ordinary baseline，不包含集合补全、allocator、路径更新或新的核心机制。正式采集源为 `262648796fd47b25a2051cc227b6838515b651c6`，数据目录 `/home/wzy/dpvlm/route_set_v1/data/observed_two_row_formal116_v1`。采集正在进行；本文不声称数据已闭合、已编码或已训练。

## 固定父场景和来源

沿纯注册的 indices0..15 共16TRAIN、indices64..75 共12DEV_MODEL，完整28个请求父、84个请求图文条件、756个采集尝试槽。必须全部选定父闭合后才原子导出，失败父不替换，不从后续TRAIN补数，不加入v4/v5/v6的DEV_COLLECTION父。其他角色只允许读取已有机械closure和布局hash，禁止其RGB、深度、轨迹、目标/成功结果。

导出入口 `scripts/export_two_row_observations.py` 复用正式collector的 `verify_corpus` 和 `live_layout_gate`，因此下游release必须保持采集器及所有依赖字节不变。输出使用独立 `observed_two_row_closed_prefix_export_v1`；没有放宽旧六任务或单障碍数据门禁。输入保持五键，current NPZ保持depth/pose/open/camera五字段。真实目标、柱体、mask、guides、route_config与类型均为单独监督/评价信息。

闭合可能含失败或中断。仅保留实际写出的图文输入；输入缺sup时，从同父已完成success attempts及实际artifact SHA重新验证正参考，不能因为sup未写完而丢弃输入或已知正例。未出现的目标输入和未尝试槽单列请求分母，不伪造图片/指令；无参考输入保留，不等于无解。unknown类型的有效参考仍训练，不把它当负例。正参考重新检查严格同态、源hash、初始末端/open、恒定reach事件、原raw/H24几何与类型验收；完整路径不截短。

## TRAIN容量和长弧质量门禁

`scripts/audit_two_row_train_reference_quality.py` 只对选定16TRAIN所有实际正参考打开路径/RGB-D数组，记录raw与模型 `resample_event_segments` H24的长度、最高/最低世界z、事件、TipValid和类型，以及正参考终点到stride2观测点的L2与L∞距离。collector保存的xyz_24与模型重采样函数是不同实现，不能假定完全同值；模型H24若出现有效率下降必须报告，不能通过删除参考把指标修好。

预声明surface capacity门槛是：每一条已知正参考终点都至少有一个有效观测XYZ点处于逐坐标±0.05m范围内。存在任一失败则停止该baseline启动，先交由TRAIN-only证据判断表示；不得用DEV挑free_offset或调bound。通过只说明表示空间有支撑，不证明模型能选对颜色/点，更不证明自由空间和机器人执行。长度和高度没有新增筛除阈值；1.4–1.8m的大弧仍按原参考保留。

## 固定普通模型与预算

[配置](../configs/observed_two_row_prefix28_selection_v1.json)固定 frozen real-Qwen官方revision89644892、RGB-D spatial encoder、straight-through peak observed anchor、surface endpoint逐坐标±5cm、H24/K4、width128/depth2/point64。端点attention辅助weight0.02、sigma0.025；原saturation正参考匹配完整访问同一可用正参考池，R<K允许重复，R>K由矩形匹配选择K个正例；不把示范条数当总解数。

训练1500×32、lr3e-4、seed0、每250步评价，192000训练候选路径状态；没有draft或更新。选模固定新两排协议 `UniqueClassifiedTipValidAtK + .05 * TipValidAtK`，保留原best和固定last1500全部结果；不混入旧单箱type分类器。当前只有普通强基线，不据其结果提前声明集合机制有效。

独立入口 `scripts/train_observed_two_row.py` 使用原 `train_observed_geometry.train` 的模型/损失/优化器/RNG/sampler/完整checkpoint恢复循环，在显式、退出必恢复的局部evaluation adapter中调用两排评价。旧脚本文件和默认入口不改。新driver/export/quality/selection/protocol SHA均写入每个checkpoint config，resume额外拒绝这些字段变化；源码、数据、步数和采样流仍受原恢复检查约束。

评价先完成所有model forward，再打开geometry/route_config标签。三目标的语言对照只重索引同父已经生成的另一目标结果，并核对同图/current/geometry；0额外forward、0额外路径状态，不合并候选池。所有真实评价请求和K4状态单列，原末尾latency诊断另记24次请求/96个路径状态；缓存头时延不等于完整Qwen端到端时延。

完成后的 `two_row_driver_receipt.json` 固定记录train、dev_model、last_dev_model三池的实际predictions NPZ、per_scene、metrics及存在语言对照时对应NPZ的SHA与字节数；包含TRAIN预测的独立可复核索引。唯一GPU入口为 `scripts/launch_observed_two_row_prefix28_v1.sh <immutable_commit>`，要求外部taskset单CPU、TRAIN门禁先通过、独立fresh缓存/输出；真实Qwen逐请求缓存和训练各有record_job日志/恢复命令。此脚本已准备但尚未执行。

## 当前实现验证

本地8项纯测试通过，Torch模块因本地环境无Torch尚未实跑。服务器冻结后需运行新增6项Torch测试：完整输入预算、无参考分母、两排选模公式、同图语言复用不额外forward、几何标签在全部预测后才读取、adapter异常恢复、driver恢复拒绝，以及tiny实际优化循环完整4步与2步中断后恢复4步的模型/优化器/RNG/sampler/hash链位等价。测试计数以实际pytest输出为准，未运行不计通过。

第一阶段只进行冻结CPU测试和正式闭合数据的导出/质量审计；质量证据通过后由root固定唯一表示并排真实Qwen缓存与普通训练。尚未发出新增GPU训练命令。
