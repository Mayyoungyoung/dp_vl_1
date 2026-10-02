# Formal116 prefix28：固定 best 的真实在线评价

状态：独立实现与纯接口测试阶段；未启动 GPU，未取得在线结果。普通训练完成、源码冻结并获根线程排队后才执行。

本地验证命令：`F:/ProgramData/anaconda3/python.exe -m pytest -q tests/test_two_row_online_evaluation.py`。测试只用合成观测/输出检查接口、时间顺序、标签隔离、缓存及训练回执哈希，不代替真实Qwen运行；实际测试结果随源码冻结记录。

只评价首28注册前缀中固定12个 DEV_MODEL 父场景（283264–283275）各3条指令：36请求，每请求K=4/H=24，144候选槽。使用普通1500步训练按既定指标选出的原 `best.pt`；检查完成状态、原配置、best SHA/step、训练源码与数据哈希。`last.pt` 留在既有缓存评价，不再运行在线模型。无额外热身、重试、修复、候选筛除、重新排序或第二轮生成。

每请求连续执行：读取并核验RGB/当前depth-camera-gripper → RGB-D反投影和传输 → 官方固定Qwen processor → 实际冻结Qwen隐藏状态和mean/last池化 → 原完整观测几何编码器与集合头 → 输出传回CPU → 保存K4候选NPZ并写SHA封条 → 才加载该请求的验收标签 → 原两排类型/语义/线段检查。连续walltime从输入读取之前到检查完成之后，包含封存和标签I/O；各阶段另报。模型加载、元数据/完整性核验、整组之后的缓存一致性比较分别计时，并另报整个进程pipeline墙钟时间及GPU保留小时。

学习几何编码器使用一次原forward的前后计时hook分解，余量为路线头、输出传输及计时开销，不重复运行模块。hook同步开销包含在连续时间内；完成后核验参数版本没有更新。训练回执绑定原best预测、per_scene、metrics的文件SHA/大小，比较前后均重新检查。

`FrozenGenerator` 只拥有冻结模型、观测文件SHA和stride，不持有监督manifest或标签。调用只接受严格五字段观测字典，current文件从已验 `front.png` 同目录的 `observation.npz` 获得；NPZ只允许depth、camera intrinsics/extrinsics、gripper pose/open。标签检查与生成函数隔离，下一请求不能接收上条标签。前置出口验证可读机械元数据和原文件字节做完整性hash，不反序列化路径/验收数组。每条标签仅在其预测封存后反序列化；不读取未来参考轨迹数组。纯测试证明标签替换不会改变生成参数，不为该测试增加真实前向。

Qwen固定官方revision `89644892e4d85e24eaac8bacfd4f463576704203`、缓存原processor/runtime与BF16/SDPA。每请求重新编码RGB与指令；即便同图3指令也不复用Qwen或点特征。原头保持4096特征、128宽/2层、64点宽、stride2、peak surface anchor、坐标残差界0.05m、无refiner。只允许GPU1/35%显存、CPU1；启动器与源码由根线程冻结。

预测值非有限仍占原槽，格式不符不补点/裁剪。首个实际生成异常封存4个NaN失败槽，并将后续请求记录为未执行而非重试；缺少实际观测的注册条件同样保留36请求/144槽总分母，另报实际编码请求数。无参考不代表无解，KnownReferenceTypeCoverage仅在已知类型非空条件报告，未知有效类型仍计TipValid。没有评分器，SelectedValid保持null；两排2cm tip clearance/3cm target及5mm start/事件阈值不变，不能称完整机械臂或执行成功。

整组请求之后，在线mean/last、token数与训练cache NPZ逐输入核对，在线paths/events与原固定best `dev_model/predictions.npz` 按ID对齐，逐候选验收决定与原per_scene对照。只比较保存数组，不增加模型调用。保留串行对批处理的浮点差异、peak可能放大的差异及全部失败；不改输出来达到一致。缓存检查结果、输入/模型/源码/输出hash、逐请求日志及完整池保留。

复现入口（实际路径及源码SHA在根线程冻结后填入执行记录）：

```text
python scripts/evaluate_observed_two_row_online.py --run <completed ordinary run> --data <closed prefix28 export> --model <pinned Qwen> --training-source <actual immutable training release> --output <fresh output>
```

`--validate-only`只进行元数据/完整性核验，不加载Qwen或声称实际推理。输出必须新目录；无自动resume。操作系统文件页可能被前置hash预热，报告明确此边界；这不是冷磁盘延迟。逐请求连续计时包含服务所需阶段，但不含机器人执行。
