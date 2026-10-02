# 旧六任务 DEV：独立父场景固定模型迁移准备

2026-10-03。当前仅完成只读资格检查和前瞻登记；尚未导出新数据、编码 Qwen、运行预测、训练或重新选择 checkpoint。判断是：可以进入固定模型迁移的准备阶段。旧12父未在已索引模型输入/指标中使用，既有物理布局门禁未检测到复用；不同 hash 不是统计独立性的证明，不将此称为新任务、OOD、IID 保证或最终锁定测试。

## 实际资格检查

源为 `/home/wzy/dpvlm/route_set_v1/data/observation_multitask_resume_probe_v1`，原6c采集，seed281000；只取原注册每任务索引16/17的12个DEV_MODEL父，不修改角色。初次检查10/12闭合，随后11/12，最终17:27:23UTC全部12/12闭合且无worker.lock；36/36注册尝试有最终记录，closure计36成功。没有提前取前10/11个父，也没有按成功替换。

旧批次没有原生 `mechanical_fingerprint.json`。本次没有向旧目录补写：仅打开这12个DEV的哈希一致初态 `reference.json`，调用固定5bb release中既有 `physical_layout`、`json_hash`、`audit_fingerprints`。与新84ee seed282000的96TRAIN+12DEV已存机械摘要比较，120行的精确物理hash、1mm物理hash、RGB文件hash均无重复。没有打开新批次原始world，也未打开任何locked父的摘要、图像、深度、路径或结果。跨角色partition注册表仅用于选择已允许的父身份。

服务器62个含显式观测路径的模型/检索config对应10份实际输入manifest，旧12父交集均为空；另核27份已保存multitask/retrieval逐场景指标的身份字段，命中0。已索引旧父只出现在跨批机械门禁的legacy/missing列表，不能把这种元数据引用误判为训练或选模使用。这是本项目可追溯运行范围的使用审计，不声称证明不存在项目外未登记使用。

最终只读程序PID458616、CPU affinity `[0]`，内部0.152秒、CUDA隐藏，exit0；已有fingerprint helper SHA `809271dd45b78fff387c9a5ae411599f8bfb044943d38d1fb1b7fa43fa3d8956`。完整输入hash、逐父closure、布局摘要和索引在 [eligibility.json](observation_legacy_dev_eligibility_v1/eligibility.json)，实际程序在 [audit_source.py](observation_legacy_dev_eligibility_v1/audit_source.py)。本轮为SSH标准输入短只读审计；没有伪称它是历史已commit训练源码或后台record_job。首次程序仅输出到工具且被截断，随后固定scope说明重新执行并完整保存；未计算模型质量。

## 已固定的前瞻模型清单

[configs/observed_multitask_legacy12_transfer_v1.json](../configs/observed_multitask_legacy12_transfer_v1.json) 保存原12父、资格报告SHA、六个训练run的summary/config/checkpoint SHA。普通与event_supported分别seed0/1/2，**各自原best与固定1500 last全部评价，共12个checkpoint条目**；不能根据新结果更换best或少报last。

原模型来自sealed prefix108，95个有正参考TRAIN父；原12开发父的选模历史不变。新旧数据任务相同、variation按相同索引规则；旧批次get_demo相机on，新批次五任务曾关闭中间渲染。这项采集实现差异需公开，即使先前单父轨迹等值审计通过，也不把跨批数据说成严格随机IID重复。

## 数据导出、缓存与评价入口方案

现有 `snapshot_multitask_observations.py` 故意只接受282000/validated-five注册，`check_multitask_model_gate`要求原生完整机械摘要；**不得放宽这两个历史门禁来适配旧6c**。准备独立的legacy DEV-only exporter，在新输出目录保存派生机械证明与源hash。它应复用既有路径约束、strict_restore检查、轨迹/事件hash核验和指纹函数，显式识别原6c的on策略与源SHA；原目录只读。导出前后重核完整12父闭合、原reference pointer及旁路布局证明。保留全部语言、三proposal和失败分母；当前36成功仍逐条核验，不凭closure直接信任route文件。

输出仍使用严格五键observations，以及分离的current RGB-D/camera/pose与监督；所有 `semantic_targets=None`、route types未知。无参考输入也保留，其参考误差为null；不得注入reach的目标坐标或成功标签。新export receipt不冒充旧snapshot协议。模型入口先显式检查该独立receipt，再调用原 `load_observed_dataset`、`load_geometry` 和 `evaluate(..., metric_aggregation='task_parent')`，无需改变旧训练器、旧模型默认行为或原训练fingerprint。

缓存直接复用 `scripts/observation_cache_qwen.py`，官方revision `89644892e4d85e24eaac8bacfd4f463576704203`，完整新manifest实编码；不从旧432条缓存改ID冒充新图像。缓存时间/加载/每输入耗时单独保存，GPU1/35%与CPU1使用已授权固定队列，冻结后执行。后续头评价可以CPU1完成：从每个原checkpoint config重建原free_offset/no-refiner结构，严格加载原权重；全部预测输出和source SHA保存。不得将新数据fingerprint写回原checkpoint来模拟resume。

主要指标仍为原observation_eval_v2的路径、末端及事件度量，加原task→parent宏平均；按seed配对，best与fixed1500分开，全部任务/父差值和无参考分母保留。语义、碰撞、完整执行和成功指标继续null。一次请求仍K4，不合并种子或checkpoint候选池；缓存、RGB-D预处理和每模型CPU前向成本分别记录，不能用缓存头延迟作完整Qwen请求延迟。

实施顺序是冻结独立export/eval入口及拒绝测试，CPU核验并正式导出，再由root一次排新Qwen缓存与十二源固定迁移。用户总授权覆盖本次固定迁移，当前清单 `execution_authorized=true`；入口尚未冻结/执行，没有发出GPU启动命令；本准备不影响并行丰富路线数据采集，也不构成新方法核心。

## 实现与冻结前验证

独立入口不改任何历史模型、门禁或指标：`export_legacy_multitask_dev.py` 完整导出，`evaluate_legacy_multitask_transfer.py` 严格按登记十二条权重做 CPU 预测，`launch_observed_multitask_legacy12_transfer_v1.sh` 从指定不可变 release 依次记录测试、导出、一次真实 Qwen 编码和十二源评价。启动时要求单核 affinity，并记录线程、CPU 集合、源文件 SHA、PID、退出码和各阶段恢复命令。

每条 checkpoint 的原 summary/config/权重 SHA、加载后 step/config 均核验；原训练 release 的模型、缓存产生器源码与当前字节一致，原 trainer SHA 一致，实际评价/几何函数 AST 一致。原 trainer 后续添加的训练辅助旗标不被误当作当前模型预测变化。Qwen cache 除新的 manifest SHA 外，全部生产配置必须等于原训练 cache，另强制 pooling=both、真实4096维输出、stride2、spatial 几何处理。原官方 revision、processor、max_pixels、dtype、依赖版本均包括在这个比较中。

每种 best/last 按 seed0/1/2 键显式配对；注册行打乱也不会错配。所有原始逐场景结果、各父/任务结果和预测 SHA 保留。新增拒绝测试覆盖未封口/锁定替换、训练布局复用、缺机械hash、标签源变化、符号链接越界、缓存生产配置变化、错误权重step/config、模型源码/评价函数变化和清单重排。Windows实际10项通过；符号链接测试因系统权限跳过，Torch模块因本地无Torch跳过。服务器冻结后应实际运行完整13项，不能将本地跳过计为通过。
