# 有序观测关系：同父三臂3000步续训执行协议

2026-10-03。状态：独立训练器已实现，尚未服务器运行、尚无三臂结果。根任务已根据真实数学/资源门授权实现；各实际阶段仍由根任务独立冻结和启动。它检验一个辅助损失候选，不将Soft-DTW、普通局部几何或集合指派包装为已经成立的新方法。旧segment v1不变。

## 已通过的数学和资源门

数学源固定 `e69ab8a134fee4eb8510c1dd52fb90e68a9e8a1c`，不修改其模块、测试或配置。原20项真实Torch测试全部通过，0skip。合成B32/K4/R9/H24/N12544下，三臂各2warmup+3timed，共15次loss/backward均有限；0真实样本、0模型forward、0optimizer。A/B/C实测中位损失耗时分别0.014036/0.140202/0.196731秒，C峰值allocated432907776B、reserved471859200B。参考描述准备0.438913秒单列。这只支持实现可承受，不能当真实训练时间或方法结果；真实训练会计入模型、grounding、缓存校验、检查点和DEV全部成本。

训练新源仅：`scripts/train_observed_ordered_relation.py`、`configs/observed_ordered_relation_training_v1.json`、`tests/test_observed_ordered_relation_training.py`、本协议。未改旧模型、普通训练器、数据导出、验收或冻结数学源。入口是 `python -m scripts.train_observed_ordered_relation`，CPU父状态检查、每臂训练和fixed-last TRAIN为显式独立stage，不串联。

## 共同起点、数据和唯一差异

原普通composite108 `peak_seed0/last.pt` SHA为`ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`，真实全局step12000。先以CPU0、隐藏CUDA执行 `--stage inspect-parent`，只加载该checkpoint与已封存元数据，0数据/0模型实例/0forward。记录实际键、model/Adam/scheduler/RNG/sampler/audit各自SHA。源码显示父状态确实包含AdamW、恒定LambdaLR、完整RNG、NumPy抽样器和实际索引链；真实checkpoint检查尚待运行。没有diffusion的t/epsilon/positive四流，不能声称恢复了不存在的字段。

三臂各从同一父状态恢复**已有Adam动量和scheduler**，不重置optimizer。所有原观测geometry/head参数继续训练；Qwen保持原冻结真实缓存。A `original_saturation` 调用原 `positive_assignment_loss`；B `xyz_divergence`、C `observed_divergence` 仅替换完整路线代价，随后相同正例saturation指派。B/C尺度、γ、XYZ/event系数和C七方向描述严格沿[e69数学协议](OBSERVED_ORDERED_RELATION_PROTOCOL.md)。不新增候选、查询或修复；未知类型和大回环正例全部保留，不将未采到的路线作为负例。

共同数据固定旧64+新增32请求TRAIN父中的95实际父/285输入/1663正参考，以及旧12DEV父/36输入。实际缺失父保持缺失，不替换。原export、质量1663全量检查、缓存receipt SHA固定到训练配置，复用旧双源验证。新variable数据、新extension DEV、score/calibration和locked均不读。生成器调用只由旧 `batch_inputs` 提供观测白名单；参考路径、描述及grounding标签只在loss侧。DEV的全部36 K4池先保存并hash，后调用原tip2cm/目标3cm几何检查；原paired-language控制复用已生成池，0附加forward。

lr3e-4、weight decay1e-4、clip1、batch32、K4/H24、event scale.2、grounding weight.02/sigma.025、surface-anchor与peak模式不变。三臂新增3000更新，结束全局15000。不是比较旧12000 A与新15000 B/C。

## 抽样、参考与预算证明

复制父checkpoint的sampler到独立临时生成器，提前生成3000×32、96000个索引。此过程不消耗训练/全局RNG。实际每一步仍调用恢复后的原sampler，逐batch与计划完全相等，并延续原索引SHA链；恢复和结束都复核计划前缀/最终sampler/累计索引链。每个实际抽中输入使用全部正参考，记录路径、事件、mask原字节的有序链，B/C和描述缓存不调用随机源。GPU/CPU Torch、NumPy全局/专用和Python RNG均保存/恢复；DEV在隔离RNG上下文运行，保持训练流可配对。

每臂实际调用预算：

| 独立阶段 | 模型batch调用 | 观察输入曝光 | 完整候选路径状态 | 优化更新 |
|---|---:|---:|---:|---:|
| TRAIN3000×32 | 3000 | 96000 | 384000 | 3000 |
| 12次DEV、每次36、B1 | 432 | 432 | 1728 | 0 |
| fixed-last全部285TRAIN、B1 | 285 | 285 | 1140 | 0 |
| 合计 | 3717 | 96717 | 386868 | 3000 |

C另有285次模型无关的参考描述准备，0模型调用；不缓存预测描述。缓存绑定每个完整参考/观测与冻结策略，训练组装逐行核SHA；保存CPU张量、hash及实际构建时间。B/A没有该缓存成本。记录原loss/grounding、C支持率、负代价计数；不把负代价裁零，也不将unknown支持缺失解释成free-space。新driver以局部scoped观察器读取原有三次Soft-DTW的XY/XX/YY返回值，finally恢复，0新增DP/forward；分别报告有效pair均值、自校正合计、固定索引XYZ及event诊断。C的shape含观测关系，这些全pair均值并非选定指派之后的loss分项，不能相加冒充精确训练目标分解。数值与梯度等价由真实Torch测试核验。

原父12000/Qwen缓存成本是共享历史投入，单列不在每臂重复计费。新增同曝光不等于同FLOPs或同时间；C的缓存、DP、自项和所有hash/同步均计入真实过程成本。每臂4500秒是跨尝试累计软边界，训练和独立fixed-last共用，不重置。每次issue前及更新边界检查；已经开始的CUDA或I/O不能硬中断，末次超限如实报。外层record_job包住内部process，缓存/DEV计时嵌套于内部，不能相加。这里没有完整Qwen在线E2E测量。

## 选模、封存及恢复

新增step250,500,...,3000恰好12次旧DEV选择。选择仍为`UniqueClassifiedTipValidAtK + .05*TipValidAtK`，严格大于才改best，平分保留较早checkpoint。12个原池全保留；best/last引用对应池，不额外生成。fixed-last285是独立命令，要求训练summary、last SHA、历史12池与ledger完整封存。主要科学判断用固定last，best另报，不在分析时换选模指标。

每25步保存完整model/optimizer/scheduler/RNG/sampler/index audit/positive chain/计划/历史/已发账本边界与成本。step1/2和每25记录真实梯度是否存在、有限、非零及norm；最终逐参数与父权重比较，不能用“设置trainable”替代更新证据。持久化`requests.jsonl`在模型或optimizer实际调用前append+fsync；异常已发调用保留为保守计费，未由完成step或完整pool证明的调用不得冒称完成。

`--stop-after 2`或25为管理暂停，在完整更新边界及该步应有DEV之后存last并状态paused；不改变3000总预算，不生成completed summary。恢复命令去掉stop-after，保留原source/config/数据/arm，带`--resume`。完成stage不可重新发调用。输出独占锁，旧父目录只读并最终重新核hash。

若硬崩溃后journal超出last边界，默认拒绝恢复，不隐形重放。唯一窄例外是：应有DEV已完整封存、hash/模型/精确36请求账本span全部一致，而last仍是该步评价前状态；恢复复用原池、选模并写checkpoint，0新forward。staging半池保留NaN/失败账本，不能自动重算。未知过程时长的硬杀尝试也拒绝将成本当零，需要根任务先依据外层记录独立收尾。正常暂停、失败、完成每次尝试都记录实际process_seconds，跨attempt相加，不以checkpoint时长代替真实已耗时。

## 测试和执行顺序

本地无Torch，纯接口测试与py_compile只证明接口部分；真实Torch测试必须在固定源、CPU0/CUDA隐藏完成且0skip，不能把本地skip当通过。测试覆盖：完整父状态恢复及Adam无storage alias、真实连续4更新对暂停2+原子保存/恢复2全部状态相等、原update顺序及A原loss/梯度精确等价、B/C同生成器有限梯度与无随机缓存、抽样/正例链和跨arm拒绝恢复、forward/optimizer/检查点rename失败禁止重放、完整DEV池复用与RNG隔离、累计预算。

根任务独立执行顺序：真实CPU测试→CPU父inspection并阅读→A训练/暂停恢复门→A fixed-last→阅读实际成本→B训练与fixed-last→C训练与fixed-last。每stage独立record_job，单CPU0/GPU1 UUID `GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab`、35%显存；不与其它GPU作业并发。脚本不会自动启动下一臂、扩大时间或追加种子。

## 固定科学判断

沿原决策卡，不新增门槛：C−A及C−B在固定last旧DEV的TipValid至少+5pp、已知Unique不降、语义正确率下降不超过2pp，目标正确子集TipClear改善；同时报告全285TRAIN拟合及12DEV父配对差、未知有效、全部失败和时间。此处是单训练seed开发筛选，不是显著性或会议贡献确认。B≈C则优先解释为成熟曲线对应修复；只有TRAIN提升不支持泛化；只让unknown增多而已知覆盖下降不支持质量—覆盖优势。失败后保留结果并依据证据决定，不在本轮调γ/描述/尺度或无限续训。
