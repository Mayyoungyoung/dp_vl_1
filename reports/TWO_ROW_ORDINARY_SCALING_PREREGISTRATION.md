# 两排观测普通基线：预先固定 16→32→64 TRAIN 扩规模

2026-10-03。prefix28 的固定 last1500 已拟合 TRAIN：TipValid87.5%、语义96.35%、匹配ADE2.728cm；相同 last 的旧12父DEV TipValid22.92%。这改变了先前只读best500 TRAIN的判断：当前主要缺口是泛化，不能声称普通集合头无法拟合。此卡预先固定两个数据规模，先执行普通基线；没有新增核心机制。

两阶段在检查新增 TRAIN 质量前同时登记。已有正式116物理登记不变，未闭合父必须等待，不按采集结果选父或替换失败父。首32存在至少一个无初始观测父的机械证据，实际可训练父数可能小于请求数；不能因此扩大前缀或伪造输入。

| 数据快照 | TRAIN注册index | DEV_MODEL注册index | 请求父/输入/路线提案 | 单次训练预算 |
|---|---|---|---|---|
| 已完成prefix28 | 0–15 | 64–75 | 28 / 84 / 756 | 1500×32，48000输入抽样，192000完整路径目标槽 |
| 新prefix44 | 0–31 | **同一64–75** | 44 / 132 / 1188 | 同上，独立从相同初始化训练 |
| 新prefix76 | 0–63 | **同一64–75** | 76 / 228 / 2052 | 同上，独立从相同初始化训练 |

TRAIN父及其三个目标、全部路线始终同组。旧DEV_COLLECTION未进入训练。DEV_SCORE、CALIBRATION、TEST_LOCKED仅允许既有机械SHA/闭合资格检查，其图像、路线、目标验收与模型指标不打开。旧12DEV是重复使用开发集，不能称最终确认测试。数据仍是固定canonical初态、两排柱±5mm等预登记窄范围的ID语料；64父不自动证明广泛几何泛化。

`ordinary_training` 两份新配置与prefix28字段逐项相等：真实冻结Qwen mean+last、stride2 RGB-D、peak/surface ±5cm、普通K4 H24 saturation、.02端点辅助、seed0、AdamW配方及每250步DEV选优规则均不改。最优选择仍为 `UniqueClassifiedTipValidAtK + .05 * TipValidAtK`，保存原选优best与固定last1500。初始化实际SHA必须等于 `7f81eba70dfcef2a3df19ebd5661ace3881abd552a1f2d4614cf490bf153bc13`；输入集合大小变化，**不要求抽样index链相同**。每父平均曝光会随规模增大而减少，应报告这一固定总训练预算取舍。

全部已记录观测、失败、缺输入、零参考、长弧和unknown类型保留；没有参考的输入不是不存在/无效标签。正参考训练范围由旧loader不变规则处理。TRAIN质量审计覆盖登记前缀内每个实际输入及每条正参考，检查同样的H24、事件和观测点端点支持；没有输入的父只保留请求与未观测分母。容量不通过时保存负证据，停止GPU阶段，不过滤失败参考或从DEV改表示。

CPU准备与GPU阶段独立启动。`prepare` 先机械核全部注册父闭合，再原子导出，核固定DEV身份，然后审计所有实际TRAIN参考；退出后由root读取结果再决定启动该阶段的GPU命令。未闭合时不创建导出/准备输出，不能把部分父当本轮已完成。`gpu` 重新核模型使用门禁和容量receipt，然后完整重编码此次全部实际输入，不借用过时缓存。新旧manifest的SHA本来不同，因此按36个DEV ID比较观察/语言/监督行、其全部原始输入/参考/验证标签SHA，以及真实缓存的mean/last数组与输入元数据。数组不等时保存差异并在训练前停止；不能根据新DEV表现选择旧或新特征。

Qwen每个阶段重新加载与重新编码旧36DEV的成本都计入。cache状态、训练墙钟、候选/优化曝光分别报告，缓存head时间不称端到端Qwen请求延迟。训练结束核初始化SHA及1500步/48000抽样/192000目标槽。原采集花费是共同语料成本，不能在16/32/64横向表里各自重复相加。

命令模板中的 `REVISION` 必须由root换为已审查、已部署的40位不可变commit。单作业`taskset -c 0`仅为root确认空闲CPU后的模板；正式collector CPU2/3继续由root统筹，不得自行提高总CPU4或GPU1/35%配额。

```bash
SRC=/home/wzy/dpvlm/route_set_v1/research_v2/releases/REVISION
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_scaling_v1.sh" REVISION 32 prepare
# root先读 runs/observation_two_row_prefix44_preparation_v1/train_quality.json 与 preparation_receipt.json
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_scaling_v1.sh" REVISION 32 gpu

# 64TRAIN全部闭合后仍执行预登记的下一规模，不按32阶段得分挑父或改超参。
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_scaling_v1.sh" REVISION 64 prepare
# root读取prefix76的独立质量receipt之后：
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_scaling_v1.sh" REVISION 64 gpu
```

每个整体阶段为fresh-only，不能在中断后重播整条pipeline。`record_job`保存每个export、quality、cache、training子进程的命令、PID、log、退出码与恢复策略。export/audit需独立新输出并保留失败；相同config的cache可以用实际记录命令恢复，训练使用该阶段记录命令加`--resume`及完整原checkpoint。恢复不改变父集合、阈值、普通训练配方或DEV身份；任何手工接续也须核同样receipt，不能绕过未通过的门禁。

研究决定依据是普通基线在相同DEV和固定总曝光下的数据规模曲线，包括语义、TipClear、TipValid、AnyTipValid、已分类Unique、unknown有效数、重复和known-reference coverage。参考不完整，不以未覆盖作为无效标签；box/tip代理不称机器人执行成功。只有这条普通曲线仍留有明确且可干预的缺口，才重新审查单一机制假设。两新规模目前仅源码/协议准备，尚未服务器执行或产生结果。
