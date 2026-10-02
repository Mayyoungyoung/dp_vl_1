# prefix44：普通基线的固定6000步收敛控制

2026-10-03。prefix44实际31TRAIN父/93输入在1500步下的TRAIN TipValid36.02%、语义81.72%、saturation loss0.00298756；原16父1500步则TipValid87.5%、loss0.00026637。扩大数据后的固定总曝光尚不足以排除欠拟合，因此新增一次有上限的普通基线检查；不能继续把全部差距解释为泛化，也不能把增加训练的收益称为新方法。

只使用已冻结prefix44，同一93TRAIN输入、旧36DEV、实际Qwen缓存、seed0、batch32、K4、H24和AdamW lr3e-4；原LambdaLR恒为1。普通架构、正例匹配、端点辅助、评价阈值和每250步开发选择均不变。总步数**预先固定6000**，不是观察DEV后延长；到达上限即结束。原`selection.ordinary_training.steps=1500`仍用于历史数据/配方身份核验，新checkpoint另写真实`steps=6000`、schedule协议和全部source/policy哈希。

唯一新实现是作用域内包装旧`base.train`入口，把收到的1500改为6000并加入审计元数据；退出或异常必恢复原函数。原训练loop、历史driver、scaling、collector文件均不修改。调用旧driver保留全部原数据门禁、best/last预测、历史与完整恢复状态。

先从相同初始化在**新目录**执行6000配置、`stop_after=1500`。禁止复制原checkpoint来伪装复现。原last SHA固定为 `166eae2300ea0f478c30bfa3f2f33d8e9c34e0fdf7346a15df628708b73a77a8`。核对model、optimizer、scheduler、显式scaler(None)、完整RNG、sampler、抽样index-chain、step、exposures、best、history及关键训练config；所有必需键必须存在，不能把双缺失当一致。只忽略总步数、输出/源码路径、stop/resume字段、计时与容器字节哈希。数值严格相等、零容差；失败则保留证据，不能自动换种子、拷权重或放宽门禁。

通过后保存少量`stage1500_snapshot`（last/best及配置、历史、来源和状态），root读取`stage1500_audit.json`后单独启动`finish`。finish从**新跑出的**1500状态恢复到6000；源/policy/reference/数据hash跨协议变化拒绝。中断恢复只能明确`--resume`，保留原stage日志；不能重播已完成阶段或覆盖旧1500实验。

若已保存step1500但snapshot/audit封存中断，允许显式stage1500 resume仅核对并补齐元数据，优化/forward均不再执行；已存在的任何snapshot字节不一致即拒绝。6000后的TRAIN诊断以独立staging保存request预算和完整hash receipt，完成后原子发布；已密封池的resume只核SHA及补外层receipt，不重新forward。尚未密封的诊断中断则保留staging与0–93实际请求边界、372完整路径状态的预留预算，fail-closed交人工核查，不能删除后无记账重跑。这一诊断边界不冒称可无损恢复部分推理。

若step6000已经保存，但旧driver在最终预测、summary或`two_row_driver_receipt.json`封存前中断，同样不自动调用base重做最终推理。必须有完整ordinary receipt及一致的summary、训练预算、三组预测产物SHA才能继续零优化收尾；缺失或不一致时保留原池和全部日志并fail-closed，交人工审计封存。不能仅依据summary存在就宣布收敛run完成。

| 成本项目 | 优化步 | 输入抽样 | 完整路径目标槽 |
|---|---:|---:|---:|
| 原prefix44参考实验，原样保留 | 1500 | 48000 | 192000 |
| 新run前缀复现 | 1500 | 48000 | 192000 |
| 新run通过门禁后续训 | 4500 | 144000 | 576000 |
| 新run累计，前两项新成本之和 | **6000** | **192000** | **768000** |

原缓存来源与编码成本继续保留为共同前置成本；本run新增Qwen编码为0。每阶段record_job保存命令、PID、log、退出码和恢复命令。base内部elapsed、driver事件墙钟、外层进程墙钟是嵌套口径，不重复相加；额外加载、比较、保存和诊断开销以实际record进程成本报告。保留所有失败和中断时间，不只报告成功尾段。

6000完成后除旧loop保存的best TRAIN、best DEV、last DEV池，再单独CPU读取固定last6000，对全部实际93TRAIN输入生成一次K4池（**93额外请求、372完整路径状态**），新增Qwen编码/优化/DEV请求均0。保持3个缺输入分母，若有无参考输入保留生成、参考误差无定义。保存预测、逐场景指标、耗时及hash；不以这次TRAIN诊断重新选best。该诊断用于判断充分拟合后剩余误差，不能当训练收益或机器人执行证书。

实际运行须root先冻结部署，单核taskset、GPU1/35%；目前仅实现/测试准备，不在本卡虚构训练结果：

```bash
SRC=/home/wzy/dpvlm/route_set_v1/research_v2/releases/REVISION
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_convergence_v1.sh" REVISION stage1500
# root读取新run的stage1500_audit.json，全部严格一致后另行启动：
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_convergence_v1.sh" REVISION finish
# 只有实际中断、检查对应PID/状态后才使用第三参数resume；不能重跑已完成stage。
```

评价仍用同一重复开发集；6000比1500多了优化和DEV选择机会，两者必须分别列best与固定last，不能称同训练预算的机制对照。即使结果改善，也只建立一个更充分训练的普通强基线。此次不扩展64父、不启动新模块或继续超过6000。
