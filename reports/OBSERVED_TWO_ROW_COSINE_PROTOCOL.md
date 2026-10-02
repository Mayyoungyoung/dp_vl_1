# 一次 fresh12000 cosine 普通基线对照

状态：实现与预注册；尚未真实训练。源冻结后先通过服务器真 Torch 工程门禁，再由 root 唯一启动。历史 base、two-row driver、constant12000 及其结果均不修改。

现有证据仅支持一个优化假设：constant12000 的末步 TRAIN 净空率为690/756=91.27%，但终点语义通过率618/756=81.75%；138个终点失败距答案3.006–4.288cm。与 best5500 相比，净空改善而共享 anchor 移动、语义下降。这个相关关系不证明学习率是原因，也不能把超过3cm阈值直接写成识别了错误物体。依据为 `observed_two_row_prefix76_convergence_v1/CONVERGENCE_RESULTS.md` 和 `TRAINING_DRIFT_ANALYSIS.json`。

唯一假设：在完全相同的样本序列和12000更新预算下，单周期降低学习率可能减少末期 anchor 波动，保留已学净空几何。可以被末步 TRAIN 语义/终点距离不改善，或固定 DEV 质量—覆盖取舍不改善证伪。不增加 warmup、floor、restart、另一个 seed 或曲线搜索；本次无收益即停止此优化方向。没有先做影子单步 pilot。

这是标准优化修复，不是任务路线集合的核心贡献。余弦调度已有 [SGDR 原文（ICLR2017）](https://arxiv.org/abs/1608.03983)；这里仅使用一个无重启周期。[PyTorch2.4 LambdaLR 官方文档](https://docs.pytorch.org/docs/2.4/generated/torch.optim.lr_scheduler.LambdaLR.html) 说明 lambda 函数本体不会随 state_dict 保存，因此必须由冻结协议重建相同函数，另行锁定源、总步数和逐步实际LR。

## 固定比较与公式

- 对照是 source `1417cdc802674a00e70d4ced3181c6a6ede2a99e` 的实际 constant12000。新模型从相同 seed0 初始化，绝不从 constant1500/12000 恢复训练。旧 last SHA `d6d150a93f739fe1087732d25d5c9c2a891c86184888582c9788af4ac97593f6`；config/summary/source-manifest 哈希写入新 JSON。
- 同 prefix76：64注册/63可用 TRAIN父、189观测输入/1108正参考；36固定 DEV/205正参考。283220的3个缺失输入保持缺失。复用固定 Qwen cache，不新增 Qwen编码，不读 reserved/locked raw。
- 同模型 surface/peak、H24/K4、saturation loss、grounding权重.02、AdamW/weight decay/梯度裁剪、batch32、种子与所有数据/检查标准。唯一优化变量是学习率。
- 第 k 次 optimizer.step 实际使用 `3e-4 × (1+cos(pi×(k−1)/12000))/2`，k=1…12000。旧循环先 optimizer.step 再 scheduler.step；第一步3e-4，第6001步1.5e-4，第12000步约5.1404e-12。完成12000后保存的下一步LR=0，未用于额外更新。
- 完整12000步：384000观测抽样、1536000候选路径状态。每250步同一个 DEV criterion 选 best，共48次，另外保留 last。禁止按本次 DEV改变 schedule、终止步数或评价阈值。

初始化前保存的模型tensor/CPU Torch RNG/sampler SHA必须逐项等于旧真实记录。最终完整 sample-stream chain 必须等于 `40c280ca213bfdd89502108ac4e82917cfa4d68fea7f961ad7f2d1de5c84b567`，最终完整 RNG 与 sampler 状态也须相同。真实预测和 Hungarian 分配可以随不同权重改变，不要求损失或匹配结果相同，不把不同匹配偷记成不同数据条件。新的1500中间权重不应等同 constant1500，因此不复制旧的“1500全状态相等”门禁。

## 隔离、恢复与计时

新 `train_observed_two_row_cosine.py` 通过 scoped adapter 调用旧 two-row/base loop，退出或异常时恢复函数引用。临时 LambdaLR 工厂构造带可序列化审计字段的子类；没有把闭包写进实例 checkpoint。每次推进记录 optimizer 刚实际使用的 LR，恢复前核对完整 trace、last_epoch、内部 step count、总步数及 optimizer 当前LR。`actual_learning_rates.json` 给出所有12000步，不只配置声称值。

独立输出 `runs/observed_two_row_prefix76_cosine_v1/peak_seed0`；pipeline 与旧 base 双层独占锁。resume只能进入同一 cosine output/源码/策略，完整恢复model、optimizer、scheduler、RNG、sampler、history、exposure。原constant checkpoint直接拒绝。进程异常保留日志、状态和完整时间；未保存更新另报最多一个250步间隔/32000候选状态的不确定上界，不能把重放成本藏入“严格同实际计算量”。没有失败的正常运行才可声称相同实际更新数。异常遗留锁需要先核验PID再人工处理，不能无核验自动删除。

已到12000的 resume只做封存恢复：必须已有旧driver完整receipt、summary和全部池哈希，否则停止人工收尾，不再调用base生成池。固定 last TRAIN 189请求/756路径状态用CPU一次，staging池已封存可恢复而不追加forward；未封存中断保留完整189请求上界并停止自动重试。

48次DEV选择=1728请求/6912路径；最终 best DEV36、last DEV36、best TRAIN189=261请求/1044路径；旧head时延探针24请求/96路径。另固定 last TRAIN189请求/756路径。相同图的语言对照重排已保存池，0新增请求。这些评估成本单列，不与训练1536000状态混称。总评估请求2202、完整路径8808（含24时延探针、189 fixed-last），新Qwen=0。缓存头时延不是Qwen端到端时延，本次不新增在线36请求。

constant真实训练两段外层耗时559.38s，可据此估算同预算约9–12分钟、约.16–.20 GPU小时；仅估计不是已用成本。新run保存实际初始化/训练/选择/最终CPU诊断外层时间、record_job过程时间、base嵌套时间与显存；不能把嵌套时间重复相加。

## 工程门禁和复现

新增五文件：driver、`launch_observed_two_row_cosine_v1.sh`、`test_two_row_cosine.py`、`configs/observed_two_row_cosine_v1.json`、本协议。旧模型/训练/评价/collector源在启动前与constant实际source manifest逐字节核验。recipe中的1500仅是原数据/架构身份，不是新总预算。

新测试覆盖固定单周期预算、k−1相位、完整LR trace、跨协议/来源/总步数拒绝、采样链、初始化、RNG、48实际选择次数、scoped引用恢复，以及真Torch scheduler/旧constant adapter等价。最关键的真Torch tiny使用实际共享训练loop：连续200步 versus100步保存并resume到200，eval_every100，逐值比较所有 STATE_FIELDS **包括history**。采用100步间隔是为了保留旧loop最近100步损失统计，避免短resume的报告窗口差异被误称不确定性；正常生产检查点250步也满足此条件。另以原真实2步循环验证constant adapter全训练state等价。没有修改旧loop。

本地实际测试：13 passed、3 skipped（0.45s，receipt `.bootstrap/cosine_local_tests.json`）；py_compile通过。3个skip均因本地无Torch，pure测试不能替代上述真实门禁。独立peer复核已完成，无当前阻断；base_lrs整体缩放仍自洽的潜在恢复缺口已在冻结前补上固定初始LR对照与拒绝测试。服务器由root在冻结source下运行pytest且不得skip3个Torch测试；CPU1、GPU隐藏。通过后root独立启动：

```bash
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/<COMMIT>/scripts/launch_observed_two_row_cosine_v1.sh <COMMIT> fresh
# 仅真实中断且核验状态/同源码后：末参数 resume
```

launcher核GPU1 UUID、35%预算和CPU1、冻结自身SHA，record_job保存PID/退出码/命令。根线程决定实际启动，不由本协议自动启动。全部原件和失败保留；结果无论正负均报告 fixed last TRAIN fit 与 fixed DEV best/last，区分训练拟合和泛化、语义与碰撞，不将普通优化改进包装成新机制。
