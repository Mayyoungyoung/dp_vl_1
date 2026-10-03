# 观测扩散真实工程验证

源码 `6e0203ba1335f9fa9975c523c657959f8bc9ab60` 的服务器 `.venv` 验证实际 **37 passed、0 failed、0 skipped**。其中core/stream实际Torch17项、driver实际Torch5项、pure协议15项；原本本机跳过的22项此次全部执行。完整用例、原stdout和PID/命令保留于 `validation/`。

实际CPU0单核、Torch 2.4.1+cu121，`CUDA_VISIBLE_DEVICES=''`、CUDA不可用；测试GPU小时0。外层record_job PID730063/child730068，`2026-10-03T03:11:57.616847+00:00` → `2026-10-03T03:12:05.416969+00:00`，exit0。验证body 5.545867790秒；外层 7.800122秒，嵌套费用不能相加。

真实测试包含：原普通共有初始化与独立/集合初值一致、候选隔离/置换、所有active参数梯度、100步cosine与40次DDIM、unknown/R不足或超K采样、实际四步与两步暂停恢复两步的model/Adam/RNG/stream/journal逐值等价；失败调用不重放；一次geometry+40denoiser；预测封存先于验收；独立6TRAIN×5时刻诊断预算。模型/数据机制测试使用合成fixture，新增schema测试使用真实历史receipt，不是正式TRAIN/DEV质量，也不能证明训练收益或已充分收敛。

部署code archive SHA `9a3b86911e2e435e6e0da6a3c55532bcb9a77bc52357e5fde630acecf0491be7`，5,570,560字节。runner SHA `a1bca2711173021757096003b84e99b10c61b1dad43e03b1c65d44713ed4de62`，wrapper SHA `a36dd3735aa5293069545c5eda0e9126d9382499a4ed7f81fc500192ac3e8667`。服务器与本地冻结字节一致，26份源码逐项与实际部署archive核对；共归档39份远端小原件。源码副本保留原字节，`.py`另加`.txt`防止被pytest误收集，原服务器路径与SHA在 `REMOTE_ARTIFACT_INDEX.json`。

这里只归档已经完成的CPU工程验证，无权重下载，无新增模型forward、规划、仿真或正式数据读取。正式训练已由ROOT独立启动并行政暂停后恢复；下面仅归档已封存pause2，不读取正在运行的结果。repeat/固定last285/去噪诊断仍分别调度。原模型、数据、历史结果不改。

## 已完成的行政暂停与实际训练脚本

fixture的部署字节与原始历史JSON只存在Git导出的换行差异：部署CRLF 7226 B / SHA `066126c810cbfe7d6107a14255b92a88d710415a2cce9b9f5a1ae3f21e1cc2fc`；历史原件LF 7103 B / SHA `5463d7fcfc77b57f86fba6893463697a4c155d7df3770c1b5bec2c39dfb919df`。归一换行后的字节和JSON内容完全相同，但不声称原始字节相同。实际测试使用已冻结部署fixture，证据见 `FIXTURE_PROVENANCE.json`；两份原件均未修改。


本次修复只从真实历史receipt的 `budget.actual_index_chain_sha256` 读取抽样链，完整历史receipt fixture已归档/核验；数据、损失、抽样、预算不变。旧0f失败证据仍保留在 `reports/observed_diffusion_startup_failure_v1`，旧验证目录不改。

实际incoming是 `observed_diffusion_train_6e0203b_v2.sh`，SHA `f44019ff27212b00cf6665c80290873b8a54ee4a822697ec45731c597a9fe552`。v1草稿曾误把新输出family替换到config路径，ROOT在发出模型调用前修正为v2；v1没有正式模型调用。v2仍读取原 `configs/observed_two_row_diffusion_v1.json`，输出新family `observed_two_row_diffusion_v1_fixed_receipt`，不覆盖旧失败。

ROOT实际independent pause2的外层PID734833/child734837，`2026-10-03T03:24:53.313741+00:00`→`2026-10-03T03:25:05.967551+00:00`，exit0。原内部状态step2/paused，6条issued=2次batch32 geometry+2次denoiser+2次optimizer，64draw/256训练路径状态。全部属于原12000步预算，额外曝光0；不是新一轮smoke。原内部状态按字节保存于administrative/，没有读取正在恢复训练的状态或质量。

暂停进程主体10.908654958秒；外层12.653810秒/0.003514947GPU小时，嵌套不相加，且与CPU验证GPU小时0分列。ROOT已另行resume。本归档没有重跑测试、启动/停止/恢复训练或新模型调用；不能由验证或step2宣称生成质量有效。
