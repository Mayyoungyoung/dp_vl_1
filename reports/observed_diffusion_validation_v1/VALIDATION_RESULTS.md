# 观测扩散真实工程验证

源码 `0f1d5bfbd4ff788a6ea311339adfe77449db9634` 的服务器 `.venv` 验证实际 **35 passed、0 failed、0 skipped**。其中core/stream实际Torch17项、driver实际Torch5项、pure协议13项；原本本机跳过的22项此次全部执行。完整用例、原stdout和PID/命令保留于 `validation/`。

实际CPU0单核、Torch 2.4.1+cu121，`CUDA_VISIBLE_DEVICES=''`、CUDA不可用；测试GPU小时0。外层record_job PID717627/child717632，`2026-10-03T02:44:58.745554+00:00` → `2026-10-03T02:45:11.685473+00:00`，exit0。验证body 7.896415325秒；外层 12.939919秒，嵌套费用不能相加。

真实测试包含：原普通共有初始化与独立/集合初值一致、候选隔离/置换、所有active参数梯度、100步cosine与40次DDIM、unknown/R不足或超K采样、实际四步与两步暂停恢复两步的model/Adam/RNG/stream/journal逐值等价；失败调用不重放；一次geometry+40denoiser；预测封存先于验收；独立6TRAIN×5时刻诊断预算。测试使用合成fixture，不是正式TRAIN/DEV质量，也不能证明训练收益或已充分收敛。

部署code archive SHA `30918aa9deeb2ee72a5429bc95b75d51758bda4cf7e81d50d9ffe2b3267cabde`，5,529,600字节。runner SHA `08c5bf34e173a7d2772b8a7b408d95edb0b32296c0a359ef6c3771213154f2a0`，wrapper SHA `465b2b4185f71fca2464e65daebd4b3d635011412f8a4abeb05cba40269bfb38`。服务器与本地冻结字节一致，25份源码逐项与实际部署archive核对；共归档33份远端小原件。源码副本保留原字节，`.py`另加`.txt`防止被pytest误收集，原服务器路径与SHA在 `REMOTE_ARTIFACT_INDEX.json`。

这里只归档已经完成的CPU工程验证，无权重下载，无新增模型forward、规划、仿真或正式数据读取。扩散正式GPU训练与各repeat/固定last285/去噪技术诊断仍待root逐阶段调度，未因测试通过自动启动。原模型、数据、历史结果不改。
