# 提案：两臂各从真实 12500 继续至 15000

仅为下一普通扩散强基线修复提案，**尚未实现、冻结或启动**。有序关系损失三臂优先使用 GPU。本次 TRAIN 筛选提供有限继续依据：低 t body 和自由 Tip/Clear 同时提高，但高 t body 退化、父级改善不占多数，远未形成方法结果。

## 固定身份与唯一变更

保留两臂原数据 285 TRAIN / 1663 正参考、同 frozen Qwen 缓存、同 geometry/model、seed0、batch32、K4、H24、lr3e-4、100 个 t、40-step DDIM。uniform 与 mean-normalized Min-SNR γ=5 继续原目标，不扫 LR、γ、步数或种子。

从各自封存的 **真实 step12500** 恢复，不重跑已完成的 500 步：uniform 来自原 `runs/observed_diffusion_minsnr_probe_v2/uniform/last.pt`（尽管旧外层 failed，已有只读 seal）；weighted 来自 `runs/observed_diffusion_minsnr_probe_v2_recovery_v1/min_snr_5/last.pt`。新独立续训协议/输出引用旧 identity 与 PT SHA，不能伪装成原 500-step 协议的 resume，也不改旧目录。

恢复完整 model/optimizer/scheduler/scaler/RNG/抽样状态及四流 digest。启动前同时验证两臂 12500 的抽样流/计数一致、旧 500 步账本/summary/teacher/free 池封存、uniform recovery seal 及正确 16000/64000 原预算。不得要求两臂模型/Adam 参数相同：它们经过不同加权目标已有真实差异。后续每个训练边界继续核对相同父/参考/t/epsilon 链，evaluation 使用隔离 RNG，不改训练流。

## 不等价预算明确列出

每臂新增 2500 步 = 80000 输入抽样 = 320000 训练路径状态。加上已经完成的 500 步，每臂从共同 12000 父 checkpoint 总计新增 3000 步 / 96000 抽样 / 384000 状态；两臂新增执行总量 640000 状态，连同已完成 probe 为 768000。共享父模型成本单列，不按两臂重记父训练，也不把原500步重新收费或重新执行。

主终点预定为 **固定 last15000** 在原 36 DEV 指令的独立 K4 结果；不以取得更佳 DEV 就提前停止。为与其他基线保持可追溯的 checkpoint 选择，新增 10 次 DEV 机会，固定在 12750、13000、…、15000，每次 36×4=144 候选，合计每臂 1440 个最终候选槽、14400 次 denoiser request-call（360 请求×40，batch/duration另记）。last15000 池是第10次已存池，最终只按 hash 复用，0 重复生成；best 从这10个新池按已使用的 `UniqueClassifiedTipValidAtK + .05*TipValidAtK` 单一规则选取，仅作次要结果。原500步没有 DEV 机会，不能补造前两次；关系头3000步有12次选择，因此 **best 的选择机会不完全相同**，主要结论使用 fixed last，附 best 明示10 vs12。

最终 TRAIN 拟合诊断仍保留全部285请求×K4=1140槽/每臂，40-step DDIM=11400 denoiser request-call，固定与 probe 相同的原始噪声政策；这是新权重的新评价，预算另计，不能复用12500池。Teacher 固定原6 TRAIN×5t=30调用/臂，保留高 t 变差可能；不用于选模。预计每臂额外评价 geometry 请求360 DEV+285 TRAIN+6 teacher=651、denoiser request-call 14400+11400+30=25830；训练本身2500 geometry/denoiser批调用与这组逐请求计数分开记。新增 Qwen=0。

沿用已验证的 checkpoint-boundary 恢复/issued-before-work 账本；保存最近 checkpoint、best 和必要阶段 snapshot。中断前已完成封存池按 hash 复用，已发但未封存评价必须 fail-closed 留档，不能悄悄重抽直到成功。新输出保存完整成本和两层耗时，报告选外层总量，不与嵌套内层相加。最终比较确认两臂输入、四流、请求噪声和候选数一致；failed/unknown 不排除，不能把 train loss 较低当成自由路径有效性。

此提案只扩大普通扩散基线的训练充分性；是否落地仍由 root 排程和代码审查决定。没有将这次已看过的 TRAIN 筛选包装为独立确认测试，也不触及新 DEV32 或任何保留集。
