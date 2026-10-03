# TRAIN6 Qwen 前缀重放：实际技术门禁通过

实际 source `eddfacaddab2d12c67f5a56fd775de173c05f9b2`；两父283200/283201的全部三目标指令，共6个固定TRAIN输入。2026-10-03 01:01:24 UTC完成，所有门禁通过。没有DEV、保留分区、正式续训或路线质量比较。本结果只支持原长度串行B1的冻结前缀重放，未证明padding/batched replay、实际学习收益或加速比。

## 原始结果

| 检查 | 实测 |
|---|---:|
| 当前官方完整特征 vs 共同头历史缓存 | 6/6逐值精确一致 |
| 写盘重读prefix后的零LoRA尾层特征 vs 完整前向 | 6/6逐值精确一致 |
| 两独立head/Adam分支的feature/loss/梯度/更新/optimizer/RNG | 2/2逻辑微步精确一致 |
| 每步更新后的full/replay特征 | 2/2精确一致，且均相对初始改变 |
| 实际改变的LoRA参数张量 | 8/8，114688参数 |
| 实际改变的普通头参数张量 | 60/60 |
| 冻结基础参数前后SHA | 625/625一致 |
| 服务器测试 | 37 passed，0 skipped，5.79秒 |

第一步68个可训练张量中64个梯度非零；四个A张量因B零初始化而为零，符合预期。第二步68/68均有非零有限梯度。两步分别用两个父的target0正参考，loss为0.07525034994和0.20142270625；它们是不同样本的链路值，不能当学习曲线或效果改善。

原官方layer26输入已包含视觉/DeepStack及0..25层结果；仅运行原26/27层与final norm，按原mask作FP32 mean+last。每次更新重算尾层；未用过时的最终4096维缓存训练。历史缓存没有完整token IDs，因此仅声明原观察行key、图像、processor/source、token count与特征数组核验，本次实际prompt/token IDs均另行保存。

## 实际预算与成本

账本28项全部完成：10次完整Qwen、10次serial尾层重放、4次K4路径头，共16个路径状态；full/replay各2次optimizer更新，合计4次。没有额外调用、重试或候选选择。首次捕获6+更新时2+更新后2构成10次full，replay同构。见 [完整账本](probe/call_ledger.json) 和 [机器成本表](COST_AND_BUDGET.json)。

| 阶段 | PID/child | UTC开始–结束 | 墙钟 |
|---|---|---|---:|
| 独立环境安装pytest8.4.2 | 666236/666237 | 2026-10-03T00:52:14.372543+00:00–2026-10-03T00:52:19.724098+00:00 | 5.351555s |
| 服务器CPU验证 | 669329/669330 | 2026-10-03T01:00:07.017873+00:00–2026-10-03T01:00:14.459403+00:00 | 7.441530s |
| 真实probe完整job | 669594/669595 | 2026-10-03T01:00:36.984377+00:00–2026-10-03T01:01:24.293499+00:00 | 47.309122s |

probe主体46.065015726s / 0.012795837702保守GPUh，嵌套于完整job 47.309122s / 0.013141422778GPUh，不能相加。GPU1、35%显存、CPU affinity `[1]`。峰值allocated 4375957504B（4.075428GiB），reserved 4406116352B（4.103516GiB）。这些是进程峰值/保守墙钟费用，不是利用率。

6份prefix序列化文件共2495388字节。账本full-feature调用中位56.903326ms、serial replay中位4.292113ms，但包含不同冷启动/梯度阶段，且未拆出backward、loss、状态复制、加载、hash、写盘。不能用二者比例声称正式训练加速或端到端时延；总probe除以两步也不是稳态微样本耗时。

## 来源、归档与限制

共同头固定composite last12000，SHA `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`。Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`，实际Torch2.4.1/Transformers4.57.1。运行源、官方依赖SHA、checkpoint/cache/export身份均见 [preflight](probe/preflight.json)。安装只发生在项目私有 `.venv-qwen`，日志保留pytest8.4.2及其依赖版本，没有升级共享环境或模型依赖。

本地下载42项JSON/文本/源码/日志，13项PT只保留远端路径、字节数和SHA；未下载权重或prefix张量。原55项全部索引及probe自身artifact_index逐条一致，[REMOTE_ARTIFACT_INDEX](REMOTE_ARTIFACT_INDEX.json)提供恢复来源，[QA](QA.json)记录核验。同步和本报告只读取已完成证据，0新增forward/search。

此probe为fresh-only、无重试恢复入口。两步状态PT是诊断证据，不能据此重新发相同预算。正式321输入prefix准备与3000×32配对必须另行冻结、实际执行及核验；本报告不宣称这些阶段已经完成。没有向MAIN添加质量行。
