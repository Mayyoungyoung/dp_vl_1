# 3D HAMSTER 首次真实观测调用：超时，未完成候选

官方权重与代码已真实加载并执行。首条 TRAIN 输入在 300 秒请求期限内只生成 87 token，JSON 未闭合；原六请求计划只发出 1 次、其余 5 次未尝试。保留原始部分输出，不能补括号当作有效路径，也没有可报告的语义、TipValid 或执行质量结果。本轮并非六场景复现完成。

实验源 b127ce8514a63917cc3f1b7278db8f65289ba985，官方代码 DAVIAN-Robotics/3D_HAMSTER@97216a8493f46301bf569d398462b8bb21c458c5、官方模型 @ddc5987a56cdcb14e5e2297817612532e46e912b。18 个模型资产、50 个官方源码资产事前核验；CPU0、GPU1、35%显存上限与私有 .venv-hamster3d，bf16、实际 greedy 生成。原 GPU generation_config 的采样默认项已被调用参数覆盖，不把 warning 当作随机采样证据。

首输入 two_row_reach_283200_target0 为原 TRAIN 的黑色球目标，同一已存 RGB-D。445 prompt token，实际 1 generate、88 model forward；首 forward 445 token，随后 87 次各 1 token，KV cache 实际生效。generation 耗时 299.847636828 秒。输入只有官方 RGB/depth/语言条件，不含 GT目标或障碍几何；相机只用于输出坐标转换，未额外输入官方网络。640px 标准化图由 root 实际查看，原图/输入token/部分输出/调用账本均保留。

模型非decoder部件及embedding/head驻GPU权重 4,387,590,624B；36 decoder层以官方 Accelerate CPU offload 执行。实际峰值 allocated 4,662,350,336B、reserved 4,808,769,536B、RSS 19,805,024,256B，低于显存上限 8,864,694,272B。失败类型 TimeoutError，不是OOM。载入完成时间距主体起点 109.398586655 秒，包含前置资产核验，不应写成纯模型构造时间。

主体 2026-10-03 04:21:28.791699–04:28:18.210197 UTC，PID764105，409.418705553 秒、0.113727420 GPU占用小时。外层 PID764099/764104，04:21:28.679908–04:28:19.854033，实际 411.174125 秒、保守 0.114215035 GPU占用小时。二者嵌套，不能相加。原作业 exit1 已结束，不自动重启。

排查已确认旧hook对每个decoder子模块进行未固定页CPU权重上传/卸载；卸载设为meta，并非每token将整层权重再GPU→CPU复制。库默认 tensor move 的 clear_cache=True 可能带来大量 cache 清理调用，但目前没有 profiler 证明耗时归因。实际单token forward 排除了每步重新完整编码 prompt 的猜测。

下一实际工程验证为项目内层级传输hook的严格8token旧/新配对：保留原bf16数学与官方实现，先验证所有实际logits逐值相等、token相等和显存/RSS限制，再判断是否减少decode耗时。该比较仅为官方外部基线运行可行性，不是新路线方法或质量收益；完整单请求仍须独立决定，不能自动扩到6请求或DEV。

复现入口为 source/research_probe_b127ce8.sh；现有failed目录拒绝覆盖。原二进制prompt/generated tokens和depth保存在 ignored runs/synced_hamster3d_train6_probe_v1，REMOTE_ARTIFACT_INDEX逐项绑定远端原件。官方资产准备另见 ../hamster3d_assets_environment_v1/PREPARATION_RESULTS.md。没有更改官方共享环境、基础模型或测试标准。
