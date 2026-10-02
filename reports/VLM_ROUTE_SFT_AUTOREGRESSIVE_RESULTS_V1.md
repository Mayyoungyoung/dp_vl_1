# 真实 Qwen SFT 自回归对照：完整失败结果

同一真实 Qwen3-VL-2B LoRA checkpoint 的独立四次 K1 与一次整集合 K4 均已生成、保存并独立评价。两组在全部 24 条 DEV 指令上均没有通过原标准的目标终点或有效尖端路线。teacher-forced token NLL 的下降不能代替生成质量；本轮不支持直接 SFT 作为主方法。

训练 source `ba984062b6c672bb7e4c2572b2112994bbbd0c95`，生成与独立分析 source `8356b09b8436d00a4f98a76ffdd8c7d84033e6ca`。checkpoint best1500 SHA256 `675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f`。真实 Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`；同 checkpoint、输入、temperature .7、top_p .9、top_k 0；每组每场景四个名义候选，独立 K1 每次最多512 token、整集合一次最多2048 token。没有重试、修复、合并随机候选池或删去失败槽。

| 实测指标 | 独立 K1 × 4 | 整集合 K4 × 1 |
|---|---:|---:|
| 请求/实际计费候选槽 | 96 / 96 | 96 / 96 |
| 格式合格且有限的 H24 路线 | 21 / 96 | 5 / 96 |
| TipValid@4 / AnyTipValid@4 / UniqueClassifiedTipValid@4 | 0 / 0 / 0 | 0 / 0 / 0 |
| 3 cm 语义目标正确率 | 0% | 0% |
| TipClear@4（失败槽仍在分母） | 16.67% | 3.125% |
| 逐场景有限路线平均末端误差 | 92.77 cm | 105.93 cm |
| 完整请求生成中位数 | 41.123 s | 41.211 s |
| 生成 + 单独检查组件中位数 | 41.124 s | 41.212 s |
| 生成 + 单独检查组件 p95 | 47.638 s | 64.486 s |

全24指令/8父场景保留，包括1条无已知正参考指令。已知参考类型覆盖仅有17条可评价指令；其余未被当作不存在路线。格式/预算失败槽分别75和91。TipValid只是既有3D尖端盒碰撞、起终点和事件检查，未认证全机械臂、桌面、IK或执行；通用机器人 Valid 和 SelectedValid 保持 null。端点误差仅统计有限格式路线，再按场景平均，不能与另一个候选加权诊断混用。

一次 sampling repeat 不等于训练多种子。实际生成总耗时1873.015秒，即0.520282 GPU小时，峰值4,624,983,552 bytes。原训练另外556.787秒/0.154663 GPU小时，3750条监督路线曝光；不会把它称为与192k/384k路线状态的回归头同训练预算。每次请求都实际读取RGB/深度、processor处理、Qwen编码、自回归；组件相加的检查时延不是部署联合计时，也未包含评分器或执行。

完整原始文本诊断表明，格式问题之外还有空间问题。独立组90条完整JSON中69条路点数错误；全部90条终点都超过3 cm。整集合合法JSON共有30条路线，其终点也全部超标。详见 [原始 JSON 诊断](vlm_sft_raw_json_endpoint_diagnostic_v1/REPORT.md)。这些额外诊断没有修复文本、重采样路线或改变上述主指标。

下一步实际执行的是固定8个TRAIN样本的语法约束K1诊断，最多8调用/8槽。语法只保证K、H24、整数和事件格式，不读目标答案、不做几何修复。先判断训练记忆是否成立，再决定训练曝光或空间表示的修复，不盲目扩展另一轮全DEV生成。深度接口存在字节缩放误差，但目前不能将约1米空间错误归因于它。

证据与复现：

- `vlm_route_sft_autoregressive_v1/best_seed0/summary.json`、120次原始request journal、逐场景预算和SHA索引；服务器同名 `runs/` 保存原预测NPZ。
- `vlm_route_sft_analysis_v1/best_seed0/summary.json` 和两组逐场景检查。原生成summary SHA为 `76e4cceda1c802d87c357e8b3bd3b49c2d31a75321aad6aedadf5bd8066a31c4`。
- `vlm_route_sft_figures_v1/` 保留全部24场景×2方法，XY投影不代替3D碰撞检查。后续版只调整版式，原始预测不变。
- 原 `*.status.json` 中保存实测复现命令、release、PID、日志和exit。重新执行必须使用新output，不覆盖这次失败证据。推理脚本 `scripts.evaluate_vlm_route_sft`，独立分析 `scripts.analyze_vlm_route_sft`。
