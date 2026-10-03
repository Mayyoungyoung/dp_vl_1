# Composite108 普通集合回归：96 请求 TRAIN 父场景，固定总训练预算

实际完成于 2026-10-03 UTC。增加 32 个 TRAIN 父场景后，预声明选择的 best 在原 36 个 DEV 输入上由 59/144 条有效候选增加到 63/144，已分类有效类型总数由 30 增到 34；AnyTipValid 仍为 30/36。固定 last12000 则从 62/144 降到 56/144。结果仅支持这次单种子、重复使用 DEV 的小幅 best 收益，不能称总体质量改善或核心方法贡献。

原始结果、逐输入/逐父配对与权重索引见 [归档索引](observed_two_row_composite108_v1/INDEX.md)、[保存池比较](observed_two_row_composite108_v1/analysis/SAVED_POOL_COMPARISON.json)。以下分析没有新增 forward、Qwen 编码、规划搜索或原始标签读取；使用已封存的预测和逐候选检查结果。

## 实验身份、数据与公平范围

唯一对照是原 [64 TRAIN constant12000](observed_two_row_prefix76_convergence_v1/CONVERGENCE_RESULTS.md)。新训练使用 source `71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c`，同普通模型、1,231,965 个参数、冻结 Qwen、constant LR 0.0003、seed0、batch32、K4、H24、12000 步。初始化模型、Torch CPU RNG 与 sampler 初态的三个 SHA 均与原对照相同。模型结构和损失没有增加模块。

请求数据为旧 64 TRAIN + 新 32 TRAIN + 原 12 DEV，共 108 父、324 输入、2916 采集槽；旧父 283220 的 3 个缺失输入继续保留缺失记录、不替换。实际 95 TRAIN 父对应 285 输入，固定 12 DEV 父对应 36 输入。1868 条正参考中 TRAIN1663、DEV205。新增 extension DEV32 未读取；本轮并非独立确认测试。

[准备阶段证据](observed_two_row_composite108_preparation_v1/QUALITY_CACHE_RESULTS.md)确认全部 1663 TRAIN 正参考通过模型 H24 和端点容量检查，unknown 和长正例未过滤。旧 225 个输入、标签和缓存逐 ID/字节保留，其中包括全部 36 DEV 缓存；仅新增 96 个 TRAIN 输入实际运行固定 Qwen 编码。输入不包含验收几何或路径答案。

两臂均为 384000 次观察抽样、1536000 个训练候选路径状态、每 250 步一次的 48 个 DEV 选模机会；按 `UniqueClassifiedTipValidAtK + 0.05 × TipValidAtK` 选 best。旧 best5500，新 best6000。训练人口由 189 增至 285 个有效输入，所以平均每输入抽样由 2031.746 降至 1347.368，实际抽样链不同。相同的是总预算与初始化，不能称同逐父曝光或同抽样流。

## 固定 36 DEV 输入的完整比较

Tip 指原两排任务的末端点路径检查：2cm 净空/碰撞与 3cm 目标阈值；不是全臂、IK 或机器人执行通过率。unknown 类型仍可有效，只是不计已分类类型覆盖。没有学习评分器的 SelectedValid 证据。

| 指标 | 原64 best5500 | 新96 best6000 | 原64 last12000 | 新96 last12000 |
|---|---:|---:|---:|---:|
| TipValid | 59/144 (40.97%) | 63/144 (43.75%) | 62/144 (43.06%) | 56/144 (38.89%) |
| AnyTipValid | 30/36 | 30/36 | 28/36 | 27/36 |
| 已分类有效不同类型总数 / 每输入 | 30 / 0.8333 | 34 / 0.9444 | 23 / 0.6389 | 26 / 0.7222 |
| 有效 unknown 候选 | 27 | 28 | 39 | 30 |
| 已分类有效重复候选 | 2 | 1 | 0 | 0 |
| 语义目标正确候选 | 126/144 | 128/144 | 110/144 | 119/144 |
| TipClear | 65/144 | 70/144 | 77/144 | 68/144 |
| 已知参考类型覆盖（逐输入平均） | 0.05880 | 0.06713 | 0.04491 | 0.03704 |
| 候选匹配 ADE（cm） | 10.7754 | 11.8138 | 10.7043 | 11.1773 |

best 的逐输入 TipValid 差为 12 胜 / 10 平 / 14 负；12 父均值为 6 胜 / 1 平 / 5 负。last 为 11 胜 / 10 平 / 15 负，父均值为 5 胜 / 2 平 / 5 负。Any 并未随 best 有效候选增加而提高，ADE 反而更大。不能用单个聚合指标覆盖这些变化，也不从此单种子宣称显著性。

| 语义 × TipClear 四格，候选数 | 原 best | 新 best | 原 last | 新 last |
|---|---:|---:|---:|---:|
| 目标正确且净空 | 59 | 63 | 62 | 56 |
| 目标正确但碰撞或几何无效 | 67 | 65 | 48 | 63 |
| 目标错误但净空 | 6 | 7 | 15 | 12 |
| 目标错误且碰撞或几何无效 | 12 | 9 | 19 | 13 |

last 的语义正确候选增加 9 个，但目标正确且不净空增加 15 个。主要限制仍包含几何泛化，不能将语义提高直接视作路线有效率提高。新运行最后 3000 步记录的 DEV TipValid 在 36.11%–45.83% 间波动；48 次历史全部保留，没有改选模规则或只挑最高 TipValid 点。

## TRAIN：公共旧输入与新增输入分开

下表均为固定 last12000；新末步 TRAIN 诊断使用已固定的 last 权重，285 个请求/1140 路径状态，CPU 执行，无新 Qwen、DEV 请求或优化更新。

| 保存池 | TipValid | AnyTipValid | 语义正确 | TipClear | 候选匹配 ADE（cm） |
|---|---:|---:|---:|---:|---:|
| 原64：旧189输入 | 575/756 (76.06%) | 177/189 | 618/756 | 690/756 | 2.79077 |
| 新96：同旧189输入 | 643/756 (85.05%) | 183/189 | 685/756 | 710/756 | 1.77314 |
| 新96：新增96输入 | 346/384 (90.10%) | 96/96 | 360/384 | 370/384 | 1.79307 |
| 新96：全部285输入 | 989/1140 (86.75%) | 279/285 | 1045/1140 | 1080/1140 | 1.77985 |

全部新 TRAIN 有效候选中，已分类不同类型总数 415、unknown565、已分类重复9，合计989；未把 unknown 记为无效或缺失负例。公共 TRAIN 拟合提高而固定 DEV last 下降，显示仍有明显训练—开发泛化差距；当前比较不能把差异归因于某个新机制。

best TRAIN 另行保留，不能拿来替代末步拟合：原 best5500 在公共189输入 TipValid646/756、匹配 ADE2.36970cm；新 best6000 的公共189为588/756、3.05156cm，新增96为305/384、3.02399cm，全部285为893/1140、3.04228cm。两臂 best 选择时刻不同。

## 实际成本、失败和时延边界

真实服务器验证为70测试通过、0跳过（含3项 Torch 测试），16.90秒，详见 [验证归档](observed_two_row_composite108_validation_v1)。质量、缓存、训练、末步诊断分别启动，没有自动跨阶段调用。

| 阶段 | 实际记录 | 时间 / 成本 | 解释 |
|---|---|---:|---|
| 新96 Qwen缓存完整 job | PID652036/652037 | 21.775545s / 0.0060487625 GPUh | 旧225容器字节复用；旧编码不重复收费 |
| 缓存编码 body，含加载 | 准备阶段 receipt | 11.501351s | 属于上一完整 job，不能相加 |
| 原64训练 base | 原归档 | 536.404626s / 0.1490012851 GPUh | 相同12000步 |
| 新96训练 base | `peak_seed0/summary.json` | 580.198116s / 0.1611661434 GPUh | 含本阶段实际训练与评价 |
| 新96完整 train job | PID652623/652624，exit0 | 593.489648s / 0.1648582356 GPUh | 00:17:54.174724–00:27:47.664372 UTC；包含上一 base |
| fixed-last TRAIN完整 CPU job | PID657528/657529，exit0 | 19.117344s | 00:28:29.023053–00:28:48.140397 UTC |
| fixed-last TRAIN评价 body | 诊断 receipt | 11.824372s | 属于上一 CPU job，不能相加 |

新训练峰值 CUDA 951.37MiB，原对照955.51MiB。训练主评价2085请求/8340候选状态，包括48次DEV选模、best/last最终DEV和best TRAIN；另有缓存时延测量24请求/96路径状态。fixed-last TRAIN追加285请求/1140路径状态独立计费，未混入训练曝光。3个缺失TRAIN输入保留，不重新生成。

本轮仅缓存几何+头计时中位3.5504ms、RGB-D读取/预处理+头中位8.0224ms，**均不含真实 Qwen 与完整路径检查，不能叫端到端时延**。新模型没有另跑在线36请求，不能搬用旧模型74ms左右的完整在线时间。归档/绘图仅分析已保存池，零新模型前向。

## 完整图、证据边界与决定

[历史48点](observed_two_row_composite108_v1/analysis/history_comparison.png)与全部12父图均保留。四张完整 contact sheets 覆盖父283264–283275、每父三个目标、best/last与全部K4候选；原图可放大：[1](observed_two_row_composite108_v1/analysis/contact_sheet_1.png)、[2](observed_two_row_composite108_v1/analysis/contact_sheet_2.png)、[3](observed_two_row_composite108_v1/analysis/contact_sheet_3.png)、[4](observed_two_row_composite108_v1/analysis/contact_sheet_4.png)。目视已逐张核查标签、坐标轴、图例和候选完整性，无缺图或裁切。灰虚线为旧64，红实线为新96，圆/叉表示保存的有效/失败判定。这些图未额外载入真值几何，不据图重新认证碰撞。

这轮建立了更大数据下的普通强基线，保留 best 的有限收益和 last 的下降。样本仍是窄几何范围的两排任务、单训练种子、重复使用同12个DEV父；扩数据改变了训练人口和逐输入曝光。尚不能证明方法贡献、复杂语义泛化或机器人执行能力，也不能把新增数据收益归入新核心机制。不得因这个结果追加未登记训练长度、选模规则或随机种子；下一研究决定应继续围绕真实泛化失效和强规划对照。

## 源与可复查命令

实际 wrapper 为 `research_v2/incoming/composite108_pipeline_71cf0c1.sh`，归档中 [字节副本](observed_two_row_composite108_v1/source/composite108_pipeline_71cf0c1.sh)及全部源 SHA 已保存。训练和诊断现已完成，不重放已存在的 fresh 输出。两个 PT 仅记录远端位置/大小/SHA，未进入普通 Git：best `1ee6bab9e1d179a5f128e25f5367fee1ef3ff99d666152b89a97f04c749ddc6f`；last `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`。实际NPZ在本地 ignored 路径保留并索引。

本轮离线配对已实际执行，复查需选择新的输出目录：

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
F:/ProgramData/anaconda3/python.exe reports/observed_two_row_composite108_v1/analyze_composite108_saved.py --old reports/observed_two_row_prefix76_convergence_v1 --new reports/observed_two_row_composite108_v1 --output <fresh_analysis_directory>
```

下载压缩包 SHA 为 `0b6013f4d511a320772adc1acd342d2bae60781bc548e1786076380f760da92a`（1,830,701字节）；46项原始文件逐项 SHA/字节数均相同。所有原件、衍生图和本报告索引见 [LOCAL_ARTIFACT_INDEX](observed_two_row_composite108_v1/LOCAL_ARTIFACT_INDEX.json)。
