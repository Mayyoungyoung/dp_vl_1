# 两排场景：固定 last1500 的 TRAIN 拟合诊断

2026-10-02 19:23 UTC 已实际完成，单训练种子、开发诊断。**last1500 在 TRAIN 的 TipValid@4 为 87.50%，同一 checkpoint 在 DEV_MODEL 为 22.92%。此前 best500 的 TRAIN 24.48% 不能代表训练结束时的拟合程度。** 当前证据支持先处理泛化差距；这不是新的论文机制或机器人执行成功证据。

## 固定比较与实际结果

原训练为 prefix28 的普通 Qwen 条件集合回归，16 个 TRAIN 父场景、每父 3 个指令，K=4、H=24，1500 步、batch32。best500 沿原 DEV 规则选出；本次固定读取原 last1500，没有继续优化或重新选 checkpoint。DEV 数字直接引用已封存的原 last1500 结果，没有新 DEV forward。

| 指标 | best500 TRAIN（原保存池） | last1500 TRAIN（本次 CPU） | last1500 DEV_MODEL（原保存池） |
|---|---:|---:|---:|
| 父场景 / 请求 / 候选槽 | 16 / 48 / 192 | 16 / 48 / 192 | 12 / 36 / 144 |
| TipValid@4 | 24.48%（47/192） | **87.50%（168/192）** | 22.92%（33/144） |
| AnyTipValid@4 | 47.92%（23/48） | **100%（48/48）** | 55.56%（20/36） |
| 语义目标正确率 | 54.17%（104/192） | **96.35%（185/192）** | 64.58%（93/144） |
| UniqueClassifiedTipValid@4 | 0.6042 | **1.4792** | 0.3611 |
| 有效、未分类路线数 / 请求 | 0.2917 | **2.0208** | 0.5556 |
| 已分类有效重复数 / 请求 | 0.0833 | **0** | 0 |
| TipClear@4 | 44.27% | **91.15%** | 33.33% |
| 已知参考类型覆盖率 | 0.0043 | **0.3630** | 0.0375 |

last 的 TRAIN—DEV TipValid 差为 **64.58 个百分点**；语义正确率差为 **31.77 个百分点**。TRAIN 的 168 条有效路线包含 71 条已分类路线、97 条未分类路线；未分类不等于无效，也不能当成 97 种新类型。已知参考不完备，覆盖率只针对已知类型，按具备已知类型的条件计算。没有用未采到的类型作为负例。

## 原训练匹配下的残差

对原 best TRAIN 池和新 last TRAIN 池重新执行**原 saturation 分配**，核对实际 positive-assignment loss 与已分配目标的均方误差一致。无新预测、路径修复或参考筛选。

| 同一 48 请求上的拟合量 | best500 | last1500 |
|---|---:|---:|
| 原 saturation loss / MSE | 0.0050795092 | **0.0002663731** |
| 分配后平均顶点距离（ADE） | 11.4374 cm | **2.7284 cm** |
| 全部顶点距离 p95 | 26.3671 cm | **5.7239 cm** |
| 最大顶点距离 | 58.6595 cm | **12.1999 cm** |

MSE 对排除起点后的 XYZ 与乘以 0.2 的事件通道计算；它不是单纯的三维距离平方。距离统计为 192 条路径 × 24 顶点的几何误差，包含共同起点。这里 best 的 11.4374 cm 使用训练分配，区别于原评价器逐候选最近参考的 10.6060 cm；没有替换历史指标。last 的上述两种 ADE 恰好数值基本一致。

训练正例拟合已有明显改善，不能再用 best500 的低 TRAIN 有效率推断“末步仍完全欠拟合”。但仍有 24/192 条 TRAIN 路线未通过 Tip 检查、最大顶点误差 12.20 cm，不能宣称已完全拟合。DEV 的有效分类重复为零，因此当前没有证据把减少重复作为主要修复靶点。后续扩大独立 TRAIN 父场景的普通基线可检验数据覆盖不足；本次结果本身不证明具体原因或方法创新。

## 预算、信息边界与失败保存

- 本次仅新增 **48 次 CPU 缓存条件 head forward、192 条完整路径状态**；新增 Qwen 编码、GPU 使用、优化步数、DEV 预测均为 **0**。语言控制重索引同图其他指令的已生成候选，额外 forward 为 0。
- 固定源码 `743d9b269c27054b05c48cc5bab15017ab3fae55`，CPU 单线程、CUDA 隐藏。实际测试为 **3 passed in 0.12s**，测试进程耗时 0.465853s；分析主体 2.378593s、记录进程 6.114334s，均 exit0。两个口径不可相加充当总成本。
- 分析任务 record PID 516359、child PID 516363，19:23:25.739922→19:23:31.854256 UTC。没有新增失败作业或重试。
- 复用数据加载器会读取原 TRAIN/DEV 开发导出的数组；**只有固定 TRAIN 48 个请求执行新增 forward**，不能把此过程表述为“只打开 TRAIN 文件”。没有读取 locked 原始内容。几何和目标验收标签仅用于预测之后的检查；不进入 head、损失更新或路径修复。
- 初末机械门禁均检查到 38 个已闭合父，无重复组或 blocked 父；另有 `two_row_reach_283220` 初始观测不可用记录，属于本次 16 个父之外，不参与本次分母，也未替换任何父。
- 检查仍是原两排 **2 cm tip-only 几何 / 3 cm 目标**规则，事件正确不等于全身无碰撞或机器人执行成功。没有评分器，因此 SelectedValid 仍为空。

## 来源、字节索引与复现

完整逐场景结果、192 条匹配残差、实际命令与状态保存在 [结果目录](observed_two_row_last_train_v1/INDEX.md)。原 `analysis/artifact_index.json` 的 5 个文件已逐项核验 SHA-256 和字节数，包括两个本地 ignored NPZ；不把权重或 NPZ 加入普通 Git。

| 关键产物 | SHA-256 |
|---|---|
| 原 last1500 checkpoint | `7020190139bc712a98cb463d3b9f6f45255cb1e5354f4028fbde76aa6c44ef85` |
| 本次原始分析报告 | `76c75588a8efeed438b1a981ad8147f4983a10091464e70457f8980e02c73184` |
| 新 last TRAIN 预测池 | `b669af164ecd7d70df8df46133d79144f133b2518476461992976d19a52aeb1b` |
| 原 best TRAIN 预测池 | `c9ffd28aede212f1d046dfd1756ad56f0a8a431940d4c941435c37b101abd381` |
| 本地原启动脚本逐字节副本 | `dbb692c2409bda02a201a4c808d5a628395a697812c81d80a452aa1ebdabb6cd` |

`SOURCE_INDEX.json` 区分实际运行报告里的源码 SHA 与 Git LF blob SHA：相同 release blob 仅转换 CRLF 后，逐字节 SHA 恰好等于运行报告的 `34e4706f…5731`，没有把不同字节谎称相同。原训练 config/summary/best TRAIN 预测已和已有同步件重新核验；last.pt 未重复下载，来源沿用既有服务器字节索引和本次报告的一致 SHA。

实际命令已在 `analysis.status.json` 与 `frozen_wrapper.sh` 保留。已完成的 fresh-only wrapper **不可原地重跑**。需要独立复现时，从同一 immutable release 执行下列命令，明确换一个不存在的新输出目录；不生成或重训基础模型：

```bash
P=/home/wzy/dpvlm/route_set_v1
cd "$P/research_v2/releases/743d9b269c27054b05c48cc5bab15017ab3fae55"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$P/.venv/bin/python" -m scripts.audit_two_row_last_train \
  --run "$P/runs/observed_two_row_prefix28_v1/peak_seed0" \
  --data "$P/data/observation_two_row_prefix28_v1" \
  --output "$P/runs/observed_two_row_last_train_independent_reproduction/analysis"
```

本报告是固定 checkpoint 的拟合与泛化诊断，保留原 prefix28 报告、选择规则、预测池和结果不变。
