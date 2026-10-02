# 固定 K4 空间路径惩罚 A*：实际 TRAIN / DEV 结果

固定空间惩罚得到少量新类型，但显著降低候选有效率并增加 CPU 时间。DEV 已知有效类型总数由 24 增到 29，TipValid 由 96/144 降到 42/144，请求中位时间由 0.914 秒增到 6.143 秒。**这是标准多样化规划的负面质量—覆盖—成本取舍，不是新方法贡献或全面优势。** 不调 sigma，不补失败槽，不把有效 unknown 当失败。

生成源固定 `1417cdc802674a00e70d4ced3181c6a6ede2a99e`。根线程先完成 98 项服务器测试，再分别启动 TRAIN 机械预检和 DEV；两阶段均真实 exit 0。原三份规划器/颜色原型源码、观测自由图、接入、2cm/3cm评价、K4、20000节点/2秒搜索上限均保持；只有 grid edge 的历史路径惩罚支持由同一离散边改为 `sigma=.05m` 的连续 raw 线段距离场。方案见 [预登记](OBSERVED_TWO_ROW_SPATIAL_PENALTY_PROTOCOL.md)。

## 数据与配对完整性

共同拟合来自固定 prefix44 的 32 请求 TRAIN 父，实际 31 父 / 93 输入 / 556 正参考；283220 缺图保留、不替换。prototype+workspace canonical SHA 为 `384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b`，仅排除不可复现的实测 `training_seconds`，参数、训练行、源文件哈希全部保留，与原实际 A*32 完全一致。

- TRAIN：预登记首四父 283200–283203，全部 12 指令；edge 与 spatial 各 48 槽，共 24 请求 / 96 槽。12/12 首条 raw 和 H24 逐值一致。
- DEV：同一组 12 父 283264–283275，全部 36 指令 / 144 槽；新 spatial 对比封存的原 A*32，不重新生成旧对照。36/36 首条 raw 和 H24 逐值一致。
- 按 ID 配对；所有候选包括 NaN、重复、unknown、有限但语义错误均保留。仅使用已保存的检查结果与路径数组做后处理，没有再次读原始观察/验收标签，也没有新的模型或规划器调用。
- 这些 DEV 已用于开发，不能写成 LOCKED、OOD 或独立最终确认。

## 全部槽的结果

| 设置 | TipValid | AnyTipValid | 已知 Unique / 条件 | 已知重复 / 条件 | 有效 unknown 总数 | 已知正例类型覆盖 | 完整 raw | 生成失败槽 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN edge | 32/48 = 66.67% | 8/12 | 8/12 = .6667 | 22/12 = 1.8333 | 2 | 8.1944% | 32 | 16 |
| TRAIN spatial | 15/48 = 31.25% | 8/12 | 10/12 = .8333 | 5/12 = .4167 | 0 | 8.1944% | 15 | 33 |
| DEV edge | 96/144 = 66.67% | 24/36 | 24/36 = .6667 | 65/36 = 1.8056 | 7 | 3.1944% | 100 | 44 |
| DEV spatial | 42/144 = 29.17% | 24/36 | 29/36 = .8056 | 13/36 = .3611 | 0 | 3.6574% | 44 | 100 |

所有生成的有限路径均通过原 tip 线段检查。DEV 的有限无效路径是同一 `283264_target1` 的目标错误：edge 四槽、spatial 两槽，均非新增碰撞。不能把总失败数全写成碰撞；也不能把“raw 完整返回”写成有效。

| 失败类别 | TRAIN edge | TRAIN spatial | DEV edge | DEV spatial |
|---|---:|---:|---:|---:|
| 无合法 goal attachment，未返回路径 | 16 | 16 | 44 | 44 |
| 两秒搜索时间耗尽，未返回路径 | 0 | 17 | 0 | 56 |
| 节点预算耗尽 | 0 | 0 | 0 | 0 |
| 已返回有限路径，但目标错误 | 0 | 0 | 4 | 2 |

TRAIN 的 17 条有效损失全部对应新增搜索超时；新类型只来自 283203 的 target0/1。该 TRAIN 已知正例覆盖率没有提高：新类型不必出现在采到的参考里，不能据此判无效或声称完整解集覆盖。

DEV 没有任何条件提高有效槽数；原来有有效解的全部 24 条件均减少有效槽，其余 12 条件有效数不变。只有以下五条件各多一种已分类有效类型，但每条件都由四条有效降至两条：

| 条件 | 新增类型（原已有 middle–middle） | 有效槽变化 |
|---|---|---:|
| 283267 target2 | positive_y–positive_y | 4 → 2 |
| 283269 target1 | over–over | 4 → 2 |
| 283272 target0 | negative_y–negative_y | 4 → 2 |
| 283273 target0 | over–over | 4 → 2 |
| 283274 target2 | positive_y–positive_y | 4 → 2 |

空间图并非在所有条件上只容纳 middle–middle；这些实际返回的替代类型已给出反例。但这不证明其它参考类型也能通过当前可见自由图，更不证明降低重复可以免费提高质量。完整 12 TRAIN 与 36 DEV 逐条件、逐候选类型/失败、按父差值分别保存在 [TRAIN 配对 JSON](observed_two_row_spatial_penalty_v1/TRAIN_PAIRED_ANALYSIS.json) 和 [DEV 配对 JSON](observed_two_row_spatial_penalty_v1/DEV_PAIRED_ANALYSIS.json)。

## 搜索与预处理的实测成本

| 设置 | 请求中位秒 | p95 秒 | 全部请求连续 wall 秒 | 其中 field 预处理秒 |
|---|---:|---:|---:|---:|
| TRAIN edge | .9082 | 1.4352 | 9.5408 | 0 |
| TRAIN spatial | 5.9805 | 6.7941 | 51.5789 | 2.6122 |
| DEV edge | .9144 | 2.1343 | 33.2439 | 0 |
| DEV spatial | 6.1425 | 6.7349 | 160.4509 | 7.8555 |

请求时间包含实际 IO/hash、反投影、定位、建图/接入、field、A*、原 proxy、保存封存与检查；共同 TRAIN 拟合另计。field 预处理位于原 A* 两秒计时器外，确实增加计算，但只占 DEV 两臂请求总时间差 127.2070 秒的 **6.18%**（TRAIN 6.21%）。不能将主要变慢解释为 field 的距离计算。

56 次 DEV 超时实际扩展节点范围 7232–8128、中位 7936，全部尚未达到 20000；记录搜索时间范围 2.0006–2.0173 秒、中位 2.0102 秒，总 112.5661 秒。原每 64 次扩展检查 deadline 的方式未变，少量超过两秒如实保留。超时对应的 field 合计 6.5489 秒已计入上表，不属于上述搜索秒数。

这一证据支持下一项**同算法、两臂共同编译实现**的工程可证伪对照：先验证邻居、代价、tie 次序、几何和返回路径一致，再重测同 20k/2s 上限。它尚未运行；不能预先保证编译后成功，不能只加速新一臂或把解释器速度写成研究机制优势。本轮不变更参数、不自动延长搜索、不补发候选。

## 作业、哈希与复现

| 阶段 | record PID / child | UTC 起止 | 内部 pipeline 秒 | 外层进程秒 |
|---|---|---|---:|---:|
| TRAIN | 568439 / 568442 | 21:14:07.146444 → 21:15:11.709278 | 64.324 | 64.562834 |
| DEV | 570899 / 570901 | 21:16:46.823662 → 21:19:30.955911 | 163.816028 | 164.132249 |

两次均 root 以 CPU core1 启动，wrapper 要求单核 affinity、OMP/BLAS/MKL/NUMEXPR 各 1，`CUDA_VISIBLE_DEVICES=''`；GPU 小时 0。TRAIN 拟合在共享两臂前执行一次，DEV 重新拟合并验证同一 canonical identity；DEV 实测拟合 2.809648 秒，不重复记旧 A*32 拟合成本。

实际 run：`/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_spatial_penalty_v1/{train_preflight,dev}`。命令与恢复策略完整保存在 [TRAIN status](observed_two_row_spatial_penalty_v1/train_preflight.status.json)、[DEV status](observed_two_row_spatial_penalty_v1/dev.status.json)；fresh-only 阶段不应重放。

- TRAIN 原报告 SHA256：`1571670e2523976e04aabe7976e64388cfded8b7c124533238799123392e719b`。
- DEV 原报告 SHA256：`b6e92322e53417e039153900b276b55134228a23e14dd26f916b5f1ee509f7b1`。
- 原 A*32 DEV 报告 SHA256：`6c108dff3c01dbf54053f80ea9eba563e57e654de6959466e360b00e2218e06e`。
- 同步 TRAIN 140 文件、DEV 199 文件，原字节/大小逐项核验；两份索引分别为 [TRAIN](observed_two_row_spatial_penalty_v1/TRAIN_SYNC_SHA256_INDEX.json)、[DEV](observed_two_row_spatial_penalty_v1/DEV_SYNC_SHA256_INDEX.json)。123 份 NPZ 仅放 ignored `runs/observed_two_row_spatial_penalty_v1/`，小证据及实际源码在 reports 同名目录。
- 新离线分析的 8 项本地测试真实通过；候选预算损坏、有限目标错误与搜索失败分离、unknown 保留、按 ID 配对、POSIX 路径映射/hash 篡改均覆盖。实际读取全部封存数组后通过，0 新规划器调用。分析没有重新进行几何判定，只复核原固定 checker 保存的决策及其池哈希。

本地复算入口：

```powershell
& 'F:/ProgramData/anaconda3/python.exe' -m pytest tests/test_spatial_penalty_saved_analysis.py -q
& 'F:/ProgramData/anaconda3/python.exe' -m scripts.analyze_observed_spatial_penalty_astar --run reports/observed_two_row_spatial_penalty_v1/train_preflight --binary-root runs/observed_two_row_spatial_penalty_v1/train_preflight --output <fresh-train-analysis.json>
& 'F:/ProgramData/anaconda3/python.exe' -m reports.observed_two_row_spatial_penalty_v1.analyze_saved_dev
```

DEV 派生脚本拒绝覆盖已存在输出；如需复算使用独立副本目录，不能删除或覆盖原封存产物。
