# Min-SNR：500 步 TRAIN 配对筛选的实际结果

2026-10-03。科学源码 `f72e1aa11a37df2001877035fbfde9a4525c0aa0`。两臂从同一个真实 step12000 checkpoint 恢复，分别完成额外 500 步至 12500；uniform 与 Min-SNR γ=5 的初态、父场景/参考/t/epsilon 四条抽样流实际严格一致。三项预登记筛选门均通过。这是 **一个训练种子的 TRAIN 内部强基线修复筛选**，没有 DEV 选择、没有核心方法成立或泛化结论。

## 全池结果与收益来源

每臂独立评价同一组 95 父、285 指令、K=4，共 1140 个候选。两池没有合并。下表全部由封存 `per_scene.json` 算术汇总；完整 285 条件和 95 父差异见 [PAIRED_ANALYSIS.json](paired_analysis/PAIRED_ANALYSIS.json)。`TipValid` 沿原 tip-only 2cm/目标 3cm 协议，不包含机械臂、IK、执行和完整任务成功证书；通用 Valid/SelectedValid 不补填。

| 指标 | uniform | Min-SNR γ=5 |
|---|---:|---:|
| TipValid | 470/1140，41.2281% | 494/1140，43.3333% |
| TipClear | 480/1140，42.1053% | 497/1140，43.5965% |
| 语义目标正确 | 1114/1140，97.7193% | 1133/1140，99.3860% |
| AnyTipValid | 237/285，83.1579% | 253/285，88.7719% |
| 已分类有效不同类型总数 / 每指令 | 212 / 0.743860 | 234 / 0.821053 |
| 已分类有效重复候选数 | 65 | 23 |
| 类型未知的有效候选数 | 193 | 237 |
| 已知参考类型覆盖，277 个有定义指令的均值 | 0.117371 | 0.125159 |
| 候选→最近正参考 ADE | 97.4733 mm | 97.0212 mm |
| 正参考→最近候选 ADE | 145.8051 mm | 140.8861 mm |

新增 24 个有效候选中，双方目标均正确的配对槽净增 17 个；语义发生改变的组净增 7 个（原错→正确且有效新增 8 个，原正确→错且失效 1 个）。因此提升包含几何成分，不能全部归于目标定位。起点、事件和有限值失败均为零；碰撞候选从 660 降至 643。

同噪声槽位中 239 个由无效变有效，215 个由有效变无效，255 个两边有效、431 个两边无效。相同槽位/初始噪声不保证语义路线身份相同。285 指令的 Tip 数变化为 110 改善、76 持平、99 变差；父级汇总为 **42 改善、10 持平、43 变差**。总均值的小幅增加不代表大多数父场景改善，也尚无独立种子支持。

分类结构也有取舍：已分类有效候选总数从 `212+65=277` 降至 `234+23=257`，减少 20；未知类型有效候选增加 44，二者合计产生 Tip 净增 24。已分类 unique 增加且重复减少属实，但不能把未知类型当成新类型或自动声称覆盖全面提升。8 个没有已知参考类型的条件继续保持覆盖未定义；不完整正参考不等于全部解集。

## Teacher 与自由生成不能混为一谈

Teacher 固定 6 个 TRAIN 输入、5 个 t、每臂 30 次去噪调用。下表保留全部噪声层的 body RMSE；不是只挑通过筛选门的低 t。

| t | uniform，mm | Min-SNR，mm | 判断 |
|---|---:|---:|---|
| 0 | 37.749 | 32.981 | 改善 |
| 25 | 72.301 | 66.841 | 改善 |
| 50 | 138.820 | 137.181 | 小幅改善 |
| 75 | 123.039 | 133.716 | 变差 |
| 99 | 128.978 | 137.026 | 变差 |

端点 teacher 误差在五层均改善，但高 t 路径主体误差变差。两臂训练目标加权不同，不能用各自 train loss 的数值大小替代自由生成质量。固定低 t 门通过只支持继续有限验证。

只读核验的历史普通集合回归 step12000 在同一有序 285 条件上为 Tip 989/1140（86.7544%）、TipClear 1080/1140、known unique 415、候选→最近参考 ADE 17.799mm。当前扩散自由 TRAIN 拟合仍有很大差距。该对照的架构、目标、累计训练量和计算不同，**只用于定位拟合差距，不作为同预算新方法比较**；来源与封存字节核验见 [ORDINARY_TRAIN_FIT_REFERENCE.json](paired_analysis/ORDINARY_TRAIN_FIT_REFERENCE.json)。

## 全候选图与检查边界

[全部父差异及 teacher 图](paired_analysis/all_parents_and_teacher.png) 覆盖全部 95 父和 5 个 t。[六图总览](paired_analysis/CONTACT_SHEET.png) 中的 5 个条件按完整表的最大 Tip 增益/损失、最大 known 增益/损失规则选取，规则及并列 ID 排序固定记录在 derivation receipt；它们是极端诊断，不能用于估计发生率。

- [283206 / target0](paired_analysis/two_row_reach_283206_target0.png)：Tip 0→4，但四条均为 unknown；高弧和回转仍完整保留。
- [283222 / target2](paired_analysis/two_row_reach_283222_target2.png)：Tip 0→3，其中 known 1、unknown 2。
- [283201 / target0](paired_analysis/two_row_reach_283201_target0.png)：Tip 3→0、known 2→0；uniform 的极长回路未裁剪或排除。
- [283207 / target0](paired_analysis/two_row_reach_283207_target0.png)：Tip 3→0，目视相近也不改原碰撞标签。
- [283209 / target2](paired_analysis/two_row_reach_283209_target2.png)：known 0→2，同时 Tip 3→2，展示覆盖与通过率取舍。

每图两臂各全部 K4、同一坐标范围、XY/XZ 投影；没有新几何 checker、raw 场景读取、修复、补样或筛选。图中极端长弧即便通过原 Tip 协议，也不能升级成完整机器人有效路线。实际逐图目视范围与图像 SHA 见 [VISUAL_QA.json](paired_analysis/VISUAL_QA.json)。

## 真实预算、失败与恢复

每臂额外 `500×32=16000` 输入抽样、`500×32×4=64000` 训练路径状态；两臂合计 128000。每臂实际 791 次 geometry 调用、11930 次 denoiser 调用、500 次 optimizer 更新，包含训练、固定 teacher 和固定 TRAIN 自由生成；新增 Qwen 编码/DEV 调用均为零。每臂自由生成 285 请求×4，40 个 DDIM 步。数据没有因成功/失败被替换。

GPU 外层工作实际耗时 uniform 77.962665 秒、weighted 72.297482 秒，合计 **150.260147 秒 / 0.041738930 GPU 小时**。uniform 科学内层 77.066784 秒、weighted 70.898538 秒已包含在外层，不重复相加。CPU 外层 tests 4.901302、parent inspect 2.935982、只读 seal 0.774568、compare 0.556431 秒另列；旧失败诊断成本在原家族中保留，不混入这两臂训练。

失败链没有覆盖：

1. 原 755 CPU 测试因 optimizer CPU storage alias 出现 14 pass / 2 fail，没有启动真实训练；f72 对 optimizer 恢复做最小 deep-copy 后，18 个实际测试全部通过，完整状态/RNG/四流严格条件未放宽。证据在独立 alias-fix/旧 validation-failure 家族。
2. 本次 f72 uniform 科学训练实际完成 12500；生成 runner 的全局文本替换误把后验约束 `16000` 改为 `18000`，导致外层 exit1。此为编排错误，不是训练失败或预算改变。原 runner、failed status/log、真实 summary/权重/账本全部保留。
3. 新 recovery 只读核验正确的 500/16000/64000 及 28 个原件并封存 uniform，0 新模型/PT load/checker/raw；未重训。weighted 和 compare 随后由 root 分别真实启动并 exit0。数值 AST 对照及真实 summary fixture 在 `local_derivation/`，不删除首次失败测试收据。

PT 留远端 SHA 索引；NPZ 位于 ignored `runs/synced_observed_diffusion_minsnr_probe_v2`，源 Python 以 `.py.txt` 归档避免测试收集污染。全部原件、派生来源和本地索引一并可追溯。本轮不向 MAIN 增加技术筛选行。

下一步只提出 [12500→15000 有界续训方案](CONTINUATION_PROPOSAL.md)，本报告未实施或启动它；不能据这一次 TRAIN 筛选声称方法贡献已成立。
