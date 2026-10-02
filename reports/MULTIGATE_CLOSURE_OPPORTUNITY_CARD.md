# 单门关闭共同失效：TRAIN机会审计预注册

2026-10-03。状态：**实现与9项本地测试完成；尚未在正式TRAIN768执行，无新训练或GPU任务。** 这是下一步方法决策的低成本否证，不是方法结果。父线程冻结源码后才允许CPU1正式运行。

## 问题和先前证据

静态类型不同的路线可能共用同一关键通道，约束变化后同时失效。问题是这种缺口是否超出传统几何多样性选择就能解决的范围，而不是先把最大覆盖适配称为创新。

受控数据有1–16种有限开口序列，但静态普通集合回归已很强。真实观测新96的已知参考每指令均少于K4，只有28/288指令有有效重复且遗漏已知类型，标签辅助替换机会仅0.104类型/指令；它目前不足以直接检验大量可行类型超过预算。六任务参考没有路线类型或同父约束变化标签。已有max覆盖补全、局部门控和局部读取更新的负结果全部保留，不在本实验重启这些模块。详见[覆盖风险卡](REFERENCE_COVERAGE_RISK_CARD.md)、[方法近邻增补](RELATED_WORK_TASK_RESPONSE_ADDENDUM.md)。

## 固定范围与输入

- 注册配置：`configs/multigate_closure_opportunity_train768_v1.json`。
- 数据固定既有`multigate_v1_partitions/development.npz`，SHA256 `f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057`；只用全部768个TRAIN父，每父一个原场景。
- 既有普通saturation K4 seed0 best，step1500，checkpoint SHA256 `b2cb51ae0de50785a12f129c2ff12ecc981a9df826b5ee7ae1462de6366f102c`。这是历史DEV选模的既定模型，不重新选择或读取DEV原始内容。
- 读取split、parent、scene ID元数据后，按NPY行流仅解码TRAIN几何、参考和模式。ZIP为跳过行可能内部解压字节，但不将DEV负载转为数组或检查其内容；拒绝含score/calibration/locked角色的归档。输入文件原始字节hash读取不是样本分析。
- 不调用原`load_dataset`，因为它会先加载整个归档；不改生产`close_opening`。审计专用函数允许关闭墙的最后一个门。
- 对每父枚举**所有present门的单门关闭**，包括无解变化。每次变化独立地从原布局关闭一个门，不累计关闭。

## 五种输出，三种性质

| 名称 | 实际条件与预算 | 可作何种结论 |
|---|---|---|
| joint4_actual | 真实已训练K4的一次CPU前向，完整生成4条 | 现有普通模型的实际TRAIN响应 |
| joint4_prefix2_zero_adaptation | 同一次K4输出的前2条，无重新前向/适配，仍收费4条 | 仅未适配截断诊断；**不能当合理训练K2基线** |
| reference_farthest_k2 | 已知正参考池R条，检查全部C(R,2)二元组，选最大整路径平方RMS距离 | 传统、特权参考池选择控制；不是生成K2 |
| reference_geometric_dpp_k2 | 同池，单位质量RBF核，sigma=.25m，固定K2 MAP | 与最远对数学等价，不宣称两份独立实证 |
| reference_response_oracle_k2 | 同池、所有单门关闭的精确验收真值，选在仍有解变化下AnyValid最大的固定二元组 | 参考池内的标签辅助机会量；不是模型可实现增益或全部连续路径上限 |

二元组在具体关闭哪扇门之前固定选定；不允许每次变化重新挑最佳参考。R=1时重复唯一正例到两个槽，合法变体/重复不被自动判错。R>=2枚举不重复参考的所有无序二元组。所有目标并列以原参考索引字典序固定裁决，不能按结果选tie。已知参考模式必须恰好覆盖当前两墙present门的笛卡尔积，并通过未修改的完整线段checker；这只是该受控有限类型集的完整性，不推广到机器人连续解集。

单位质量RBF核有`L_ii=1`，K2行列式为`1-exp(-d_ij²/sigma²)`，对平方距离严格单调。因此直接使用同一精确最远对，避免数值下溢制造伪差异。该事实必须与DPP/最远对结果同时报告。

## 指标、分母和停止规则

保留所有关闭事件的逐父/逐变化明细、实际4条路径、每候选有效性/碰撞/模式、参考响应矩阵和选中索引。静态Valid/AnyValid/UniqueValid与变化后相同指标分开。按R<2、R=2、R>2分层。

主要条件指标先在每父**仍有解的关闭事件**内平均，再等权平均父。无解事件计数、包含无解的AnyValid另报；若某父所有关闭均无解，保留该父和事件，条件指标为null，不凭空造训练失败目标。零条件分母不填0。所有阈值保持原受控checker，机器人3cm/2cm协议不混入。

**5个百分点只用于TRAIN低成本筛选，不是显著性、可实现收益或最终方法证据。**

1. 主机会量：参考响应最优K2减传统几何最远K2的等父AnyValid。低于.05，不支持在当前数据中开发共同失效目标以超越几何选择。
2. 辅助机会量：参考响应最优K2减未适配prefix2。若仅这一项达到.05，只能说明需检查真正K2训练；不能将截断弱基线的缺口归于新机制。
3. 两项都低于.05则停止此设置方向，不根据DEV更换门、变化分布或增加约束。
4. 任一机会保留时，**下一步先以同预算真正训练普通K2 saturation基线**，之后才决定是否存在可检验的方法配对。当前未授权或启动这项训练。
5. joint4与参考K2的差仅作异预算定位，不能据此宣称方法胜利。没有补全、重算、筛出隐藏候选或学习评分器。

## 实测与恢复

本地第一次收集测试因Python旧版本不支持订阅表达式中的star语法而失败；改为普通`np.concatenate`后，9项测试实际通过，0.68s。覆盖最后门关闭、无解保留、固定pair而非逐变化oracle、DPP等价与tie、完整参考拒绝、TRAIN行读取、locked/cross-role拒绝以及全部present门响应一致性。该失败不影响原模型/数据。

冻结后由`record_job.py --resume-strategy fresh-output`记录PID、日志、退出码与重新运行命令。审计输出必须新目录；无训练恢复或`--resume`。正式CPU结果、实际source/config/input/output SHA及时间只能在执行后填入，不以测试通过代替实测。

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  scripts/audit_multigate_closure_opportunity.py \
  --config configs/multigate_closure_opportunity_train768_v1.json \
  --output /home/wzy/dpvlm/route_set_v1/runs/multigate_closure_opportunity_train768_v1/audit
```

以上命令待冻结后执行，不是已经完成的复现记录。实际物理路径以配置为准，不使用新任务语言、真实观测或机器人执行的表述。
