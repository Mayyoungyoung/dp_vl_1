# 六任务旧批 DEV12：固定模型迁移实际结果

2026-10-03。冻结模型在此前未用于本项目模型训练、选模或质量指标的 12 个同任务父场景上实测完成。原 best 三种子的宏路径误差平均改善 4.74%，但仅 2/3 种子改善；固定 step1500 的宏路径误差反而退化 1.49%。因此本次提供有限的 best-checkpoint 迁移支持，不能称跨种子、训练阶段或所有父场景稳定有效；不再据此追加种子或调参。这个常规 attention 辅助仍不是核心方法贡献。

数据角色仍为原 DEV_MODEL；只作为未使用父场景的迁移开发证据，不能称新任务、OOD、IID 保证或最终 locked 测试。语义、碰撞、完整执行和成功率均未测、保持 null。事件序列准确率只说明参考 open 状态序列一致，不能视为交互成功。

## 实际运行与数据完整性

源码 `fe564531a0d66a95ca055f10a90a3f49e334bc37`，launcher SHA `48852b17323e8d40e8658e882c97fa447eafcc82b2e7f5a2ce173740b7c4fa1e`。唯一 SSH session8018 正常 exit0；服务器完整 13/13 测试通过，无跳过。原 12 父/36 请求尝试全部闭合，逐条复验后 36 条正参考、48 条语言输入、0 个无参考父；没有删难例或替换父场景。登记机械对照 120 行无重复；运行前后完整只读门禁、源文件 hash 检查均通过，未打开 locked 父内容。

四阶段 record_job 均 exit0：测试 PID473502/child473503，导出 PID473510/child473526，Qwen PID473532/child473533，CPU 固定评价 PID473610/child473611。每个阶段均记录实际 UTC、命令、日志和恢复策略。物理 GPU1/35%、CPU affinity `[0]`、Torch/OMP 线程1；Qwen 在17:57:27.754UTC完成并释放GPU，评价在17:57:40.875UTC完成。

全部12个 checkpoint 的实际加载 step/config、summary/config/PT SHA、原训练源码、当前推理源码和 Qwen cache 配置通过核对。best 使用旧开发集已经选出的 checkpoint；last 固定1500；没有重训、再选模型、推理筛选或跨种子合池。原权重在服务器和本地均逐文件 hash 一致，全部预测二进制已同步。

## 宏指标：任务内父等权，六任务等权

下表为三种子均值 ± 样本标准差；误差单位 cm。正差表示辅助版误差更大。best 与 last 严格分开。

| 固定来源 | 方法 | ADE cm | 末端误差 cm | 事件状态准确率 | 事件序列准确率 |
|---|---|---:|---:|---:|---:|
| best | ordinary | 8.9879 ± 0.2921 | 11.5124 ± 0.5655 | 98.9352% | 100.0% |
| best | event_supported | 8.5621 ± 0.2621 | 10.6636 ± 1.0125 | 98.8310% | 100.0% |
| last | ordinary | 8.9621 ± 0.2494 | 11.3679 ± 0.3777 | 98.9680% | 100.0% |
| last | event_supported | 9.0961 ± 0.7336 | 10.8122 ± 0.7691 | 98.8754% | 100.0% |

| 来源 | seed | ΔADE cm | Δ末端 cm |
|---|---:|---:|---:|
| best | 0 | -0.7809 | -2.1173 |
| best | 1 | +0.0914 | +0.8869 |
| best | 2 | -0.5879 | -1.3160 |
| last | 0 | -0.7036 | -1.6840 |
| last | 1 | +0.0845 | +0.1305 |
| last | 2 | +1.0210 | -0.1138 |

原 best 的配对 ΔADE 为 −0.4258 ± 0.4582cm，Δ末端为 −0.8488 ± 1.5556cm；seed1 两项均退化。last 的配对 ΔADE 为 +0.1340 ± 0.8634cm，Δ末端为 −0.5558 ± 0.9847cm。事件状态均值分别退化 0.1042/0.0926 个百分点；序列准确率全部100%。三种子样本不足以作统计显著性保证。

## 全部任务与父场景差值

每行均对同 seed 配对后，再对三 seed 平均；没有从最好的 seed 或父场景选展示。

| 任务 | best ΔADE cm | best Δ末端 cm | last ΔADE cm | last Δ末端 cm |
|---|---:|---:|---:|---:|
| pick_and_lift | -1.1354 | -1.6557 | -0.2420 | -0.5452 |
| pick_up_cup | +0.1779 | -0.1147 | +0.0421 | +0.3072 |
| push_button | -0.3803 | -0.8397 | -0.1766 | -0.6985 |
| reach_target | -1.0618 | -1.9307 | -0.4553 | -1.0833 |
| slide_block_to_target | -0.0978 | +0.0100 | +1.4592 | -0.5579 |
| take_lid_off_saucepan | -0.0574 | -0.5619 | +0.1764 | -0.7568 |

| 父场景 | best ΔADE cm | best Δ末端 cm | last ΔADE cm | last Δ末端 cm |
|---|---:|---:|---:|---:|
| pick_and_lift_291016 | -1.8663 | -0.5896 | -0.5893 | +1.2054 |
| pick_and_lift_291017 | -0.4046 | -2.7219 | +0.1054 | -2.2958 |
| pick_up_cup_321016 | +0.1757 | +0.4249 | +0.6607 | +0.2662 |
| pick_up_cup_321017 | +0.1801 | -0.6544 | -0.5765 | +0.3481 |
| push_button_301016 | -1.0604 | -0.8798 | -0.2951 | -1.6177 |
| push_button_301017 | +0.2999 | -0.7996 | -0.0581 | +0.2208 |
| reach_target_281016 | -2.9128 | -5.4957 | -2.2560 | -6.5087 |
| reach_target_281017 | +0.7893 | +1.6343 | +1.3453 | +4.3420 |
| slide_block_to_target_331016 | -0.3314 | -0.0814 | +1.4640 | +0.5075 |
| slide_block_to_target_331017 | +0.1358 | +0.1013 | +1.4544 | -1.6234 |
| take_lid_off_saucepan_311016 | -0.3109 | -0.9133 | -0.2943 | -1.8100 |
| take_lid_off_saucepan_311017 | +0.1961 | -0.2105 | +0.6472 | +0.2964 |

best 的三种子平均 ADE 在6/12父改善、6/12父退化，只有3/12父在三个种子中都改善；last 同样只有6/12父平均改善。reach_target_281016 改善较大，但同任务另一父281017稳定退化。last 中 slide_block_to_target 两父都退化，任务平均 ADE 增加1.4592cm。不能将宏均值归因于所有交互目标定位均改善。

## 成本、预算与复现

本次无新增训练。48个真实 RGB+语言请求只编码一次：cache内部加载2.753秒、内部总6.182秒；包括Python导入/进程启动的完整记录阶段11.145秒，即占用0.003096 GPU小时。单次编码记录中位55.92ms/P95 62.64ms，峰值显存4,279,413,760bytes。这个时间不包括路线头，不能作完整端到端规划时延。共享RGB-D读取预处理0.342秒；十二份CPU评价每份48请求合计各约0.60–0.70秒，完整CPU评价阶段13.072秒含权重核验/加载。

每个模型每请求K4且一次前向，没有筛选；12×48×4=2304个已评价完整候选路径状态分别记账，绝不合并为更大的oracle池。原训练和原cache成本已在对应历史run单独登记，这里只计新增迁移费用。

实际命令为 `taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/fe564531a0d66a95ca055f10a90a3f49e334bc37/scripts/launch_observed_multitask_legacy12_transfer_v1.sh fe564531a0d66a95ca055f10a90a3f49e334bc37`。完整pipeline不允许覆盖重跑；需要复核时使用已记录的独立stage命令和新的输出目录，保持原PT与导出不变。

实际证据：[完整十二源结果](observed_multitask_legacy12_transfer_v1/evaluation/summary.json)、[全seed/任务/父分析](observed_multitask_legacy12_transfer_v1/artifact_analysis.json)、[二进制与元数据SHA索引](observed_multitask_legacy12_transfer_v1/artifact_index.json)、[导出receipt](observed_multitask_legacy12_transfer_v1/snapshot/export_manifest.json)、[实际cache receipt](observed_multitask_legacy12_transfer_v1/actual_cache_receipt.json)。本地只读分析脚本为 `scripts/analyze_legacy_multitask_transfer.py`，实际脚本SHA写入分析JSON；服务器真实运行源始终为fe564。

研究判断：保留常规辅助在旧best选模策略下的有限迁移收益，也保留last和逐父退化；当前不再追加种子或调参。下一步先在正式两排真实多模式数据建立同信息的普通强基线，不能把此辅助或查询结构改名为新方法。
