# 真正K2训练：普通有限预算基线登记

2026-10-03。状态：**代码、配置与本地13项相关测试完成；尚未服务器测试或训练，等待冻结和root GPU队列。** 本轮不是共同失效新方法；[TRAIN审计](MULTIGATE_CLOSURE_OPPORTUNITY_RESULTS.md)已经表明传统几何最远对/DPP在当前单门变化设置达到响应最优上限。无论本轮K2结果怎样，都不恢复joint-risk创新叙述。

## 实际历史设置核验

已读取既有`runs/v2_round2/saturation_k4_seed0/config.json`及报告副本：objective=saturation，3000更新、batch64、lr3e-4、width192、depth3、seed0、每500步静态DEV选模。原训练入口`routeset/train_v2.py`选择`UniqueValid + .05*Valid`，严格提升才替换best；best/last均保留。新launcher在执行时重新读取服务器真实历史config并逐项比较，避免依赖文字记忆。

新基线唯一训练条件改变是`candidates=2`，使用真实max_candidates2的普通SetRegressor和原saturation匹配。参考少于2时允许合法重复，未匹配路线不监督不存在。真实全TRAIN/DEV静态数据与历史相同，数据SHA固定`f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057`；本轮不使用score/calibration/locked。

- 完整训练输入曝光192000，梯度路径槽384000；历史K4为768000路径槽。相同更新/样本曝光不等于相同完整路径预算。
- K2查询参数较少，初始化RNG消费也可能随shape改变。不能声称与K4完整参数初始值逐元素相同。
- 不新增损失、风险组、数据变化分布、注意力模块、修复或候选筛选。
- GPU仅物理1/35%显存。OMP_NUM_THREADS=4，原trainer硬请求4个Torch线程；launcher以系统affinity及显式taskset将自己和后代限制为一个允许CPU，记录实际affinity。训练CPU成本不能与历史4核wall time直接公平比较。

## 唯一新TRAIN后处理

固定原静态DEV-selected best后，由独立`audit_multigate_k2_closure.py`实际一次前向生成768×2条，保留预测、全部父和全部单门关闭。不重选checkpoint，不新增DEV关闭评价。

复用原审计的全部3806个关闭事件、包含无解的分母和hash核验参考选择；既有K4审计与结果不改。当前checker、模型、几何及原审计helper的原始源码SHA必须等于7b496e2实际provenance，否则拒绝。对每父比较actual K2与旧joint4、旧zero-adaptation prefix2和传统/特权参考池结果；异预算与特权信息限制沿用原报告。

“actual K2”必须来自已完成的真正K2训练、原summary best hash及相同超参、实际384000槽；K4 checkpoint、截断数组或改变选模设置均拒绝。

## 冻结后执行与恢复

配置：`configs/multigate_k2_saturation_budget_check_v1.json`。launcher：`scripts/launch_multigate_k2_saturation.py`。固定完整release SHA由root审核后传入；launcher拒绝非同commit目录/配置，核对真实GPU1 UUID、历史config及data hash，再顺序执行原训练器和CPU审计。

输出为`runs/multigate_k2_saturation_budget_check_v1/seed0`和`train_closure_audit`，job在`receipts`。launcher本身只接受fresh输出，避免重复启动。训练中断时使用`launch_receipt.json`保存的原命令追加`--resume`恢复，不改steps/参数；审计中断用记录命令的新输出路径重算。原trainer具备模型、优化器、调度器、全RNG和sampler checkpoint。不会用新launcher重建或覆盖旧K4实验。

本地新4测加原9测实际13通过，1.08s：历史超参/曝光保护、真正K2 checkpoint拒绝K4、所有原closure/无解分母保持、父序与重复身份拒绝。实际服务器CPU测试、GPU开始/完成、数值与hash留待执行后填入。

未来研究建议将依托已经取得真实多模式支撑的新两排观测数据，先确认同父多目标/约束变化能稳定采集；不通过给双墙附加人为risk groups制造有利新结果。
