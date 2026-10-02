# 多任务观测集合回归兼容协议

这是原有普通集合回归强基线的任务适配，不是新增机制贡献。当前实现尚未在六任务数据上训练；不能将下面的接口验证写成方法有效性结果。

## 输入、参考和终点

模型继续仅使用真实冻结 Qwen 的 RGB+语言隐藏状态、当前夹爪位姿与开合、当前 RGB-D 和相机参数。任务名称和父场景编号只用于采样与报告，不是网络输入。监督来自成功记录的三维夹爪轨迹与开合序列。

新增可选 `endpoint_mode=free_offset`：从原有输出层预测全部后续三维点相对当前末端的自由偏移，允许 lift/lid 等任务的终点位于可见表面之外。原几何编码器仍提供观测条件；不再把最终点限制在观测表面 anchor 的逐坐标 ±5 cm。输出层及全部几何和路线参数继续训练，没有新增参数。第一点和初始开合保持真实当前状态。

新六任务使用 `grounding_weight=0`，禁止沿用 reach 表面末点 attention 辅助监督，也禁止套用 reach TipValid 选模或已停止的草稿修正模块。历史默认 `surface_anchor` 的模块、随机初始化和前向公式保持不变。

## 数据和实时门禁

`observations.jsonl` 仍严格只有 `id,parent_id,split,image,instruction` 五个键。父场景内多条语言是同一目标的改写，不能当成独立物理布局；各改写共享当前观测和 0–3 条正参考。`semantic_targets=None` 保持为空；失败采集不监督“无解”。

新训练自动识别同目录 `snapshot_manifest.json`，要求四个快照文件的 SHA256 一致，检查注册父场景只属于 TRAIN/DEV_MODEL，并重新调用机械哈希审计。若后续采集发现任一所选任务存在跨划分重复，旧快照也会在下一次模型使用前被拒绝。该审计只读取其他角色的注册表和已保存机械哈希，不读取其图像、路径、目标或验收结果。

没有初始观测的 setup failure 留在快照采集分母，不伪造模型输入。已有观测但零参考的指令保留于预测与报告；只在训练损失和参考误差的可评价分母中排除。当前模型使用门禁及原请求/失败计数写入运行目录。

## 采样、度量与恢复

可选 `sampling_mode=task_parent_language` 依次均匀抽取：有正参考的训练任务、该任务的父场景、该父场景的语言改写。这样新增语言改写不会增加该任务或父场景的期望曝光权重。默认 `uniform` 调用与历史完全相同。随机采样状态写入 checkpoint；断点恢复不得更换采样或指标聚合策略。

可选 `metric_aggregation=task_parent` 先平均父场景内语言，再平均任务内可评价父场景，最后等权平均可评价任务。选择协议名为 `task_parent_reference_ADE_v1`，以该宏平均 `candidate_matched_ADE_m` 最小选模。原始按指令平均值、按父和按任务明细同时保留。语义指标、连续碰撞、路线类型、机器人执行和原任务成功率均未验收，不由参考 ADE 或事件匹配推断。

输出 H24 三维路径与开合概率。报告候选匹配/参考匹配 ADE、末端误差、相对参考的事件状态和压缩事件序列匹配，并明确参考分母。保留 best 和 final checkpoint、全部对应预测。每请求 K 个完整路径状态，无内部额外草稿更新或筛除。

新任务建议参数片段：

```text
--endpoint-mode free_offset --anchor-mode soft --grounding-weight 0
--sampling-mode task_parent_language --metric-aggregation task_parent
--selection-metric reference_ADE --refinement-mode none
```

## 验证状态

- 本地 8 个无 Torch 元数据测试通过（0.60 秒），包含历史均匀随机流一致、三层采样比例与恢复、宏平均分母、快照修改拒绝、采集后新增跨角色重复的实时拒绝，以及真实旧 new32/new64 快照的自动兼容与显式错传拒绝。代码编译与差异空白检查通过。
- 另外 5 个 Torch CPU 测试已实现、待固定发布版本实际执行：默认前向逐位一致、自由终点/全部模块梯度与输入白名单、变长改写和空参考的真实 loader/evaluator 组合、配置拒绝、完整微型训练与中断恢复逐位一致。测试隐藏状态是明确标识的合成 fixture，不是真实 Qwen 实验。
- 实际六任务快照已由独立采集流程在固定 `5fb74b` 源码完成：`data/observation_multitask_prefix24_v1`，24 父场景、72 条正参考、96 条语言输入，其中 TRAIN 为 12 父、36 条参考。快照 manifest SHA256 为 `0d62b1e52b8729a4f129b1dbc0207307f5373280604f6cdb0c6b22069be7bffa`。该流程的 TRAIN 审计报告 36/36 参考保留事件；12 条参考终点不存在逐坐标 ±5 cm 内的已采样观测点，说明 hard surface endpoint 表示的容量限制。此证据不能证明 soft anchor 在所有情况下不可行。
- 真实 Qwen 缓存和训练仍待队列执行；本报告未声称这些任务已取得生成或执行成功结果。
