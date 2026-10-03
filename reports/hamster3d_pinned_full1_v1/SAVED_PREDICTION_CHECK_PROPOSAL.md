# 下一步建议：仅已保存 TRAIN0 的技术验收（尚未执行）

建议root批准后独立运行一次CPU、零forward、零规划搜索的原checker检查。对象固定为当前已封存4个world点及4个官方动作，K=1；先绑定原输入/结果/NPZ/source SHA，再读取此单个TRAIN样本原route_config/verification与known-positive types。不得打开其它TRAIN请求或DEV/reserved，不重新生成或重选点。

1. 主结果直接把4点当原始折线提交 `scripts.evaluate_observed_two_row.scene_metrics`；该API允许可变点数，保持原2cm tip线段、3cm目标、5mm起点和reach事件标准。动作映射明确：Close→0、Open→1、None沿用前态，因此本原序列为[0,0,0,1]；保留原open current=1。先写映射receipt再打开验收标签，不能为通过而消除事件。
2. **不添加current起点、不把点1改成目标终点、不删除点2..4、不转换为H24平滑路径、不修复碰撞。** 现首点距离current约42.43cm是已知表示边界，应如实单列原始折线起点失败；官方系统可能在waypoint之间另需执行器连接，但这不能由本次模型候选自动获得信用。若将来研究current→first连接段，须独立预登记adapter/完整新增段成本与几何验收，不能覆盖本原结果。
3. 原始目标正确率必须针对最后第4点，与原任务指定black sphere比较；可以附加“4点各自到指定目标中心距离”的描述表解释关键点语义，但不能把最靠近目标的点替换终点或升级K。质量字段原归档仍null，新check独立目录记录这一次posthoc TRAIN技术验收。
4. 报告全部唯一候选的finite/format、起点、语义终点、事件、全线段净空及known/unknown类型，及各失败因素；known-positive coverage是参考下界，非全部解。保留无学习评分器、不支持SelectedValid、无全臂/IK/执行证明。任何通过或失败都不改变本次生成预算151forward/151tokens/1candidate及已记成本。

此检查仅用于厘清官方稀疏manipulation-waypoint接口与当前reach路线验收的差异，不作为方法优势或正式系统级机器人成功率。若原候选表示不满足任务，也应报告，不增发技术请求追正结果。
