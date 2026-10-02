# 两排路线的独立评价协议

在新的正式模型预测之前登记。`scripts/evaluate_observed_two_row.py` 仅接受已生成路径和独立标签；不调用模型、模拟器、规划或修路。它复用旧障碍评价器的目标3cm、起点5mm、恒定reach事件以及完整线段对全部物理箱体2cm间隙检查。旧单障碍评价器默认调用和指标保持原样。

路线类型使用采集器原有 `crossing_signature(path, route_config)`，依据实际路径依次穿越两排的关系判定 negative_y/middle/positive_y/over。这是有限障碍穿行关系，不是完整同伦等价。配置、实测四柱几何和目标必须匹配；1微米仅沿用采集时物理尺寸读回比较容差，不改变2cm碰撞边界。guide编号不参与预测类型判断。

每次提交的K条全部计入分母；重复、非有限坐标、缺事件、错误目标或碰撞均不补发。Valid unknown仍计TipValid，但不增加已分类不同类型数。ReferenceCoverage仅覆盖已知正参考类型；没有采到的类型不是负例，新生成的有效已知类型也可计入Unique。未知/空参考分母明确标为不可计算，不填0。可选逐路径分数只能从同一K中选择，缺分数SelectedTipValid为null；它不变成oracle AnyValid。

指标继续命名为TipValid/UniqueClassifiedTipValid，完整机器人Valid/SelectedValid仍为null。只检查新增物理柱与末端线段，不能据此认证手臂、桌面、IK或执行成功。路径长度照实报告，没有偷偷增加长度阈值来删除大绕行。

纯函数接口为 `scene_metrics(paths, opened, current, geometry, specification, reference_types, route_config, scores=None)`。全部几何/目标/类型配置仅供预测后评价，不能传入观测模型forward。正式导出/评价入口另需核验允许的父场景、角色与来源哈希；本纯函数不自行打开任何数据角色。

本地10项针对性检查通过：实际两排关系、重复、碰撞、未知有效、超出参考池的新类型、over、错误目标/起点/事件、非有限槽、阈值和几何失配。与原默认评价器逐候选Validity一致。旧默认self-test及既有reserved-obstacle回归也通过（合并11项）。首次新测试中的共享列表别名导致目标失配用例未实际篡改单一字段，已修正fixture；评价代码未为过测试放宽标准。旧self-test首次使用module入口缺少历史scripts导入路径，改用其实际脚本入口后通过。

当前没有新模型预测结果，本文件不是效果报告。
