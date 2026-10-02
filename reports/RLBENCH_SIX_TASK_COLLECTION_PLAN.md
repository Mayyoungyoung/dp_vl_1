# 六任务正式采集准备：144 个预注册父场景

状态：实现与恢复测试准备中，尚未启动正式采集。原始任务演示加入自由前接近提案，数据名称始终为 RLBench-derived；不宣称三个几何不同轨迹就是三种任务路线。

候选任务仅保留已有成功、恢复和成本证据的六项：reach_target、pick_and_lift、push_button、take_lid_off_saucepan、pick_up_cup、slide_block_to_target。旧四任务小批 4 父 12/12 成功，总 433.585 秒；来源以旧作业实际执行副本与 hash 为准，不追认后续 commit 为当时源。新四任务中 cup 三次合计 137.540 秒、slide 122.473 秒，均 3/3 成功和严格记录状态／RGB／物体清单／语言恢复。peg 215.607 秒、stack 473.091 秒，仅保留可采性证据，不纳入本轮正式任务。

每任务 24 个独立父场景，每父 3 个预声明提案，共请求 144 父、432 个路线尝试。父索引前 16 个分配 TRAIN，随后 DEV_MODEL、DEV_SCORE、CALIBRATION、TEST_LOCKED 各 2 个；语言改写共享父划分。全部角色在采集前写入不可变 partition manifest，新种子与现有 pilot 不重叠。角色在磁盘中物理分开，顶层不导出 locked 图像、路线或任务结果。

采用父级工作进程与持久化事务。每个父从独立模拟器进程构建，避免前一父残余状态。每次提案保存完整 pose/open 序列、相机、耗时、原任务成功状态及 hash；失败也是已完成尝试，不能为凑成功条数重采。已完成父在恢复时直接跳过。中断但未完成的父重新构建预注册 seed/variation 的初态，必须与已有 world、RGB、depth、相机、当前状态及语言严格一致，才继续尚未提交的尝试；不一致明确保存恢复阻塞，不覆盖参考。已开始却未完成的尝试留下 interrupted 记录，其耗时和提案代价不被抹掉。

模型入口仍为严格五字段 observations：id、parent_id、split、image、instruction。实际 depth/camera/current 状态从单独 observation 文件读取；supervision 按 id 连接完整成功路径和事件，不向生成条件加入轨迹答案、采集引导点或 simulator task_low_dim_state。参考为空的输入仍保留；演示数量不代表解空间大小。事件不在采集阶段被拉平。

恢复状态机、输出 join 和角色隔离先做纯文件测试。跨进程重建能否满足严格相等必须由真实模拟器小批恢复试验确认；纯单测通过不能替代这项证据。正式作业由 root 排定 CPU 槽后启动，当前不启动。
