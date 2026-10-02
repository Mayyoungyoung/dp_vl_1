# 六任务正式采集准备：144 个预注册父场景

状态：采集器与一个 TRAIN 父的真实跨进程恢复验证完成；root 已启动固定源码的单 CPU 正式采集，完成情况以实际 job 为准。原始任务演示加入自由前接近提案，数据名称始终为 RLBench-derived；不宣称三个几何不同轨迹就是三种任务路线。

候选任务仅保留已有成功、恢复和成本证据的六项：reach_target、pick_and_lift、push_button、take_lid_off_saucepan、pick_up_cup、slide_block_to_target。旧四任务小批 4 父 12/12 成功，总 433.585 秒；来源以旧作业实际执行副本与 hash 为准，不追认后续 commit 为当时源。新四任务中 cup 三次合计 137.540 秒、slide 122.473 秒，均 3/3 成功和严格记录状态／RGB／物体清单／语言恢复。peg 215.607 秒、stack 473.091 秒，仅保留可采性证据，不纳入本轮正式任务。

每任务 24 个独立父场景，每父 3 个预声明提案，共请求 144 父、432 个路线尝试。父索引前 16 个分配 TRAIN，随后 DEV_MODEL、DEV_SCORE、CALIBRATION、TEST_LOCKED 各 2 个；语言改写共享父划分。全部角色在采集前写入不可变 partition manifest，新种子与现有 pilot 不重叠。角色在磁盘中物理分开，顶层不导出 locked 图像、路线或任务结果。

采用父级工作进程与持久化事务。每个父从独立模拟器进程构建，避免前一父残余状态。每次提案保存完整 pose/open 序列、相机、耗时、原任务成功状态及 hash；失败也是已完成尝试，不能为凑成功条数重采。已完成父在恢复时直接跳过。中断但未完成的父重新构建预注册 seed/variation 的初态，必须与已有 world、RGB、depth、相机、当前状态及语言严格一致，才继续尚未提交的尝试；不一致明确保存恢复阻塞，不覆盖参考。已开始却未完成的尝试留下 interrupted 记录，其耗时和提案代价不被抹掉。

模型入口仍为严格五字段 observations：id、parent_id、split、image、instruction。实际 depth/camera/current 状态从单独 observation 文件读取；supervision 按 id 连接完整成功路径和事件，不向生成条件加入轨迹答案、采集引导点或 simulator task_low_dim_state。参考为空的输入仍保留；演示数量不代表解空间大小。事件不在采集阶段被拉平。

恢复状态机、输出 join 和角色隔离先做纯文件测试。跨进程重建能否满足严格相等必须由真实模拟器小批恢复试验确认；纯单测通过不能替代这项证据。正式作业由 root 排定 CPU 槽后启动，当前不启动。

代码入口为 `scripts/observation_collect_multitask.py`。当前本地 8 项纯文件测试通过，覆盖分区不交叉、不重新分配既有父、源码指纹变更拒绝恢复、完成失败不补跑、成功路线 hash 核验、空参考 join、TRAIN 导出不读取 locked 内容、严格重建数值、活跃锁保护及多会话成本计入。每次 worker 从开始到关闭模拟器都留独立会话记录；最终会话不能覆盖早期失败或部分执行耗时。未正常终止的会话只记挂钟上界，不能当精确计算成本。setup 失败闭合记录明确请求 3 槽、未尝试 3 槽。

variation 固定为 `parent_index % task.variation_count()`。因此角色之间可能存在不同任务 variation 分布；这是已知划分属性，不能将后续表现暗称 IID 泛化。父场景几何随机化仍以各自种子决定，不因结果重排父或 variation。

预声明的实际验证使用同一注册 TRAIN 父 `reach_target_281000`：先完成首个提案后以预声明 exit 3 结束 worker，再在新模拟器进程中重建同父参考并完成剩余两槽。仅当所有已持久化状态、RGB-D、相机、当前状态及语言严格一致时继续；预声明时这项真实重启证据尚未完成。

这项真实验证现已完成。固定 `6c4446921317fab43066d4928e7a5cdcbbf340d1` 在服务器先通过 8 项纯文件测试及 shell 语法检查，随后首 worker PID 281072 完成 slot 0 并按预期 exit 3，实际耗时 34.976 秒；新进程 PID 281538 恢复同父并完成 slot 1、2，exit 0，58.753 秒。跨进程的 RGB、depth、pose/open、相机内外参、已记录世界状态和语言均精确相等，物体清单一致。首 slot 的记录文件 SHA 保持不变；三次尝试全部严格恢复并达到原任务成功条件。两个完整会话累计 93.728 秒分别保留，outer probe 最终 exit 0。证据见 `observation_multitask_resume_probe_v1/resume_validation.json`；这验证一个 TRAIN 父的真实重启，不扩张为所有任务已验证。

正式启动器 `scripts/launch_six_task_formal.sh` 已准备，collector 源硬固定同一 6c 版本，禁 GPU、单 CPU，继续使用已注册的 `data/observation_multitask_resume_probe_v1`。名称沿用首次恢复验证目录以避免复制或重复采集首父；其分区 manifest 从开始就是完整 144 父注册。正式阶段会跳过已闭合首父，新增物理采集到该同一语料；尚未由本 agent 启动。相机在运动阶段仍默认开启，独立性能候选不改变这一版本。

root 已固定正式 launcher 版本 `c9b3f23` 并启动 `runs/observation_multitask_six_formal_v1`，collector 仍固定 6c，单 CPU 持续采集。后续完成情况应读取实际 job 和机械状态，不能从启动记录推断完成。性能优化只在独立训练父审计，不修改此运行作业。

旧四任务实际成功源副本 `observation_collect_rlbench.py` 的 SHA256 为 `0a86f823874381ba7dd822774a5324e4a550d6855f82a932b9f336b748d916c3`，对应 wrapper SHA256 为 `3157ff9224559c625d65721ec18c75d7836fba48805604f0ea07e3068b266cb8`。按旧 4 任务总耗时及 cup/slide 小批线性推算，六任务每类 24 父约需 4.6 个 CPU 进程小时，尚未计不同 variation、独立进程启动和失败重建的波动；仅作排程估计，不是实测正式成本。

独立性能候选仅完成源码核验：已安装 RLBench `backend/scene.py` SHA256 为 `d95fe8c900d9a5f7efc77bf7cd3d8697514ddbc59d3fa885b60523b4695d690d`。逐步观测会根据当前相机开关执行 RGB-D 渲染，pose/open 则独立读取。可在严格初态图像核验后临时关闭 front RGB-D，并在下一次恢复校验前开启；不得调用会删除 sensor 的 `_set_camera_properties()`。正式默认尚未采用此优化；需要相同 TRAIN 父和提案的独立 on/off 实测及全部额外尝试预算后再决定。

性能候选 harness 已实现为 `scripts/benchmark_demo_render.py`，仅在 proposal 0 的 `Scene.get_demo` 调用期间临时切换相机配置，初始／严格恢复图像以及 pose/open 采样语义保留。固定 ed710 的 TRAIN 首父 on/off 实测已完成：初态精确相等、55 步 pose/open 逐值相等、两臂原任务均成功，worker 从 28.036 秒降至 10.532 秒。额外两提案独立归档。详细成本、源码与限制见 `RLBENCH_DEMO_RENDER_PERFORMANCE.md`。下一步已授权其余五个任务首 TRAIN 父各一组 on/off 验证，尚待固定新版本后运行。它不覆盖自由前缀阶段，也不会自动采用到正式采集。
